# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Main entry point for Value Optimization (VO) / Value-Policy Optimization (VPO) training.

This extends the standard PPO training with RLVR-VO objectives:
- Value-based loss using GAE from constructed values
- Optional GRPO-style actor loss
- Periodic reference model updates (every N steps)

Usage:
    python -m recipe.vo.main_vo \
        data.train_files=... \
        data.val_files=... \
        vo.gamma=1.0 \
        vo.beta=0.005 \
        vo.update_ref_per_n_step=20 \
        ...
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type


@hydra.main(config_path="config", config_name="vo_trainer", version_base=None)
def main(config):
    """Main entry point for VO training with Hydra configuration management."""
    run_vo(config)


def run_vo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed VO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed VO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        if config.transfer_queue.enable:
            ray_init_kwargs["TRANSFER_QUEUE_ENABLE"] = "1"
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(VOTaskRunner)

    # Create and run the task runner
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class VOTaskRunner:
    """Ray remote class for executing distributed VO training tasks."""

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)

        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import CriticWorker

            critic_cls = CriticWorker
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker

            critic_cls = CriticWorker
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(critic_cls)

        return critic_cls

    def add_ref_policy_worker(self, config, ref_policy_cls=None):
        """Add reference policy worker.
        
        The reference policy uses the same worker class as the actor (ActorRolloutRefWorker).
        This is consistent with how main_ppo.py handles reference policies.
        """
        from verl.trainer.ppo.ray_trainer import Role
        
        if ref_policy_cls is None:
            # Use the same class as actor if not provided
            if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import ActorRolloutRefWorker
                ref_policy_cls = ActorRolloutRefWorker
            elif config.actor_rollout_ref.actor.strategy == "megatron":
                from verl.workers.megatron_workers import ActorRolloutRefWorker
                ref_policy_cls = ActorRolloutRefWorker
            else:
                raise NotImplementedError

        self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
        return ref_policy_cls

    def add_reward_model_worker(self, config):
        """Add reward model worker."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import RewardModelWorker

            rm_cls = RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker

            rm_cls = RewardModelWorker
        elif config.reward_model.strategy == "sglang":
            from verl.workers.roles.reward_model_engine.sglang_reward_model import RewardModelWorker

            rm_cls = RewardModelWorker
        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.RewardModel] = ray.remote(rm_cls)

        return rm_cls

    def run(self, config):
        """Execute the VO training process.

        Args:
            config: Complete configuration object for training.
        """
        from transformers import AutoTokenizer

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        # Import VO trainer
        from recipe.vo.vo_ray_trainer import RayVOTrainer

        # Print host information
        local_host_name = socket.gethostname()
        print(f"Hostname: {local_host_name}")

        # Determine backend
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)

        use_critic = need_critic(config)
        # VO algorithm ALWAYS requires reference policy for computing logratio = log_prob - ref_log_prob
        # Without ref_log_prob, the gradient flow is broken (logratio becomes zeros_like with no grad)
        use_reference_policy = True  # Force True for VO training

        if use_critic:
            critic_cls = self.add_critic_worker(config)

        # Add RefPolicy to the role_worker_mapping so it's available for resource pool setup.
        # Note: The VO trainer's _init_worker_group may choose to NOT create a separate RefPolicy
        # worker if using exact/candidate_set KL mode (it uses actor_rollout_ref role instead).
        ref_cls = self.add_ref_policy_worker(config, ref_policy_cls=actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=use_reference_policy,
            use_critic=use_critic,
        )

        # Load tokenizer (and processor) first — required by load_reward_manager
        tokenizer_path = config.actor_rollout_ref.model.path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        processor = None
        if OmegaConf.select(config.actor_rollout_ref.model, "processor_path") is not None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                config.actor_rollout_ref.model.processor_path, trust_remote_code=True
            )

        # Reward: same as DAPO — reward_model.enable only gates the neural RM *worker*;
        # we always load reward_fn/val_reward_fn (rule-based reward_manager, e.g. dapo).
        if config.reward_model.enable and config.reward_model.get("strategy") is not None:
            rm_cls = self.add_reward_model_worker(config)
        reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=0,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
            **config.reward_model.get("reward_kwargs", {}),
        )
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=1,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
            **config.reward_model.get("reward_kwargs", {}),
        )

        # Setup resource pools
        resource_pool_spec = {}
        mapping_for_rp = {}

        resource_pool_spec["actor_rollout"] = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        mapping_for_rp[Role.ActorRollout] = "actor_rollout"

        if use_critic:
            if config.critic.resource_pool == "actor_rollout":
                mapping_for_rp[Role.Critic] = "actor_rollout"
            else:
                resource_pool_spec["critic"] = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
                mapping_for_rp[Role.Critic] = "critic"

        if use_reference_policy:
            # Use .get() to safely access optional config key
            ref_resource_pool = config.actor_rollout_ref.ref.get("resource_pool", "actor_rollout")
            if ref_resource_pool == "actor_rollout":
                mapping_for_rp[Role.RefPolicy] = "actor_rollout"
            else:
                resource_pool_spec["ref"] = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
                mapping_for_rp[Role.RefPolicy] = "ref"

        if config.reward_model.enable and config.reward_model.get("strategy") is not None:
            if config.reward_model.resource_pool == "actor_rollout":
                mapping_for_rp[Role.RewardModel] = "actor_rollout"
            else:
                resource_pool_spec["reward_model"] = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
                mapping_for_rp[Role.RewardModel] = "reward_model"

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping_for_rp,
        )

        # Dataset and sampler
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("val_max_samples", -1),
        )

        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Device name
        device_name = config.trainer.get("device", None)

        # Create VO trainer
        trainer = RayVOTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_sampler=train_sampler,
            device_name=device_name,
        )

        # Initialize workers and run training
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_files, data_config, tokenizer, processor=None, max_samples=-1):
    """Create RL dataset from files."""
    from verl.utils.dataset.rl_dataset import RLDataset

    return RLDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=data_config.get("prompt_key", "prompt"),
        max_prompt_length=data_config.get("max_prompt_length", 1024),
        truncation=data_config.get("truncation", "left"),
        max_samples=max_samples,
    )


def create_rl_sampler(data_config, dataset):
    """Create RL sampler for dataset."""
    sampler_config = data_config.get("sampler", None)
    if sampler_config is None:
        return None

    sampler_type = sampler_config.get("type", "random")
    if sampler_type == "random":
        return None  # Use default random sampling

    # Load custom sampler
    sampler_cls_path = sampler_config.get("class_path")
    if sampler_cls_path:
        sampler_cls = load_extern_type(sampler_cls_path)
        return sampler_cls(dataset, **sampler_config.get("kwargs", {}))

    return None


if __name__ == "__main__":
    main()
