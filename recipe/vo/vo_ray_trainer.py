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
VO (Value Optimization) / VPO (Value-Policy Optimization) Trainer with Ray-based single controller.

This trainer extends DAPO trainer with RLVR-VO objectives:
1. Value-based loss using GAE from constructed values
2. Optional GRPO-style actor loss
3. Periodic reference model updates (every N steps)

Reference: Intuitor RLVR-VO implementation
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip

from .vo_core_algos import (
    compute_vo_values_and_advantages,
    sync_ref_model_from_policy,
    compute_ref_sync_diff,
)
from .losses import precompute_adaptive_beta_from_old_policy


class RayVOTrainer(RayPPOTrainer):
    """
    Value Optimization (VO) / Value-Policy Optimization (VPO) Trainer.
    
    This trainer implements RLVR-VO training which uses:
    1. Value-based loss with GAE from constructed values (Eq.6 recurrence)
    2. Optional GRPO-style actor loss for VPO
    3. Periodic reference model updates
    
    Key differences from standard PPO/DAPO:
    - Values are constructed via recurrence, not predicted by a separate critic
    - v0_hat is computed as group MC estimate
    - Reference model is updated every N steps instead of being fixed
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # VO-specific configuration
        vo_config = self.config.get("vo", {})
        self.vo_gamma = vo_config.get("gamma", 1.0)
        self.vo_beta = vo_config.get("beta", 0.005)
        self.vo_gae_lambda = vo_config.get("gae_lambda", 0.95)
        self.vo_cliprange_value = vo_config.get("cliprange_value", 0.2)
        self.vo_use_grpo_actor = vo_config.get("use_grpo_actor", False)  # Default OFF
        self.vo_use_ppo_actor = vo_config.get("use_ppo_actor", False)    # Default OFF
        self.vo_use_final_value_loss = vo_config.get("use_final_value_loss", False)
        self.vo_grpo_coef = vo_config.get("grpo_coef", 0.5)
        self.vo_cliprange_ppo = vo_config.get("cliprange_ppo", 0.2)
        self.vo_adaptive_beta = bool(vo_config.get("adaptive_beta", False))
        self.vo_adaptive_beta_scope = str(vo_config.get("adaptive_beta_scope", "per_prompt_batch"))
        self.vo_beta_upper_bound = float(vo_config.get("beta_upper_bound", 1.0))
        # KL mode: "exact" (default) or "candidate_set"
        self.vo_kl_mode = vo_config.get("kl_mode", "exact")
        self.vo_exact_kl_vocab_chunk_size = int(vo_config.get("exact_kl_vocab_chunk_size", 0))
        self.vo_candidate_kl_topk = vo_config.get("candidate_kl_topk", 128)
        self.vo_candidate_kl_M = vo_config.get("candidate_kl_M", 0)
        if self.vo_kl_mode not in ("exact", "candidate_set"):
            raise ValueError(
                "VO trainer does not support kl_mode=approximate; use exact or candidate_set."
            )

        # Reference model update configuration
        self.update_ref_per_n_step = vo_config.get("update_ref_per_n_step", 20)
        self.ref_update_kl_threshold = vo_config.get("ref_update_kl_threshold", None)
        if self.ref_update_kl_threshold is not None:
            self.ref_update_kl_threshold = float(self.ref_update_kl_threshold)
        self.update_ref_buffer_step = int(vo_config.get("update_ref_buffer_step", 0))
        self.refresh_adam_upon_ref_update = bool(vo_config.get("refresh_adam_upon_ref_update", False))
        self.last_ref_update_step = 0
        self._pending_ref_buffer_updates = []
        self._ref_buffer_root = None

        # Require VO loss: fail fast if actor is not configured for VO (no silent vanilla fallback)
        pl = getattr(self.config.actor_rollout_ref.actor, "policy_loss", None)
        loss_mode = getattr(pl, "loss_mode", None) if pl is not None else None
        if loss_mode != "vo":
            raise RuntimeError(
                "VO trainer requires actor policy_loss.loss_mode=vo. "
                "Currently loss_mode=%r. Add actor_rollout_ref.actor.policy_loss.loss_mode=vo in config or run_vo_dapo.sh."
                % (loss_mode,)
            )

        print(f"[VO Trainer] Initialized with config:")
        print(f"  - gamma: {self.vo_gamma}")
        print(f"  - beta: {self.vo_beta}")
        print(f"  - gae_lambda: {self.vo_gae_lambda}")
        print(f"  - cliprange_value: {self.vo_cliprange_value}")
        print(f"  - use_grpo_actor: {self.vo_use_grpo_actor}")
        print(f"  - use_ppo_actor: {self.vo_use_ppo_actor}")
        print(f"  - use_final_value_loss: {self.vo_use_final_value_loss}")
        print(f"  - adaptive_beta: {self.vo_adaptive_beta}")
        print(f"  - adaptive_beta_scope: {self.vo_adaptive_beta_scope}")
        print(f"  - beta_upper_bound: {self.vo_beta_upper_bound}")
        print(f"  - grpo_coef: {self.vo_grpo_coef}")
        print(f"  - update_ref_per_n_step: {self.update_ref_per_n_step}")
        print(f"  - ref_update_kl_threshold: {self.ref_update_kl_threshold}")
        print(f"  - update_ref_buffer_step: {self.update_ref_buffer_step}")
        print(f"  - refresh_adam_upon_ref_update: {self.refresh_adam_upon_ref_update}")
    
    def init_workers(self):
        """
        Override parent's init_workers to use role="actor_rollout_ref" for the ActorRollout worker.
        This enables both actor and ref capabilities in the same worker, required for compute_log_prob_ref_and_kl.
        """
        from omegaconf import OmegaConf
        from verl.single_controller.ray import RayClassWithInitArgs
        from verl.single_controller.ray.base import create_colocated_worker_cls
        from verl.trainer.ppo.ray_trainer import Role
        from verl.utils.config import omega_conf_to_dataclass
        
        use_combined_actor_ref = self.vo_kl_mode in ("exact", "candidate_set")
        actor_role = "actor_rollout_ref" if use_combined_actor_ref else str(Role.ActorRollout)
        
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=actor_role,
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError("VO trainer requires hybrid_engine=True")
        
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls
        
        if self.use_reference_policy and not use_combined_actor_ref:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls
        
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options") is not None
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name
        
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
        
        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()
        
        self.ref_policy_wg = None
        if self.use_reference_policy and not use_combined_actor_ref:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()
        
        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()
        
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()
        
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager
            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )
    
    def _maybe_update_ref_model(self, metrics: dict):
        """
        Update reference model from current policy.
        
        Triggers:
        1) periodic step interval (vo.update_ref_per_n_step > 0), or
        2) KL threshold (vo.ref_update_kl_threshold is set and vo/exact_kl_mean >= threshold).
        """
        if not self.use_reference_policy:
            return
        self._maybe_apply_buffered_ref_model(metrics)

        step_triggered = (
            self.update_ref_per_n_step > 0
            and (self.global_steps - self.last_ref_update_step) >= self.update_ref_per_n_step
        )

        kl_value = metrics.get("vo/exact_kl_mean", None)
        kl_triggered = (
            self.ref_update_kl_threshold is not None
            and kl_value is not None
            and float(kl_value) >= self.ref_update_kl_threshold
        )

        if not (step_triggered or kl_triggered):
            return

        reason_tokens = []
        if step_triggered:
            reason_tokens.append(f"step_interval={self.update_ref_per_n_step}")
        if kl_triggered:
            reason_tokens.append(f"vo/exact_kl_mean={float(kl_value):.6f}>=threshold={self.ref_update_kl_threshold:.6f}")
        reason = ", ".join(reason_tokens)
        print(f"[VO Trainer] Updating reference model at step {self.global_steps} ({reason})")

        if hasattr(self, 'actor_rollout_wg') and hasattr(self, 'ref_policy_wg'):
            if self.update_ref_buffer_step > 0:
                if self._ref_buffer_root is None:
                    self._ref_buffer_root = os.path.join(self.config.trainer.default_local_dir, "ref_buffer_snapshots")
                    os.makedirs(self._ref_buffer_root, exist_ok=True)
                snapshot_id = uuid.uuid4().hex[:8]
                snapshot_dir = os.path.join(
                    self._ref_buffer_root, f"step_{self.global_steps:08d}_{snapshot_id}"
                )
                self.actor_rollout_wg.save_actor_model_to_ref_buffer(snapshot_dir)
                apply_step = self.global_steps + self.update_ref_buffer_step
                self._pending_ref_buffer_updates.append(
                    {
                        "snapshot_dir": snapshot_dir,
                        "trigger_step": int(self.global_steps),
                        "apply_step": int(apply_step),
                    }
                )
            else:
                # Request sync from actor to ref immediately
                self.actor_rollout_wg.sync_ref_model()
                if self.refresh_adam_upon_ref_update:
                    self.actor_rollout_wg.refresh_adam_moments_keep_step()

        self.last_ref_update_step = self.global_steps
        immediate_update_flag = 0.0 if self.update_ref_buffer_step > 0 else 1.0
        metrics["vo/ref_model_updated"] = max(float(metrics.get("vo/ref_model_updated", 0.0)), immediate_update_flag)

    def _maybe_apply_buffered_ref_model(self, metrics: dict):
        """
        Apply queued delayed reference updates when their apply_step is reached.
        """
        if self.update_ref_buffer_step <= 0:
            return
        if not self._pending_ref_buffer_updates:
            return

        applied = 0
        while self._pending_ref_buffer_updates and self._pending_ref_buffer_updates[0]["apply_step"] <= self.global_steps:
            entry = self._pending_ref_buffer_updates.pop(0)
            self.actor_rollout_wg.load_ref_model_from_ref_buffer(entry["snapshot_dir"])
            applied += 1
            metrics["vo/ref_model_updated"] = 1.0
            if self.refresh_adam_upon_ref_update:
                self.actor_rollout_wg.refresh_adam_moments_keep_step()
    
    def _compute_vo_data(self, batch: DataProto, metrics: dict) -> DataProto:
        """
        Compute VO-specific data: v0_hat, values_old, advantages.
        Log VO metrics for wandb, aligned with Intuitor RLVR-VO trainer style.
        """
        batch = compute_vo_values_and_advantages(
            data=batch,
            gamma=self.vo_gamma,
            gae_lambda=self.vo_gae_lambda,
            beta=self.vo_beta,
        )
        response_mask = batch.batch["response_mask"].bool()

        # VO metrics for wandb (mirror Intuitor rlvr_vo_trainer)
        if "v0_hat" in batch.batch:
            v0_hat = batch.batch["v0_hat"]
            metrics["vo/v0_hat_mean"] = v0_hat.mean().item()
            metrics["vo/v0_hat_std"] = v0_hat.std().item() if v0_hat.numel() > 1 else 0.0
        if "final_rewards" in batch.batch:
            final_rewards = batch.batch["final_rewards"]
            metrics["vo/final_reward_mean"] = final_rewards.mean().item()
            metrics["vo/final_reward_std"] = final_rewards.std().item() if final_rewards.numel() > 1 else 0.0
        if "advantages" in batch.batch:
            adv = batch.batch["advantages"]
            denom = response_mask.float().sum().clamp(min=1.0)
            adv_mean = (adv * response_mask.float()).sum().item() / denom.item()
            adv_sq = ((adv - adv_mean) ** 2 * response_mask.float()).sum().item() / denom.item()
            metrics["vo/advantage_mean"] = adv_mean
            metrics["vo/advantage_std"] = adv_sq ** 0.5
        if "returns" in batch.batch:
            ret = batch.batch["returns"]
            denom = response_mask.float().sum().clamp(min=1.0)
            ret_mean = (ret * response_mask.float()).sum() / denom
            ret_var = ((ret - ret_mean) ** 2 * response_mask.float()).sum() / denom
            metrics["vo/returns_mean"] = ret_mean.item()
            metrics["vo/returns_std"] = ret_var.clamp(min=0).sqrt().item()
        if "values_old" in batch.batch:
            vold = batch.batch["values_old"]
            denom = response_mask.float().sum().clamp(min=1.0)
            metrics["vo/values_old_mean"] = (vold * response_mask.float()).sum().item() / denom.item()
        # values_old_final_rms: rollout-time RMS at final token vs true reward.
        if "values_old" in batch.batch and "final_rewards" in batch.batch:
            vold = batch.batch["values_old"]
            true_reward = batch.batch["final_rewards"]  # (Bsz,) true outcome per sequence
            response_lengths = response_mask.sum(dim=-1, dtype=torch.long)
            t_last = response_lengths.clamp(min=1) - 1  # (Bsz,)
            bsz = vold.shape[0]
            v_at_final = vold[torch.arange(bsz, device=vold.device), t_last]  # (Bsz,)
            err_sq = ((v_at_final - true_reward.to(vold.dtype)) ** 2).mean()
            metrics["vo/values_old_final_rms"] = err_sq.clamp(min=0).sqrt().item()
        # vo/exact_kl_mean: mask-weighted mean of kl_per_token (when vo_kl_mode is exact/candidate_set)
        if "kl_per_token" in batch.batch:
            kpt = batch.batch["kl_per_token"].float() * response_mask.float()
            denom = response_mask.float().sum().clamp(min=1.0)
            metrics["vo/exact_kl_mean"] = (kpt.sum() / denom).item()

        return batch

    def _maybe_precompute_global_adaptive_beta(self, batch: DataProto, metrics: dict) -> DataProto:
        """
        Precompute adaptive-beta tensors once on the trainer's full repeated batch.

        The actor workers receive only local shards. Doing the solve here gives `mini_batch`
        and `per_prompt_batch` their natural global semantics for the current training step.
        """
        if not self.vo_adaptive_beta:
            return batch
        if not self.vo_use_final_value_loss:
            raise ValueError("vo.adaptive_beta=True requires vo.use_final_value_loss=True.")

        uid = batch.non_tensor_batch.get("uid", None)
        beta_tensor, beta_raw_tensor = precompute_adaptive_beta_from_old_policy(
            data=batch.batch,
            uid=uid,
            gamma=float(self.vo_gamma),
            base_beta=float(self.vo_beta),
            beta_upper_bound=float(self.vo_beta_upper_bound),
            scope=self.vo_adaptive_beta_scope,
        )
        batch.batch["vo_beta"] = beta_tensor
        batch.batch["vo_adaptive_beta_raw"] = beta_raw_tensor
        metrics["vo/beta_precompute_global"] = float(beta_tensor.float().mean().item())
        metrics["vo/adaptive_beta_raw_precompute_global"] = float(beta_raw_tensor.float().mean().item())
        return batch
    
    def fit(self):
        """
        The training loop for VO/VPO.
        Extends DAPO training with VO-specific value computation and reference model updates.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="VO Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)
                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    # True accuracy metrics BEFORE filtering.
                    # Log regardless of whether filter_groups is enabled.
                    score_key = "acc" if "acc" in new_batch.non_tensor_batch else "score"
                    if score_key in new_batch.non_tensor_batch:
                        try:
                            raw_scores = new_batch.non_tensor_batch[score_key]
                            uids = new_batch.non_tensor_batch["uid"]
                            prompt_scores = {}
                            for uid, sc in zip(uids, raw_scores):
                                v = float(sc)
                                if uid not in prompt_scores:
                                    prompt_scores[uid] = []
                                prompt_scores[uid].append(v)
                            all_scores = list(raw_scores)
                            all_correct = sum(
                                1 for scores in prompt_scores.values()
                                if all(abs(s - 1.0) < 1e-6 for s in scores)
                            )
                            all_incorrect = sum(
                                1 for scores in prompt_scores.values()
                                if all(abs(s - 0.0) < 1e-6 for s in scores)
                            )
                            total_prompts = len(prompt_scores)
                            metrics.update({
                                "true_accuracy/score_mean": float(np.mean(all_scores)) if all_scores else 0.0,
                                "true_accuracy/score_std": float(np.std(all_scores)) if all_scores else 0.0,
                                "true_accuracy/all_correct_ratio": float(all_correct / total_prompts) if total_prompts > 0 else 0.0,
                                "true_accuracy/all_incorrect_ratio": float(all_incorrect / total_prompts) if total_prompts > 0 else 0.0,
                                "true_accuracy/all_correct_count": int(all_correct),
                                "true_accuracy/all_incorrect_count": int(all_incorrect),
                                "true_accuracy/total_prompts": int(total_prompts),
                            })
                        except Exception as e:
                            print(f"[WARNING] Failed to compute true_accuracy metrics: {e}")

                    # DAPO-style filtering
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )
                        if metric_name not in new_batch.non_tensor_batch:
                            raise ValueError(
                                f"filter_groups.metric={metric_name!r} not found in non_tensor_batch. "
                                f"Available keys: {list(new_batch.non_tensor_batch.keys())}"
                            )

                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many."
                                )
                        else:
                            # Keep whole prompt groups (by uid) to avoid truncating partial groups.
                            # Partial-group truncation silently corrupts group-based statistics.
                            prompt_bsz = self.config.data.train_batch_size
                            selected_prompt_uids = []
                            selected_prompt_uid_set = set()
                            for uid in batch.non_tensor_batch["uid"]:
                                if uid not in selected_prompt_uid_set:
                                    selected_prompt_uids.append(uid)
                                    selected_prompt_uid_set.add(uid)
                                    if len(selected_prompt_uids) >= prompt_bsz:
                                        break
                            keep_idxs = [
                                idx
                                for idx, uid in enumerate(batch.non_tensor_batch["uid"])
                                if uid in selected_prompt_uid_set
                            ]
                            batch = batch[keep_idxs]

                    # === Updating ===
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance batch
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs (and optionally ref + kl_per_token in one pass)
                    use_vo_kl_combined = (
                        self.vo_kl_mode in ("exact", "candidate_set")
                        and self.use_reference_policy
                    )
                    if use_vo_kl_combined:
                        batch.meta_info["vo_kl_mode"] = self.vo_kl_mode
                        batch.meta_info["vo_exact_kl_vocab_chunk_size"] = self.vo_exact_kl_vocab_chunk_size
                        batch.meta_info["vo_candidate_kl_topk"] = self.vo_candidate_kl_topk
                        batch.meta_info["vo_candidate_kl_M"] = self.vo_candidate_kl_M
                        with marked_timer("old_log_prob", timing_raw, "blue"):
                            out = self.actor_rollout_wg.compute_log_prob_ref_and_kl(batch)
                            if "entropys" in out.batch:
                                entropys = out.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_agg = agg_loss(
                                    loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                                )
                                metrics["actor/entropy"] = entropy_agg.detach().item()
                                out.batch.pop("entropys")
                            batch = batch.union(out)
                            if "kl_per_token" not in batch.batch:
                                raise ValueError(
                                    "VO trainer requires 'kl_per_token' from compute_log_prob_ref_and_kl "
                                    "when vo.kl_mode is exact/candidate_set."
                                )
                    else:
                        raise RuntimeError(
                            "VO trainer only supports exact/candidate_set KL modes. "
                            "Received unsupported execution path."
                        )

                    # compute values (if critic is used)
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout IS weights and mismatch metrics
                    batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                    metrics.update(is_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        # ===== VO-specific: Compute values and advantages =====
                        batch = self._compute_vo_data(batch, metrics)
                        
                        # Also compute standard GRPO advantage for actor loss
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        use_sce_as_reward = self.config.algorithm.get("use_sce_as_reward", False)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            use_sce_as_reward=use_sce_as_reward,
                        )
                        batch = self._maybe_precompute_global_adaptive_beta(batch, metrics)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # ===== VO-specific: Maybe update reference model =====
                    self._maybe_update_ref_model(metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # True accuracy from reward_extra_info (same as run_entropy_tokens_dapo / DAPO)
                nt = getattr(batch, "non_tensor_batch", None)
                if nt is not None:
                    if "acc" in nt:
                        metrics["train/acc_mean"] = float(np.mean(nt["acc"]))
                    elif "score" in nt:
                        metrics["train/acc_mean"] = float(np.mean(nt["score"]))
                timing_raw = defaultdict(float)

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        
        # Save final checkpoint
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
