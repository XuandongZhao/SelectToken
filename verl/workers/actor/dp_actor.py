# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    get_global_entropy_top_mask,
    get_probability_based_mask,
    get_entropy_probability_mask,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.utils.attention_utils import (
    index_first_axis,
    pad_input,
    rearrange,
    unpad_input,
)
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        
        # Set up probability metrics computation functions (following entropy pattern)
        prob_metrics_with_chunking = self.config.get("prob_metrics_from_logits_with_chunking", False)
        if prob_metrics_with_chunking:
            max_probs_from_logits = verl_F.max_probs_from_logits_with_chunking
            sum_of_squares_from_logits = verl_F.sum_of_squares_from_logits_with_chunking
        else:
            max_probs_from_logits = verl_F.max_probs_from_logits
            sum_of_squares_from_logits = verl_F.sum_of_squares_from_logits
        
        self.compute_max_probs_from_logits = (
            torch.compile(max_probs_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)
            else max_probs_from_logits
        )
        self.compute_sum_of_squares_from_logits = (
            torch.compile(sum_of_squares_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)
            else sum_of_squares_from_logits
        )
        
        # Set up self-certainty score computation (following entropy pattern)
        certainty_with_chunking = self.config.get("self_certainty_with_chunking", False)
        if certainty_with_chunking:
            self_certainty_score_fn = verl_F.self_certainty_score_with_chunking
        else:
            self_certainty_score_fn = verl_F.self_certainty_score
        
        self.compute_self_certainty_score = (
            torch.compile(self_certainty_score_fn, dynamic=True)
            if self.config.get("use_torch_compile", True)
            else self_certainty_score_fn
        )
        
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        compute_prob_metrics=False,
        return_response_logits=False,
        response_logits_format: str = "dense",
    ):
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            prob_metrics: # dict or None
            response_logits:
                - dense: (bs, response_len, V)
                - packed: dict with keys:
                    - "format": "packed"
                    - "logits": (N_valid, V)
                    - "index": (bs, response_len) with -1 for invalid positions
                only when return_response_logits=True and not use_fused_kernels
        """
        response_length = micro_batch["responses"].size(-1)
        response_logits_out = None
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import (
                        process_multi_modal_inputs_for_minicpmo,
                    )

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # Compute probability metrics if requested
                    max_probs_rmpad = None
                    sum_of_squares_rmpad = None
                    self_certainty_rmpad = None
                    if compute_prob_metrics:
                        # DEBUG: Check logit statistics
                        if torch.distributed.get_rank() == 0:
                            sample_logits = logits_rmpad[0]  # First token
                            print(
                                f"[DEBUG] Logits stats - max: {sample_logits.max().item():.2f}, "
                                f"min: {sample_logits.min().item():.2f}, "
                                f"mean: {sample_logits.mean().item():.2f}, "
                                f"std: {sample_logits.std().item():.2f}"
                            )

                        max_probs_rmpad = self.compute_max_probs_from_logits(logits_rmpad)
                        sum_of_squares_rmpad = self.compute_sum_of_squares_from_logits(logits_rmpad)
                        # Self-certainty score: logsumexp(logits) - mean(logits)
                        self_certainty_rmpad = self.compute_self_certainty_score(logits_rmpad)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                # Pad prob_metrics if computed
                prob_metrics = None
                if compute_prob_metrics:
                    full_max_probs = pad_input(
                        hidden_states=max_probs_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_sum_of_squares = pad_input(
                        hidden_states=sum_of_squares_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_self_certainty = pad_input(
                        hidden_states=self_certainty_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    prob_metrics = {
                        'max_probs': full_max_probs.squeeze(-1)[:, -response_length - 1 : -1],
                        'sum_of_squares': full_sum_of_squares.squeeze(-1)[:, -response_length - 1 : -1],
                        'self_certainty': full_self_certainty.squeeze(-1)[:, -response_length - 1 : -1],
                    }

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if return_response_logits and not self.use_fused_kernels:
                    # Build an inverse map from flattened (B*S) positions -> unpadded row index.
                    total_slots = batch_size * seqlen
                    inv_idx = torch.full((total_slots,), -1, device=indices.device, dtype=torch.long)
                    inv_idx[indices] = torch.arange(indices.numel(), device=indices.device, dtype=torch.long)

                    # Response logits correspond to dense positions [:, -response_length-1:-1].
                    response_start = seqlen - response_length - 1
                    response_pos = torch.arange(
                        response_start, response_start + response_length, device=indices.device, dtype=torch.long
                    )
                    batch_base = (
                        torch.arange(batch_size, device=indices.device, dtype=torch.long).unsqueeze(1) * seqlen
                    )
                    response_flat_pos = (batch_base + response_pos.unsqueeze(0)).reshape(-1)
                    response_rmpad_idx = inv_idx[response_flat_pos].reshape(batch_size, response_length)
                    valid = response_rmpad_idx >= 0

                    if response_logits_format == "packed":
                        if valid.any():
                            packed_logits = logits_rmpad.index_select(0, response_rmpad_idx[valid])
                        else:
                            packed_logits = logits_rmpad.new_empty((0, logits_rmpad.size(-1)))
                        response_logits_out = {
                            "format": "packed",
                            "logits": packed_logits.detach(),
                            "index": response_rmpad_idx.detach(),
                        }
                    else:
                        vocab_size = logits_rmpad.size(-1)
                        response_logits = torch.zeros(
                            (batch_size, response_length, vocab_size),
                            device=logits_rmpad.device,
                            dtype=logits_rmpad.dtype,
                        )
                        if valid.any():
                            gathered = logits_rmpad.index_select(0, response_rmpad_idx[valid])
                            response_logits[valid] = gathered
                        response_logits_out = response_logits.detach()

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    if return_response_logits:
                        response_logits_out = logits.detach()

                    # Compute probability metrics if requested
                    prob_metrics = None
                    if compute_prob_metrics:
                        max_probs = self.compute_max_probs_from_logits(logits)
                        sum_of_squares = self.compute_sum_of_squares_from_logits(logits)
                        self_certainty = self.compute_self_certainty_score(logits)
                        prob_metrics = {
                            'max_probs': max_probs,
                            'sum_of_squares': sum_of_squares,
                            'self_certainty': self_certainty,
                        }
                    
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            if return_response_logits:
                return entropy, log_probs, prob_metrics, response_logits_out
            return entropy, log_probs, prob_metrics

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        self_certainty_lst = []
        compute_self_certainty = self.config.get('compute_self_certainty', False)
        
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, prob_metrics = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                    compute_prob_metrics=compute_self_certainty
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if compute_self_certainty and prob_metrics is not None:
                self_certainty_lst.append(prob_metrics['self_certainty'])

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        self_certainty = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if compute_self_certainty and len(self_certainty_lst) > 0:
            self_certainty = torch.concat(self_certainty_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if compute_self_certainty and self_certainty is not None:
                self_certainty = restore_dynamic_batch(self_certainty, batch_idx_list)

        return log_probs, entropys, self_certainty

    def compute_log_prob_and_response_logits_for_micro_batch(
        self,
        micro_batch: dict,
        temperature: float,
        return_entropy: bool = False,
        response_logits_format: str = "dense",
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Run one micro_batch and return (log_probs, response_logits) or (log_probs, response_logits, entropy).
        response_logits is (B, L_resp, V) or None when use_fused_kernels (exact/candidate KL need logits).
        Used by VO worker to compute exact/candidate-set KL per micro-batch without storing full batch logits.
        """
        self.actor_module.eval()
        with torch.no_grad():
            entropy, log_probs, _, response_logits = self._forward_micro_batch(
                micro_batch,
                temperature=temperature,
                calculate_entropy=return_entropy,
                compute_prob_metrics=False,
                return_response_logits=True,
                response_logits_format=response_logits_format,
            )
        if return_entropy:
            return log_probs, response_logits, entropy
        return log_probs, response_logits

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # VO loss needs values_old, v0_hat, final_rewards, and kl_per_token (for vo/exact_kl_mean)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if loss_mode == "vo":
            for key in (
                "values_old",
                "v0_hat",
                "final_rewards",
                "kl_per_token",
                "ref_log_prob",
                "vo_beta",
                "vo_adaptive_beta_raw",
            ):
                if key in data.batch.keys() and key not in select_keys:
                    select_keys.append(key)
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if loss_mode == "vo" and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if loss_mode == "vo":
                    from recipe.vo.losses import precompute_adaptive_beta_from_old_policy

                    vo_config = getattr(self.config, "vo_config", {}) or {}
                    adaptive_beta = bool(vo_config.get("adaptive_beta", getattr(self.config, "vo_adaptive_beta", False)))
                    if adaptive_beta:
                        if not vo_config.get(
                            "use_final_value_loss", getattr(self.config, "vo_use_final_value_loss", False)
                        ):
                            raise ValueError("vo.adaptive_beta=True requires vo.use_final_value_loss=True.")
                        has_precomputed = (
                            "vo_beta" in mini_batch.batch and "vo_adaptive_beta_raw" in mini_batch.batch
                        )
                        if not has_precomputed:
                            base_beta = float(vo_config.get("beta", getattr(self.config, "vo_beta", 0.005)))
                            beta_upper_bound = float(vo_config.get("beta_upper_bound", 1.0))
                            gamma = float(vo_config.get("gamma", getattr(self.config, "vo_gamma", 1.0)))
                            adaptive_beta_scope = str(vo_config.get("adaptive_beta_scope", "per_prompt_batch"))
                            uid = mini_batch.non_tensor_batch.get("uid", None)
                            beta_tensor, beta_raw_tensor = precompute_adaptive_beta_from_old_policy(
                                data=mini_batch.batch,
                                uid=uid,
                                gamma=gamma,
                                base_beta=base_beta,
                                beta_upper_bound=beta_upper_bound,
                                scope=adaptive_beta_scope,
                            )
                            mini_batch.batch["vo_beta"] = beta_tensor
                            mini_batch.batch["vo_adaptive_beta_raw"] = beta_raw_tensor

                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    entropy_top_ratio = self.config.get('entropy_top_ratio', None)
                    if entropy_top_ratio is not None and not (0 <= entropy_top_ratio <= 1):
                        raise ValueError(f"Invalid {entropy_top_ratio=}")
                    
                    # Check mask mode
                    mask_mode = self.config.get('mask_mode', 'entropy')  # 'entropy', 'probability', or 'entropy-probability'
                    max_prob_threshold = self.config.get('max_prob_threshold', 0.5)
                    
                    if entropy_coeff != 0 or entropy_top_ratio is not None:
                        calculate_entropy = True
                    
                    # Compute prob metrics when using probability mask, entropy-probability mask, or when configured
                    # This also computes self-certainty score for logging
                    compute_prob_metrics = (mask_mode in ['probability', 'entropy-probability']) or self.config.get('compute_self_certainty', True)
                    
                    entropy, log_prob, prob_metrics = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                        compute_prob_metrics=compute_prob_metrics
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]
                    
                    entropy_top_mask = None
                    if mask_mode == 'entropy' and entropy_top_ratio is not None:
                        entropy_top_mask = get_global_entropy_top_mask(entropy=entropy, response_mask=response_mask, top_ratio=entropy_top_ratio)
                    elif mask_mode == 'probability' and prob_metrics is not None:
                        entropy_top_mask = get_probability_based_mask(
                            log_prob=log_prob,
                            max_probs=prob_metrics['max_probs'],
                            sum_of_squares=prob_metrics['sum_of_squares'],
                            response_mask=response_mask,
                            max_prob_threshold=max_prob_threshold,
                        )
                    elif mask_mode == 'entropy-probability' and entropy_top_ratio is not None and prob_metrics is not None:
                        entropy_top_mask = get_entropy_probability_mask(
                            entropy=entropy,
                            log_prob=log_prob,
                            sum_of_squares=prob_metrics['sum_of_squares'],
                            response_mask=response_mask,
                            top_ratio=entropy_top_ratio,
                        )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # vo -> recipe.vo.losses.vo_loss (RLVR-VO value optimization)

                    if loss_mode == "vo":
                        # VO loss path: use vo_loss for value-based actor loss (RLVR-VO)
                        from recipe.vo.losses import vo_loss

                        if "values_old" not in model_inputs or "v0_hat" not in model_inputs:
                            raise ValueError(
                                "VO loss requires values_old and v0_hat in batch; "
                                "ensure the VO trainer runs _compute_vo_data before update_actor."
                            )
                        # Pass full model_inputs so no_padding_2_padding gets indices/max_seq_len/max_response_len if used
                        model_output_vo = {"log_probs": log_prob, "entropy": entropy}
                        policy_loss, vo_metrics = vo_loss(
                            self.config, model_output_vo, model_inputs, dp_group=None
                        )
                        loss = policy_loss * loss_scale_factor
                        loss.backward()
                        micro_batch_metrics.update(vo_metrics)
                        micro_batch_metrics["actor/pg_loss"] = policy_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/pg_clipfrac"] = 0.0
                        micro_batch_metrics["actor/ppo_kl"] = 0.0
                        micro_batch_metrics["actor/pg_clipfrac_lower"] = 0.0
                    else:
                        # Standard policy loss path (vanilla, gpg, clip_cov, kl_cov, etc.)
                        rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                            entropy_top_mask=entropy_top_mask,
                        )

                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(
                                loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                            )
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                        if self.config.use_kl_loss:
                            ref_log_prob = model_inputs["ref_log_prob"]
                            kld = kl_penalty(
                                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                            )
                            kl_loss = agg_loss(
                                loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                            )
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        loss = policy_loss * loss_scale_factor
                        loss.backward()

                        micro_batch_metrics.update(
                            {
                                "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                                "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                                "actor/ppo_kl": ppo_kl.detach().item(),
                                "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                            }
                        )
                    
                    # Log mask ratio if mask is used
                    if entropy_top_mask is not None:
                        if mask_mode == 'entropy':
                            mask_name = "entropy_top_mask"
                        elif mask_mode == 'probability':
                            mask_name = "probability_mask"
                        else:  # 'entropy-probability'
                            mask_name = "entropy_probability_mask"
                        mask_ratio_value = verl_F.mask_ratio(entropy_top_mask, response_mask)
                        micro_batch_metrics[f"actor/{mask_name}_ratio"] = mask_ratio_value
                    
                    # Log self-certainty score if prob_metrics are available
                    if prob_metrics is not None and 'self_certainty' in prob_metrics:
                        self_certainty = prob_metrics['self_certainty']
                        # Compute aggregated self-certainty score using seq-mean-token-mean:
                        # First mean over valid tokens within each response, then average over responses
                        self_certainty_agg = agg_loss(loss_mat=self_certainty, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")
                        # Don't scale by loss_scale_factor - self-certainty is a metric, not a loss
                        micro_batch_metrics["actor/self_certainty_score"] = self_certainty_agg.detach().item()
                    
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
