# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from collections.abc import Mapping
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.roles.utils.padding import no_padding_2_padding


def _compute_terminal_affine_components(
    *,
    v0_hat: torch.Tensor,
    logratio: torch.Tensor,
    kl: torch.Tensor,
    r: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute terminal value decomposition V_T(beta) = a + beta * b.

    a is the terminal value when beta=0. b is the terminal coefficient for beta.
    """
    if gamma == 0.0:
        raise ValueError("vo terminal-value construction requires gamma != 0.")

    dtype = v0_hat.dtype
    mask = response_mask.to(torch.bool)
    a = v0_hat
    b = torch.zeros_like(v0_hat)
    s = (logratio - kl).to(dtype=dtype)
    r_t = r.to(dtype=dtype)
    gamma_t = torch.as_tensor(float(gamma), device=v0_hat.device, dtype=dtype)

    _, seqlen = s.shape
    for t in range(seqlen):
        a_next = (a - r_t[:, t]) / gamma_t
        b_next = (b + s[:, t]) / gamma_t
        mask_t = mask[:, t]
        a = torch.where(mask_t, a_next, a)
        b = torch.where(mask_t, b_next, b)

    return a, b


def _compute_adaptive_beta_raw_closed_form(
    *,
    v0_hat: torch.Tensor,
    logratio: torch.Tensor,
    kl: torch.Tensor,
    r: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    eps: float = 1e-12,
    beta_upper_bound: float = 1.0,
) -> torch.Tensor:
    """
    Compute raw (unclamped) adaptive beta minimizing terminal-zero MSE in final-value-loss mode.

    For each sample i, terminal value is affine in beta:
        V_T^{(i)}(beta) = a_i + b_i * beta
    The batch objective is:
        L(beta) = mean_i (V_T^{(i)}(beta))^2
    Closed-form minimizer:
        beta_raw = - sum_i b_i a_i / sum_i b_i^2
    """
    if gamma == 0.0:
        raise ValueError("vo.adaptive_beta requires gamma != 0.")

    with torch.no_grad():
        a, b = _compute_terminal_affine_components(
            v0_hat=v0_hat.detach().to(dtype=torch.float64),
            logratio=logratio.detach().to(dtype=torch.float64),
            kl=kl.detach().to(dtype=torch.float64),
            r=r.detach().to(dtype=torch.float64),
            response_mask=response_mask,
            gamma=gamma,
        )

        denom = (b * b).sum()
        if denom.item() <= eps:
            return torch.tensor(beta_upper_bound, device=v0_hat.device, dtype=v0_hat.dtype)
        num = (b * a).sum()
        beta_raw = -num / denom
        if not torch.isfinite(beta_raw):
            return torch.tensor(beta_upper_bound, device=v0_hat.device, dtype=v0_hat.dtype)
        return beta_raw.to(device=v0_hat.device, dtype=v0_hat.dtype)


def _clamp_adaptive_beta(*, beta_raw: torch.Tensor, base_beta: float, beta_upper_bound: float) -> torch.Tensor:
    return beta_raw.new_tensor(min(beta_upper_bound, max(base_beta, float(beta_raw.item()))))


def _validate_adaptive_beta_scope(scope: str) -> str:
    if scope not in ("per_prompt_batch", "mini_batch"):
        raise ValueError(
            f"Unsupported vo.adaptive_beta_scope={scope!r}. Expected 'per_prompt_batch' or 'mini_batch'."
        )
    return scope


def precompute_adaptive_beta_from_old_policy(
    *,
    data: TensorDict,
    uid: Optional[np.ndarray | list[str]] = None,
    gamma: float,
    base_beta: float,
    beta_upper_bound: float,
    scope: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute adaptive beta tensors from rollout-time VO quantities."""
    scope = _validate_adaptive_beta_scope(scope)

    if "v0_hat" not in data or "old_log_probs" not in data or "ref_log_prob" not in data:
        raise ValueError(
            "Adaptive VO beta precompute requires 'v0_hat', 'old_log_probs', and 'ref_log_prob' in batch."
        )
    if "kl_per_token" not in data or "final_rewards" not in data or "response_mask" not in data:
        raise ValueError(
            "Adaptive VO beta precompute requires 'kl_per_token', 'final_rewards', and 'response_mask' in batch."
        )

    response_mask = data["response_mask"].to(torch.bool)
    old_log_prob = data["old_log_probs"]
    ref_log_prob = data["ref_log_prob"].to(old_log_prob.device, dtype=old_log_prob.dtype)
    kl = data["kl_per_token"].to(old_log_prob.device, dtype=old_log_prob.dtype)
    v0_hat = data["v0_hat"].to(old_log_prob.device)
    final_rewards = data["final_rewards"].to(old_log_prob.device, dtype=old_log_prob.dtype)

    response_lengths = response_mask.sum(dim=-1, dtype=torch.long)
    t_last = response_lengths.clamp(min=1) - 1
    r = torch.zeros_like(old_log_prob)
    r.scatter_(1, t_last.unsqueeze(1), final_rewards.unsqueeze(1))
    r = r * response_mask
    old_logratio = old_log_prob - ref_log_prob

    bsz = old_log_prob.shape[0]
    beta = old_log_prob.new_empty((bsz,))
    beta_raw = old_log_prob.new_empty((bsz,))

    if scope == "mini_batch":
        beta_raw_scalar = _compute_adaptive_beta_raw_closed_form(
            v0_hat=v0_hat,
            logratio=old_logratio,
            kl=kl,
            r=r,
            response_mask=response_mask,
            gamma=gamma,
            beta_upper_bound=beta_upper_bound,
        )
        beta_scalar = _clamp_adaptive_beta(
            beta_raw=beta_raw_scalar, base_beta=base_beta, beta_upper_bound=beta_upper_bound
        )
        beta.fill_(float(beta_scalar.item()))
        beta_raw.fill_(float(beta_raw_scalar.item()))
        return beta, beta_raw

    if uid is None:
        raise ValueError("vo.adaptive_beta_scope='per_prompt_batch' requires non_tensor_batch['uid'].")
    uid_array = np.asarray(uid)
    if uid_array.shape[0] != bsz:
        raise ValueError(f"Expected uid length {bsz}, got {uid_array.shape[0]}.")

    unique_uids, inverse_indices = np.unique(uid_array, return_inverse=True)
    inverse_indices = torch.as_tensor(inverse_indices, device=old_log_prob.device, dtype=torch.long)
    group_beta = []
    group_beta_raw = []
    for group_idx in range(len(unique_uids)):
        group_mask = inverse_indices == group_idx
        beta_raw_scalar = _compute_adaptive_beta_raw_closed_form(
            v0_hat=v0_hat[group_mask],
            logratio=old_logratio[group_mask],
            kl=kl[group_mask],
            r=r[group_mask],
            response_mask=response_mask[group_mask],
            gamma=gamma,
            beta_upper_bound=beta_upper_bound,
        )
        beta_scalar = _clamp_adaptive_beta(
            beta_raw=beta_raw_scalar, base_beta=base_beta, beta_upper_bound=beta_upper_bound
        )
        group_beta.append(beta_scalar)
        group_beta_raw.append(beta_raw_scalar)

    group_beta_tensor = torch.stack(group_beta)
    group_beta_raw_tensor = torch.stack(group_beta_raw)
    beta = group_beta_tensor[inverse_indices]
    beta_raw = group_beta_raw_tensor[inverse_indices]
    return beta, beta_raw


def vo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None, force_use_grpo_actor: Optional[bool] = None):
    """
    Value-Optimization (VO) loss for RLVR training.
    Aligned with Intuitor rlvr_vo_trainer._compute_loss when use_ppo_actor=False and use_grpo_actor=False.

    Core algorithm (when both actor flags are False):
    - Recurrence: B_t = beta*log(π/π_ref) - beta*KL_t - r_t, V_{t+1} = (V_t + B_t)/gamma, vpred = V[:-1].
    - GAE from values_old: delta = r + gamma*m_next*v_old_next - values_old, then gae = delta_t + gamma*λ*gae.
    - returns = (advantages + values_old).detach().
    - loss = masked mean of 0.5*max((vpred-returns)^2, (v_clipped-returns)^2).
    Policy is updated only through this value loss (vpred depends on current π via B); no separate critic.

    Optional (when use_grpo_actor=True): add GRPO-style clipped ratio * advantage. We use data["advantages"]
    (GAE) for the actor; Intuitor uses group-normalized final_rewards when use_grpo_actor=True.

    Optional (when use_final_value_loss=True): override value loss with terminal-zero SE(V(s_T), 0) only.
    GAE is still computed (for use_grpo_actor); final_rewards required.

    Optional (when use_ppo_actor=True and unified_ppo_last_token=True):
    PPO actor uses VO-closed-form TD residual epsilon_t = beta * (log(pi/ref)_t - KL_t)
    for all tokens (including final token), then runs GAE on epsilon.
    """
    log_prob = model_output["log_probs"]
    entropy = model_output.get("entropy", None)

    # Only call no_padding_2_padding when data has indices (no-padding / left-right layout).
    # When use_remove_padding, the actor already returns padded (bsz, response_length) tensors.
    has_indices = tu.get_non_tensor_data(data=data, key="indices", default=None) is not None
    if has_indices:
        log_prob = no_padding_2_padding(log_prob, data)  # (bsz, response_length)
        if entropy is not None:
            entropy = no_padding_2_padding(entropy, data)  # (bsz, response_length)
    # else: log_prob / entropy are already (bsz, response_length)

    metrics = {}
    response_mask = data["response_mask"].to(bool)

    # Retrieve VO-specific data
    old_log_prob = data["old_log_probs"]
    ref_log_prob = data.get("ref_log_prob", None)
    advantages = data["advantages"]
    values_old = data.get("values_old", None)  # V_{theta_old} from rollout
    v0_hat = data.get("v0_hat", None)  # Group MC estimate of V(s0)
    final_rewards = data.get("final_rewards", None)  # RLVR final reward

    # Get VO configuration
    vo_config = getattr(config, "vo_config", None)
    # Keep DictConfig / dict-like mappings; do not silently drop non-dict mapping types.
    if vo_config is None:
        vo_config = {}
    elif not isinstance(vo_config, Mapping):
        vo_config = {}
    gamma = vo_config.get("gamma", getattr(config, "vo_gamma", 1.0))
    base_beta = float(vo_config.get("beta", getattr(config, "vo_beta", 0.005)))
    cliprange_value = vo_config.get("cliprange_value", getattr(config, "vo_cliprange_value", 0.2))
    gae_lambda = vo_config.get("gae_lambda", getattr(config, "vo_gae_lambda", 0.95))
    use_grpo_actor = vo_config.get("use_grpo_actor", getattr(config, "vo_use_grpo_actor", False))
    if force_use_grpo_actor is not None:
        use_grpo_actor = bool(force_use_grpo_actor)
    use_ppo_actor = vo_config.get("use_ppo_actor", getattr(config, "vo_use_ppo_actor", False))
    use_final_value_loss = vo_config.get(
        "use_final_value_loss", getattr(config, "vo_use_final_value_loss", False)
    )
    unified_ppo_last_token = vo_config.get(
        "unified_ppo_last_token", getattr(config, "vo_unified_ppo_last_token", True)
    )
    grpo_coef = vo_config.get("grpo_coef", getattr(config, "vo_grpo_coef", 0.5))
    cliprange_ppo = vo_config.get("cliprange_ppo", getattr(config, "vo_cliprange_ppo", 0.2))
    adaptive_beta = bool(vo_config.get("adaptive_beta", getattr(config, "vo_adaptive_beta", False)))
    adaptive_beta_scope = _validate_adaptive_beta_scope(
        str(vo_config.get("adaptive_beta_scope", "per_prompt_batch"))
    )
    beta_upper_bound = float(vo_config.get("beta_upper_bound", 1.0))
    vo_kl_mode = vo_config.get("kl_mode", getattr(config, "vo_kl_mode", "exact"))
    if vo_kl_mode not in ("exact", "candidate_set"):
        raise ValueError(
            "VO loss does not support kl_mode=approximate; use exact or candidate_set."
        )

    loss_agg_mode = config.loss_agg_mode

    # VO requires values_old and v0_hat; do not fall back to vanilla PPO (caller must ensure VO data).
    if values_old is None or v0_hat is None:
        raise ValueError(
            "VO loss requires 'values_old' and 'v0_hat' in batch. "
            "Ensure the VO trainer runs _compute_vo_data / compute_vo_values_and_advantages before update_actor."
        )

    # Current policy logratio and KL for B_t. VO requires ref_log_prob and kl_per_token.
    if ref_log_prob is None:
        raise ValueError(
            "VO loss requires 'ref_log_prob' in batch. Ensure VO trainer provides reference log probs."
        )
    if "kl_per_token" not in data:
        raise ValueError(
            "VO loss requires 'kl_per_token' for exact/candidate_set KL. "
            "Ensure VO trainer uses compute_log_prob_ref_and_kl."
        )
    logratio = log_prob - ref_log_prob
    kl = data["kl_per_token"].to(log_prob.device, dtype=log_prob.dtype)
    if kl.shape != log_prob.shape and has_indices:
        kl = no_padding_2_padding(kl, data)

    # Token rewards: only last valid token gets final reward
    response_lengths = response_mask.sum(dim=-1, dtype=torch.long)
    t_last = response_lengths.clamp(min=1) - 1
    r = torch.zeros_like(log_prob)
    if final_rewards is not None:
        final_rewards = final_rewards.to(r.dtype)
        r.scatter_(1, t_last.unsqueeze(1), final_rewards.unsqueeze(1))
        r = r * response_mask

    beta_raw = log_prob.new_tensor(base_beta)
    if adaptive_beta:
        if not use_final_value_loss:
            raise ValueError(
                "vo.adaptive_beta=True requires vo.use_final_value_loss=True."
            )
        if "vo_beta" not in data or "vo_adaptive_beta_raw" not in data:
            raise ValueError(
                "vo.adaptive_beta=True requires precomputed 'vo_beta' and 'vo_adaptive_beta_raw' in batch."
            )
        beta_raw = data["vo_adaptive_beta_raw"].to(log_prob.device, dtype=log_prob.dtype)
        beta = data["vo_beta"].to(log_prob.device, dtype=log_prob.dtype)
        if beta.dim() != 1 or beta.shape[0] != log_prob.shape[0]:
            raise ValueError(
                f"Expected vo_beta to have shape ({log_prob.shape[0]},), got {tuple(beta.shape)}."
            )
        if beta_raw.dim() != 1 or beta_raw.shape[0] != log_prob.shape[0]:
            raise ValueError(
                f"Expected vo_adaptive_beta_raw to have shape ({log_prob.shape[0]},), got {tuple(beta_raw.shape)}."
            )
        metrics["vo/beta"] = float(beta.float().mean().item())
        metrics["vo/adaptive_beta_raw"] = float(beta_raw.float().mean().item())
    else:
        beta = log_prob.new_tensor(base_beta)
        metrics["vo/beta"] = float(beta.item())
        metrics["vo/adaptive_beta_raw"] = float(beta_raw.item())

    # Construct current value prediction via recurrence: V_{t+1} = (V_t + B_t) / gamma
    # where B_t = beta * logratio_t - beta * KL_t - r_t
    beta_term = beta.unsqueeze(-1) if beta.dim() == 1 else beta
    B = beta_term * logratio - beta_term * kl - r
    Bsz, L = log_prob.shape
    device = log_prob.device

    # Build Vprefix using a list to preserve gradient flow (in-place assignment breaks autograd)
    # V_t is (Bsz,), B[:, t] is (Bsz,), mask[:, t] is (Bsz,)
    V_list = [v0_hat.to(log_prob.dtype)]  # (Bsz,)
    V_t = V_list[0]

    for t in range(L):
        V_next = (V_t + B[:, t]) / gamma  # (Bsz,)
        mask_t = response_mask[:, t].bool()  # (Bsz,)
        V_t = torch.where(mask_t, V_next, V_t)  # (Bsz,)
        V_list.append(V_t)

    # Stack into (Bsz, L+1) then take first L columns for vpred
    Vprefix = torch.stack(V_list, dim=1)  # (Bsz, L+1)
    vpred = Vprefix[:, :-1]  # (Bsz, L)

    # Compute GAE from values_old
    zeros = torch.zeros((Bsz, 1), device=device, dtype=vpred.dtype)
    v_old_next = torch.cat([values_old[:, 1:], zeros], dim=1)
    m_next = torch.cat([
        response_mask[:, 1:].float(),
        torch.zeros((Bsz, 1), device=device, dtype=response_mask.dtype).float()
    ], dim=1)

    delta = r + gamma * m_next * v_old_next - values_old

    # GAE computation
    gae_advantages = torch.zeros_like(delta)
    gae = torch.zeros((Bsz,), device=device, dtype=vpred.dtype)
    for t in reversed(range(L)):
        gae = delta[:, t] + gamma * gae_lambda * gae
        gae = gae * response_mask[:, t].float()
        gae_advantages[:, t] = gae

    unified_ppo_advantages = None
    if use_ppo_actor and unified_ppo_last_token:
        # Unified actor delta from the same rollout snapshot as values_old:
        # use old logratio and precomputed KL, and keep EOS bootstrap through
        # VO recurrence (instead of forcing terminal next-value to zero).
        old_logratio = old_log_prob - ref_log_prob
        B_old = beta_term * old_logratio - beta_term * kl - r
        v_old_next_unified = (values_old + B_old) / gamma
        delta_unified = r + gamma * v_old_next_unified - values_old

        unified_ppo_advantages = torch.zeros_like(delta_unified)
        gae_unified = torch.zeros((Bsz,), device=device, dtype=vpred.dtype)
        for t in reversed(range(L)):
            gae_unified = delta_unified[:, t] + gamma * gae_lambda * gae_unified
            gae_unified = gae_unified * response_mask[:, t].float()
            unified_ppo_advantages[:, t] = gae_unified

    returns = (gae_advantages + values_old).detach()

    # v_at_final: value at last valid token (EOS or last token before padding)
    Bsz, L = vpred.shape
    response_lengths = response_mask.sum(dim=-1, dtype=torch.long)
    t_last = response_lengths.clamp(min=1) - 1
    v_at_final = vpred[torch.arange(Bsz, device=vpred.device), t_last]
    # v_at_terminal: value of terminal state after consuming final token.
    # This corresponds to V_T in the VO recurrence (Vprefix has shape [B, L+1]).
    v_at_terminal = Vprefix[torch.arange(Bsz, device=vpred.device), t_last + 1]
    terminal_no_beta, terminal_beta_coeff = _compute_terminal_affine_components(
        v0_hat=v0_hat.to(log_prob.dtype),
        logratio=logratio,
        kl=kl,
        r=r,
        response_mask=response_mask,
        gamma=gamma,
    )
    terminal_beta_contrib = terminal_beta_coeff * beta if beta.dim() == 1 else terminal_beta_coeff * beta
    terminal_no_beta_mse = (terminal_no_beta ** 2).mean()

    denom = response_mask.float().sum().clamp(min=1.0)

    if use_final_value_loss:
        # Override: terminal-zero loss after the final transition has already consumed the last reward.
        # GAE is still computed above (used for data["advantages"] when use_grpo_actor=True).
        if final_rewards is None:
            raise ValueError(
                "use_final_value_loss=True requires 'final_rewards' in batch. "
                "Ensure the reward model provides final_rewards."
            )
        value_loss = (v_at_terminal ** 2).mean()
        value_rms = value_loss.clamp(min=0).sqrt()
    else:
        # Original: clipped value loss over all tokens
        v_clipped = torch.clamp(vpred, values_old - cliprange_value, values_old + cliprange_value)
        vf_loss_1 = (vpred - returns) ** 2
        vf_loss_2 = (v_clipped - returns) ** 2
        vf_loss = 0.5 * torch.max(vf_loss_1, vf_loss_2)
        value_loss = (vf_loss * response_mask.float()).sum() / denom
        if final_rewards is not None:
            value_rms = ((v_at_final - final_rewards.to(vpred.dtype)) ** 2).mean().clamp(min=0).sqrt()
        else:
            value_rms = ((vpred - returns) ** 2 * response_mask.float()).sum() / denom
            value_rms = value_rms.clamp(min=0).sqrt()

    loss = value_loss

    # Optional actor loss
    if use_grpo_actor:
        ratio = torch.exp(log_prob - old_log_prob)
        ratio_clipped = torch.clamp(ratio, 1.0 - cliprange_ppo, 1.0 + cliprange_ppo)

        # Use advantages for actor loss (already group-normalized in compute_advantage)
        adv_det = advantages.detach()
        ppo_loss_per_token = -torch.min(ratio * adv_det, ratio_clipped * adv_det)
        actor_loss = (ppo_loss_per_token * response_mask.float()).sum() / denom
        loss = loss + grpo_coef * actor_loss

        clipfrac = ((ratio != ratio_clipped).float() * response_mask.float()).sum() / denom
        metrics["vo/grpo_clipfrac"] = clipfrac.detach().item()
        metrics["vo/actor_loss"] = actor_loss.detach().item()
    elif use_ppo_actor:
        ratio = torch.exp(log_prob - old_log_prob)
        ratio_clipped = torch.clamp(ratio, 1.0 - cliprange_ppo, 1.0 + cliprange_ppo)

        if unified_ppo_last_token:
            if unified_ppo_advantages is None:
                raise RuntimeError("Internal error: unified_ppo_advantages was not computed.")
            adv_det = unified_ppo_advantages.detach()
            metrics["vo/ppo_unified_last_token"] = 1.0
        else:
            # Legacy PPO actor advantage path based on values_old/final-reward terminal handling.
            adv_det = gae_advantages.detach()
            metrics["vo/ppo_unified_last_token"] = 0.0
        ppo_loss_per_token = -torch.min(ratio * adv_det, ratio_clipped * adv_det)
        actor_loss = (ppo_loss_per_token * response_mask.float()).sum() / denom
        loss = loss + grpo_coef * actor_loss

        clipfrac = ((ratio != ratio_clipped).float() * response_mask.float()).sum() / denom
        metrics["vo/ppo_clipfrac"] = clipfrac.detach().item()
        metrics["vo/actor_loss"] = actor_loss.detach().item()

    # Metrics (Intuitor-style: loss, value_rms, value_loss, exact_kl when from kl_per_token)
    mean_kl = (kl * response_mask.float()).sum() / denom
    metrics.update({
        "vo/loss": loss.detach().item(),
        "vo/value_loss": value_loss.detach().item(),
        "vo/value_rms": value_rms.detach().item(),
        "vo/kl_mean": mean_kl.detach().item(),
        "vo/vpred_mean": (vpred * response_mask.float()).sum().detach().item() / denom.item(),
        "vo/returns_mean": (returns * response_mask.float()).sum().detach().item() / denom.item(),
    })
    if use_final_value_loss:
        metrics["vo/final_state_no_beta_mse"] = terminal_no_beta_mse.detach().item()
        metrics["vo/final_state_improvement_ratio"] = (
            value_loss.detach() / terminal_no_beta_mse.detach().clamp(min=1e-8)
        ).item()
        valid_sign = (terminal_beta_contrib != 0) & (terminal_no_beta != 0)
        metrics["vo/value_correct_sign_ratio_valid_frac"] = valid_sign.float().mean().item()
        if valid_sign.any():
            sign_match = torch.sign(terminal_beta_contrib[valid_sign]) == torch.sign(-terminal_no_beta[valid_sign])
            metrics["vo/value_correct_sign_ratio"] = sign_match.float().mean().item()
        else:
            metrics["vo/value_correct_sign_ratio"] = 0.0
    if "kl_per_token" in data:
        exact_kl_mean = (data["kl_per_token"].float() * response_mask.float()).sum() / denom
        metrics["vo/exact_kl_mean"] = exact_kl_mean.detach().item()

    # Add entropy loss if configured
    if entropy is not None:
        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
        entropy_coeff = config.entropy_coeff
        loss -= entropy_coeff * entropy_loss

    # Add KL loss if configured
    if config.use_kl_loss and ref_log_prob is not None:
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode)
        loss += kl_loss * config.kl_loss_coef
        metrics["vo/kl_loss"] = kl_loss.detach().item()

    return loss, metrics


def vpo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """
    Value-Policy Optimization (VPO) loss - combines VO value loss with PPO policy loss.
    This is a convenience wrapper that sets use_grpo_actor=True.
    """
    return vo_loss(config, model_output, data, dp_group, force_use_grpo_actor=True)
