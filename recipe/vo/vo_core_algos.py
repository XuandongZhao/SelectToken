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
Core algorithms for Value Optimization (VO) / Value-Policy Optimization (VPO) training.
This implements the RLVR-VO objective for training language models using value-based RL.

Reference: Intuitor RLVR-VO implementation
"""

from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import register_adv_est, AdvantageEstimator


# -----------------------------------------------------------------------------
# Exact KL and candidate-set KL (no full seq*vocab storage; call per micro-batch)
# -----------------------------------------------------------------------------

def exact_kl_from_logits(
    pol_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    temperature: float = 1.0,
    vocab_chunk_size: int = 0,
) -> torch.Tensor:
    """
    Exact forward KL per token: KL(π || π_ref) = Σ_a π(a) [log π(a) - log π_ref(a)].
    Inputs are (B, L, V); output is (B, L).
    If vocab_chunk_size > 0, computes KL in vocab chunks to reduce transient memory.
    """
    if pol_logits.shape != ref_logits.shape:
        raise ValueError(
            f"pol_logits/ref_logits shape mismatch: {tuple(pol_logits.shape)} vs {tuple(ref_logits.shape)}"
        )
    if pol_logits.dim() != 3:
        raise ValueError(f"exact_kl_from_logits expects 3D logits (B,L,V), got dim={pol_logits.dim()}")

    if temperature != 1.0:
        pol_logits = pol_logits / temperature
        ref_logits = ref_logits / temperature

    vocab_size = pol_logits.size(-1)
    use_chunk = (
        vocab_chunk_size is not None
        and vocab_chunk_size > 0
        and vocab_chunk_size < vocab_size
    )

    # Fast path: full-vocab computation.
    if not use_chunk:
        pol_logp = torch.log_softmax(pol_logits, dim=-1)
        ref_logp = torch.log_softmax(ref_logits, dim=-1)
        pol_p = torch.exp(pol_logp)
        return (pol_p * (pol_logp - ref_logp)).sum(dim=-1)

    # Memory-lean path: chunk both normalization and KL accumulation over vocab.
    def _chunked_logsumexp(logits: torch.Tensor, chunk_size: int) -> torch.Tensor:
        max_per_token = None
        for i in range(0, vocab_size, chunk_size):
            j = min(i + chunk_size, vocab_size)
            cur_max = logits[..., i:j].max(dim=-1, keepdim=True).values
            max_per_token = cur_max if max_per_token is None else torch.maximum(max_per_token, cur_max)

        # Keep accumulator in fp32 for numeric stability; it is only (B, L, 1).
        sum_exp = torch.zeros_like(max_per_token, dtype=torch.float32)
        for i in range(0, vocab_size, chunk_size):
            j = min(i + chunk_size, vocab_size)
            # Avoid explicit fp32 chunk materialization to reduce peak memory.
            sum_exp = sum_exp + torch.exp(logits[..., i:j] - max_per_token).sum(
                dim=-1, keepdim=True, dtype=torch.float32
            )
        return max_per_token.to(torch.float32) + torch.log(sum_exp)

    pol_logz = _chunked_logsumexp(pol_logits, vocab_chunk_size)
    ref_logz = _chunked_logsumexp(ref_logits, vocab_chunk_size)

    kl = torch.zeros(pol_logits.shape[:2], device=pol_logits.device, dtype=torch.float32)
    for i in range(0, vocab_size, vocab_chunk_size):
        j = min(i + vocab_chunk_size, vocab_size)
        pol_logp_chunk = pol_logits[..., i:j] - pol_logz
        ref_logp_chunk = ref_logits[..., i:j] - ref_logz
        pol_p_chunk = torch.exp(pol_logp_chunk)
        kl = kl + (pol_p_chunk * (pol_logp_chunk - ref_logp_chunk)).sum(dim=-1, dtype=torch.float32)

    return kl.to(pol_logits.dtype)


def candidate_set_kl_from_logits(
    pol_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    K: int,
    M: int = 0,
    vocab_size: Optional[int] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Forward KL over a restricted candidate set per token (top-K from policy + M random).
    Reduces memory from O(V) to O(K+M) per token. Inputs (B, L, V); output (B, L).
    """
    if temperature != 1.0:
        pol_logits = pol_logits / temperature
        ref_logits = ref_logits / temperature
    B, L, V = pol_logits.shape
    device = pol_logits.device
    if vocab_size is None:
        vocab_size = V
    # Candidate indices: top-K from policy logits
    topk_val, topk_idx = torch.topk(pol_logits, k=min(K, V), dim=-1)
    if M > 0:
        rand_idx = torch.randint(0, vocab_size, (B, L, M), device=device, dtype=topk_idx.dtype)
        cand_idx = torch.cat([topk_idx, rand_idx], dim=-1)
    else:
        cand_idx = topk_idx
    pol_c = pol_logits.gather(-1, cand_idx)
    ref_c = ref_logits.gather(-1, cand_idx)
    pol_logp_c = pol_c - torch.logsumexp(pol_c, dim=-1, keepdim=True)
    ref_logp_c = ref_c - torch.logsumexp(ref_c, dim=-1, keepdim=True)
    pol_p_c = torch.exp(pol_logp_c)
    kl = (pol_p_c * (pol_logp_c - ref_logp_c)).sum(dim=-1)
    return kl


def compute_vo_values_and_advantages(
    data: DataProto,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    beta: float = 0.005,
) -> DataProto:
    """
    Compute values and advantages for Value Optimization training.
    
    This function computes:
    1. v0_hat: Group Monte-Carlo estimate of V(s0) with discount
    2. values_old: Constructed V_{theta_old}(s_t) via recurrence
    3. advantages: GAE-based advantages from values_old
    
    Args:
        data: DataProto containing:
            - token_level_rewards: (bs, response_length) final rewards are on last token
            - response_mask: (bs, response_length)
            - old_log_probs: (bs, response_length) policy log probs
            - ref_log_prob: (bs, response_length) reference policy log probs (optional)
            - uid: unique id for grouping samples
        gamma: Discount factor (default 1.0 for episodic)
        gae_lambda: GAE lambda parameter
        beta: KL penalty coefficient
        
    Returns:
        Updated DataProto with:
            - v0_hat: (bs,) group MC estimate of initial value
            - values_old: (bs, response_length) constructed values
            - advantages: (bs, response_length) GAE advantages
            - returns: (bs, response_length) GAE returns
            - final_rewards: (bs,) scalar final reward per sequence
    """
    response_mask = data.batch["response_mask"]
    token_level_rewards = data.batch["token_level_rewards"]
    old_log_probs = data.batch["old_log_probs"]
    ref_log_prob = data.batch.get("ref_log_prob", None)
    uid = data.non_tensor_batch.get("uid", None)
    
    device = response_mask.device
    Bsz, L = response_mask.shape
    
    # Extract final rewards (sum of token-level rewards, typically only last token has reward)
    final_rewards = token_level_rewards.sum(dim=-1)  # (Bsz,)
    
    # Compute sequence lengths and last token index
    lengths = response_mask.sum(dim=1, dtype=torch.long)
    t_last = lengths.clamp(min=1) - 1
    
    # ===== Compute v0_hat: Group MC estimate of V(s0) =====
    # G0 = gamma^{t_last} * R_final
    disc0 = torch.pow(torch.tensor(gamma, device=device), t_last.float())
    g0 = disc0 * final_rewards
    
    # Group by uid to get group mean
    if uid is None:
        raise ValueError(
            "VO value computation requires non_tensor_batch['uid'] to compute group-wise v0_hat."
        )
    uid_array = np.array(uid)
    if uid_array.shape[0] != Bsz:
        raise ValueError(
            f"VO value computation requires uid length to match batch size, got {uid_array.shape[0]} vs {Bsz}."
        )
    unique_uids, inverse_indices = np.unique(uid_array, return_inverse=True)
    inverse_indices = torch.tensor(inverse_indices, device=device, dtype=torch.long)

    # Compute group means
    num_groups = len(unique_uids)
    group_sums = torch.zeros(num_groups, device=device, dtype=g0.dtype)
    group_counts = torch.zeros(num_groups, device=device, dtype=torch.long)

    group_sums.scatter_add_(0, inverse_indices, g0)
    group_counts.scatter_add_(0, inverse_indices, torch.ones(Bsz, device=device, dtype=torch.long))

    group_means = group_sums / group_counts.float().clamp(min=1)
    v0_hat = group_means[inverse_indices]
    
    # ===== Construct token-level rewards =====
    # In RLVR, reward is only on the final token
    r = torch.zeros_like(old_log_probs)
    r.scatter_(1, t_last.unsqueeze(1), final_rewards.unsqueeze(1))
    r = r * response_mask
    
    # ===== Compute KL divergence =====
    # VO requires precomputed kl_per_token (exact or candidate-set) and ref_log_prob.
    kl_per_token = data.batch.get("kl_per_token", None)
    if kl_per_token is None:
        raise ValueError(
            "VO value computation requires 'kl_per_token' in batch for exact/candidate_set KL."
        )
    if ref_log_prob is None:
        raise ValueError(
            "VO value computation requires 'ref_log_prob' in batch."
        )
    kl = kl_per_token.to(old_log_probs.device).to(old_log_probs.dtype)
    logratio = old_log_probs - ref_log_prob
    
    # ===== Construct values_old via Eq.(6) recurrence =====
    # B_k = beta * log(pi/ref)(a_k) - beta * KL(s_k) - r_k
    B = beta * logratio - beta * kl - r
    
    Vprefix = torch.zeros((Bsz, L + 1), device=device, dtype=old_log_probs.dtype)
    Vprefix[:, 0] = v0_hat
    
    # Recurrence: V_{t+1} = (V_t + B_t) / gamma
    for t in range(L):
        V_next = (Vprefix[:, t] + B[:, t]) / gamma
        Vprefix[:, t + 1] = torch.where(response_mask[:, t].bool(), V_next, Vprefix[:, t])
    
    values_old = Vprefix[:, :-1].detach()
    
    # ===== Compute GAE advantages from values_old =====
    zeros = torch.zeros((Bsz, 1), device=device, dtype=values_old.dtype)
    v_old_next = torch.cat([values_old[:, 1:], zeros], dim=1)
    m_next = torch.cat([
        response_mask[:, 1:].float(),
        torch.zeros((Bsz, 1), device=device, dtype=response_mask.dtype).float()
    ], dim=1)
    
    delta = r + gamma * m_next * v_old_next - values_old
    
    # GAE computation (backward pass)
    advantages = torch.zeros_like(delta)
    gae = torch.zeros((Bsz,), device=device, dtype=values_old.dtype)
    
    for t in reversed(range(L)):
        gae = delta[:, t] + gamma * gae_lambda * gae
        gae = gae * response_mask[:, t].float()
        advantages[:, t] = gae
    
    returns = (advantages + values_old).detach()
    
    # Store results in data
    data.batch["v0_hat"] = v0_hat
    data.batch["values_old"] = values_old
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    data.batch["final_rewards"] = final_rewards
    
    return data


@register_adv_est("vo")
def compute_vo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for VO training.
    This combines GRPO-style group normalization with value-based advantage estimation.
    
    For VO, advantages are computed differently - they come from GAE using constructed values.
    This function provides a fallback GRPO-style advantage when full VO data is not available.
    """
    scores = token_level_rewards.sum(dim=-1)
    
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        
        scores = scores.unsqueeze(-1) * response_mask
    
    return scores, scores


def sync_ref_model_from_policy(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tau: float = 1.0,
) -> None:
    """
    Synchronize reference model parameters from policy model.
    
    Args:
        policy_model: The current policy model
        ref_model: The reference model to update
        tau: Interpolation factor (1.0 = full copy, < 1.0 = exponential moving average)
    """
    with torch.no_grad():
        for param, ref_param in zip(policy_model.parameters(), ref_model.parameters()):
            if param is None or ref_param is None:
                continue
            if param.shape != ref_param.shape:
                continue
            
            if tau == 1.0:
                ref_param.data.copy_(param.data)
            else:
                ref_param.data.mul_(1 - tau).add_(param.data * tau)


def compute_ref_sync_diff(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    numel: int = 128,
) -> float:
    """
    Compute sample parameter difference between policy and reference model.
    Used for logging/debugging reference model sync.
    """
    for p_policy, p_ref in zip(policy_model.parameters(), ref_model.parameters()):
        if p_policy is None or p_ref is None:
            continue
        if p_policy.numel() == 0 or p_ref.numel() == 0:
            continue
        
        n = min(numel, p_policy.numel(), p_ref.numel())
        pm = p_policy.view(-1)[:n].detach()
        pr = p_ref.view(-1)[:n].detach()
        return torch.mean(torch.abs(pm - pr)).item()
    
    return float("nan")
