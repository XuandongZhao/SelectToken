# Value Optimization (VO) / Value-Policy Optimization (VPO) Recipe

This recipe implements the RLVR-VO (Rule-based Language Verification with Reward via Value Optimization) training objective from the Intuitor project.

## Overview

VO/VPO training combines value-based reinforcement learning with policy optimization:

1. **Value-based Loss**: Uses GAE (Generalized Advantage Estimation) from constructed values rather than a separate critic network
2. **GRPO-style Actor Loss**: Optional policy gradient loss with clipped importance sampling  
3. **Periodic Reference Model Updates**: Updates the reference model every N steps (default: 20)

## Key Features

- **No Separate Critic Network**: Values are constructed via a recurrence relation (Eq.6) using:
  - Group Monte-Carlo estimate of initial value (v0_hat)
  - KL penalty coefficient (beta)
  - Discount factor (gamma)

- **Reference Model Sync**: Unlike standard PPO/DAPO which keeps the reference model fixed, VO periodically synchronizes the reference model with the current policy

- **Compatible with DAPO**: Can be used as a drop-in replacement for DAPO training with additional VO objectives

## Configuration

### VO-specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vo.gamma` | 1.0 | Discount factor for value computation |
| `vo.beta` | 0.005 | KL penalty coefficient in value recurrence |
| `vo.adaptive_beta` | false | When true, compute per-batch closed-form `beta_raw` minimizing terminal-zero final-value loss, then use `max(vo.beta, beta_raw)` |
| `vo.beta_upper_bound` | 1.0 | Upper bound for effective beta; final beta is clamped to this value |
| `vo.gae_lambda` | 0.95 | GAE lambda for advantage estimation |
| `vo.cliprange_value` | 0.2 | Value clipping range |
| `vo.use_grpo_actor` | true | Whether to use GRPO-style actor loss |
| `vo.grpo_coef` | 0.5 | Coefficient for GRPO actor loss |
| `vo.cliprange_ppo` | 0.2 | PPO clip range for actor |
| `vo.update_ref_per_n_step` | 20 | Update reference model every N steps |
| `vo.ref_update_kl_threshold` | null | If set (e.g., `0.005`), update ref when `vo/exact_kl_mean` at current step is above threshold |
| `vo.update_ref_buffer_step` | 0 | If >0, ref-update trigger saves actor snapshot to disk and applies it to ref after X steps |
| `vo.refresh_adam_upon_ref_update` | false | Clear optimizer state after each ref update (resets Adam moments and optimizer `step`) |
| `vo.kl_mode` | exact | KL computation mode. Supported: `exact`, `candidate_set` |

`vo.kl_mode=approximate` is not supported in VO trainer and will raise a `ValueError`.

`vo.adaptive_beta=true` requires `vo.use_final_value_loss=true`; otherwise VO loss raises a `ValueError`.
When `vo.use_final_value_loss=true`, the supervised target is terminal-state zero after the final reward has already been consumed by the last transition.
If the closed-form solve is degenerate (`sum(b^2)` too small) or non-finite, adaptive beta uses `vo.beta_upper_bound` (default `1.0`).

## Usage

### Multi-node Training (8 nodes x 8 GPUs)

```bash
# Set your paths
export RAY_DATA_HOME="${HOME}/verl"
export MODEL_PATH="${RAY_DATA_HOME}/models/Qwen2.5-Math-7B"
export TRAIN_FILE="${RAY_DATA_HOME}/data/dapo-math-17k.parquet"
export TEST_FILE="${RAY_DATA_HOME}/data/aime-2024.parquet"

# Run training
bash recipe/vo/run_dapo_vo_7b_math.sh
```

### Single-node Training (1 node x 8 GPUs)

```bash
# For testing and development
bash recipe/vo/run_dapo_vo_7b_math_single_node.sh
```

### Custom Configuration

You can override any parameter via command line:

```bash
python3 -m recipe.vo.main_vo \
    vo.update_ref_per_n_step=10 \
    vo.beta=0.01 \
    vo.use_grpo_actor=False \
    ...
```

## Files

- `vo_core_algos.py`: Core algorithms for VO value/advantage computation
- `vo_ray_trainer.py`: Ray-based trainer extending RayPPOTrainer with VO objectives
- `main_vo.py`: Main entry point for training
- `config/vo_trainer.yaml`: Default configuration
- `run_dapo_vo_7b_math.sh`: Multi-node training script
- `run_dapo_vo_7b_math_single_node.sh`: Single-node training script

## Metrics

VO training logs the following additional metrics:

| Metric | Description |
|--------|-------------|
| `vo/v0_hat_mean` | Mean of group MC value estimates |
| `vo/final_reward_mean` | Mean final reward |
| `vo/value_loss` | Value loss; terminal-zero MSE on `V(s_T)` when `use_final_value_loss=true` |
| `vo/value_rms` | RMS companion of `vo/value_loss` |
| `vo/values_old_final_rms` | Rollout-time RMS of `values_old` at the final valid token vs final reward |
| `vo/kl_mean` | Mean KL divergence from `kl_per_token` (exact/candidate_set path) |
| `vo/vpred_mean` | Mean value prediction |
| `vo/returns_mean` | Mean returns |
| `vo/final_state_no_beta_mse` | Terminal-zero MSE when beta contribution is removed |
| `vo/final_state_improvement_ratio` | `vo/value_loss / max(vo/final_state_no_beta_mse, 1e-8)` |
| `vo/value_correct_sign_ratio` | Fraction of valid samples where beta correction moves terminal value toward zero |
| `vo/value_correct_sign_ratio_valid_frac` | Fraction of samples with non-zero baseline and correction signs |
| `vo/grpo_clipfrac` | GRPO clip fraction (if enabled) |
| `vo/actor_loss` | Actor loss (if GRPO enabled) |
| `vo/ref_model_updated` | Indicator when ref model is updated |
| `vo/beta` | Effective beta used in this batch (after lower-bound clamp by `vo.beta`) |
| `vo/adaptive_beta_raw` | Raw adaptive beta before clamp |

## Reference

This implementation is based on the RLVR-VO approach from the Intuitor project:
- [Intuitor RLVR-VO Trainer](https://github.com/xxx/Intuitor/blob/main/src/open_r1/rlvr_vo_trainer.py)

## Loss Functions

The VO loss combines:

1. **Clipped Value Loss**:
   ```
   L_value = 0.5 * max((V - returns)^2, (V_clipped - returns)^2)
   ```

   When `vo.use_final_value_loss=true`, this is replaced by terminal-zero supervision:
   ```
   L_value = mean(V(s_T)^2)
   ```

2. **GRPO Actor Loss** (optional):
   ```
   L_actor = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
   ```

3. **Total Loss**:
   ```
   L_total = L_value + grpo_coef * L_actor
   ```

## Tips

1. **Reference Model Updates**: The `update_ref_per_n_step` parameter controls how often the reference model is synchronized. Lower values make training more dynamic but may be less stable.

2. **Beta Parameter**: The `vo.beta` parameter controls the KL penalty in the value recurrence. Higher values encourage staying closer to the reference policy.

3. **GRPO vs Pure VO**: Setting `vo.use_grpo_actor=False` uses pure value-based training without the policy gradient term.
