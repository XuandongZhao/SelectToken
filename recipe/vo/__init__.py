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
Value Optimization (VO) / Value-Policy Optimization (VPO) recipe for RLVR training.

This implements the RLVR-VO objective from the Intuitor project:
- Value-based loss using GAE from constructed values
- Optional GRPO-style actor loss
- Periodic reference model updates
"""

from .vo_core_algos import (
    compute_vo_values_and_advantages,
    sync_ref_model_from_policy,
    compute_ref_sync_diff,
)
from .vo_ray_trainer import RayVOTrainer

__all__ = [
    "compute_vo_values_and_advantages",
    "sync_ref_model_from_policy",
    "compute_ref_sync_diff",
    "RayVOTrainer",
]
