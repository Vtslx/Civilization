from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CivilizationAblationConfig:
    use_memory_path: bool = True
    use_state_path: bool = True
    use_rule_path: bool = True
    use_chain_state_loss: bool = True
    use_final_state_loss: bool = True
    use_priority_control_loss: bool = True
    use_centroid_separation_loss: bool = True
    use_hard_negative_loss: bool = True


def default_ablation_config(config: CivilizationAblationConfig | None = None) -> CivilizationAblationConfig:
    return config or CivilizationAblationConfig()
