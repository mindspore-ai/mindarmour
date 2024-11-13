"""
This module prepares attacks.
"""
from .optimization_based_attack import OptimizationBasedAttacker


def prepare_attack(model, loss, cfg_attack):
    """prepares attack optimization methods."""
    if cfg_attack.attack_type == "optimization":
        attacker = OptimizationBasedAttacker(model, loss, cfg_attack)
    else:
        raise ValueError(f"Invalid type of attack {cfg_attack.attack_type} given.")

    return attacker


__all__ = ["prepare_attack"]
