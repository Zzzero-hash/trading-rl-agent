from typing import Any


class RuleEngine:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def adjust_position_sizing(self, current_position: float, regime: str) -> float:
        if regime == "bull":
            return current_position * float(self.config["bull_multiplier"])
        elif regime == "bear":
            return current_position * float(self.config["bear_multiplier"])
        return current_position

    def set_stop_loss(self, current_price: float, regime: str) -> float:
        if regime == "bull":
            return current_price * (1 - float(self.config["bull_stop_loss"]))
        elif regime == "bear":
            return current_price * (1 + float(self.config["bear_stop_loss"]))
        return current_price

    def activate_strategy(self, regime: str) -> bool:
        return regime in self.config["active_strategies"]
