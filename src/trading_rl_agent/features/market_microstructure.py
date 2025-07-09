"""Market microstructure feature calculations."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..core.logging import get_logger


@dataclass
class MicrostructureConfig:
    """Configuration for microstructure feature calculations."""

    volume_window: int = 5


class MarketMicrostructure:
    """Compute simple market microstructure features."""

    def __init__(self, config: Optional[MicrostructureConfig] = None) -> None:
        self.config = config or MicrostructureConfig()
        self.logger = get_logger(self.__class__.__name__)

    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic microstructure features to ``df``."""
        result = df.copy()
        result["hl_spread"] = (
            result["high"].astype(float) - result["low"].astype(float)
        ) / result["close"].replace(0, np.nan)
        result["close_open_diff"] = result["close"] - result["open"]
        result["volume_imbalance"] = result["volume"] - result["volume"].rolling(
            self.config.volume_window, min_periods=1
        ).mean()
        return result

    def get_feature_names(self) -> list[str]:
        """Return names of generated microstructure features."""
        return ["hl_spread", "close_open_diff", "volume_imbalance"]
