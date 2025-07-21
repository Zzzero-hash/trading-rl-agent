"""Cross-asset correlation feature calculations."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.trading_rl_agent.core.logging import get_logger


@dataclass
class CrossAssetConfig:
    """Configuration for cross-asset features."""

    corr_window: int = 5
    prefix: str = "bench"


class CrossAssetFeatures:
    """Compute rolling correlation with a reference asset."""

    def __init__(self, config: CrossAssetConfig | None = None) -> None:
        self.config = config or CrossAssetConfig()
        self.logger = get_logger(self.__class__.__name__)

    def add_cross_asset_features(
        self,
        df: pd.DataFrame,
        reference: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add rolling correlation of ``df`` with ``reference``."""
        result = df.copy()
        ref_close = reference["close"].reset_index(drop=True).iloc[: len(result)]
        result["_ref_close"] = ref_close
        result["ret"] = np.log(result["close"] / result["close"].shift(1))
        result["ref_ret"] = np.log(ref_close / ref_close.shift(1))
        result[f"corr_{self.config.prefix}"] = (
            result["ret"].rolling(self.config.corr_window, min_periods=1).corr(result["ref_ret"])
        )
        result.drop(columns=["_ref_close", "ret", "ref_ret"], inplace=True)
        return result

    def get_feature_names(self) -> list[str]:
        """Return names of generated cross-asset features."""
        return [f"corr_{self.config.prefix}"]
