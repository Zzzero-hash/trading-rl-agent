"""Alternative data feature calculations."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..core.logging import get_logger


@dataclass
class AlternativeDataConfig:
    """Configuration for alternative data features."""

    sentiment_column: str = "news_sentiment"


class AlternativeDataFeatures:
    """Attach alternative data such as sentiment scores."""

    def __init__(self, config: Optional[AlternativeDataConfig] = None) -> None:
        self.config = config or AlternativeDataConfig()
        self.logger = get_logger(self.__class__.__name__)

    def add_alternative_features(
        self, df: pd.DataFrame, sentiment: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        result = df.copy()
        col = self.config.sentiment_column
        if sentiment is not None:
            result[col] = sentiment.reset_index(drop=True).iloc[: len(result)]
        else:
            result[col] = 0.0
        return result

    def get_feature_names(self) -> list[str]:
        """Return names of generated alternative data features."""
        return [self.config.sentiment_column]
