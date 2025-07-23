import hashlib
import time
from pathlib import Path

import pandas as pd


class CacheManager:
    """Intelligent caching for pipeline data."""

    def __init__(self, cache_dir: Path = Path("data/cache")):
        """Initialize the cache manager."""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, operation: str, **kwargs: str) -> str:
        """Generate cache key for operation."""
        key_data = f"{operation}_{kwargs}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def get_cached_data(self, cache_key: str, ttl_hours: int = 24) -> pd.DataFrame | None:
        """Get cached data if valid."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        if cache_file.exists():
            mod_time = cache_file.stat().st_mtime
            if time.time() - mod_time < (ttl_hours * 3600):
                try:
                    return pd.read_parquet(cache_file)
                except Exception:
                    pass

        return None

    def cache_data(self, cache_key: str, data: pd.DataFrame) -> None:
        """Cache data."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        data.to_parquet(cache_file)
