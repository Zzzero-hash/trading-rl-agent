"""Position sizing utilities."""

from __future__ import annotations


def kelly_position_size(
    expected_return: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_kelly_fraction: float = 0.25,
) -> float:
    """Return Kelly criterion position size fraction.

    Parameters
    ----------
    expected_return : float
        Expected return of the strategy (unused but kept for API compatibility).
    win_rate : float
        Historical probability of a winning trade between 0 and 1.
    avg_win : float
        Average profit of winning trades.
    avg_loss : float
        Average loss of losing trades (positive value).
    max_kelly_fraction : float, optional
        Maximum fraction of capital to risk, by default 0.25.
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0

    b = avg_win / avg_loss
    p = win_rate
    q = 1 - win_rate
    kelly_fraction = (b * p - q) / b
    kelly_fraction = max(0.0, kelly_fraction)
    return min(kelly_fraction, max_kelly_fraction)
