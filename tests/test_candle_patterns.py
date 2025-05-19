"""
Tests for candlestick pattern detection functions.
"""
import pandas as pd
import numpy as np
import pytest

from src.data.features import (
    detect_doji,
    detect_hammer,
    detect_engulfing,
    detect_shooting_star,
    detect_morning_star,
    detect_evening_star,
)

from src.data.candle_patterns import (
    detect_inside_bar,
    detect_outside_bar,
    detect_tweezer_top,
    detect_tweezer_bottom,
    detect_three_white_soldiers,
    detect_three_black_crows,
    detect_harami,
    detect_dark_cloud_cover,
    detect_piercing_line,
    compute_candle_stats,
)

# Import the original functions to compare with advanced patterns
from src.data.features import (
    detect_doji as features_detect_doji,
    detect_hammer as features_detect_hammer,
    detect_engulfing as features_detect_engulfing,
)


def create_test_df():
    """Create a basic DataFrame for testing"""
    df = pd.DataFrame({
        'open': [10, 12, 14, 15, 16, 15, 14, 13, 15, 16],
        'high': [15, 14, 16, 17, 18, 16, 15, 17, 18, 17],
        'low': [8, 10, 13, 14, 15, 14, 13, 12, 14, 15],
        'close': [12, 14, 15, 16, 15, 14, 13, 15, 16, 14],
        'volume': [1000, 1500, 2000, 1800, 1600, 1400, 1200, 2200, 2400, 1400]
    })
    return df


def test_detect_doji():
    """Test doji pattern detection"""
    # Create data with a doji (open == close)
    df = pd.DataFrame({
        'open': [10, 12, 15, 15, 16],
        'high': [15, 14, 17, 17, 18], 
        'low': [8, 10, 13, 14, 15],
        'close': [12, 14, 15, 15, 16]
    })
    
    result = detect_doji(df)
    assert 'doji' in result.columns
    
    # Index 2 should be a doji (open == close == 15)
    assert result['doji'][2] == 1
    
    # Other candles should not be doji
    assert result['doji'][0] == 0
    assert result['doji'][1] == 0


def test_detect_hammer():
    """Test hammer pattern detection"""
    # Create data with a hammer pattern
    df = pd.DataFrame({
        # Previous downtrend
        'open': [12, 11, 10, 9.5, 9],
        'high': [13, 12, 10.2, 10, 9.5],
        'low': [11, 10, 9.8, 7, 8.5],
        'close': [11, 10, 9.9, 9.3, 9.2]
    })
    
    result = detect_hammer(df)
    assert 'hammer' in result.columns
    
    # Index 3 should be a hammer (small body, long lower shadow, in downtrend)
    assert result['hammer'][3] == 1
    
    # Other candles should not be hammers
    assert result['hammer'][:3].sum() == 0
    assert result['hammer'][4] == 0


def test_detect_inside_bar():
    """Test inside bar pattern detection"""
    df = pd.DataFrame({
        'open': [10, 12, 13.5, 14],
        'high': [15, 14, 13.8, 16],
        'low': [8, 10, 10.2, 13],
        'close': [12, 13, 13.7, 15]
    })
    
    result = detect_inside_bar(df)
    assert 'inside_bar' in result.columns
    
    # Index 2 should be an inside bar (high < prev high, low > prev low)
    assert result['inside_bar'][2] == 1
    
    # Other candles should not be inside bars
    assert result['inside_bar'][0] == 0  # First candle has no previous
    assert result['inside_bar'][1] == 0  # Not inside previous
    assert result['inside_bar'][3] == 0  # Not inside previous


def test_detect_outside_bar():
    """Test outside bar pattern detection"""
    df = pd.DataFrame({
        'open': [10, 12, 9, 14],
        'high': [15, 14, 16, 16],
        'low': [8, 10, 8, 13],
        'close': [12, 13, 15, 15]
    })
    
    result = detect_outside_bar(df)
    assert 'outside_bar' in result.columns
    
    # Index 2 should be an outside bar (high > prev high, low < prev low)
    assert result['outside_bar'][2] == 1
    
    # Other candles should not be outside bars
    assert result['outside_bar'][0] == 0  # First candle has no previous
    assert result['outside_bar'][1] == 0  # Not outside previous
    assert result['outside_bar'][3] == 0  # Not outside previous


def test_detect_engulfing():
    """Test bullish and bearish engulfing patterns"""
    df = pd.DataFrame({
        # Bullish engulfing setup
        'open': [12, 11, 9, 11],
        'high': [13, 12, 12, 14],
        'low': [11, 9, 8, 9],
        'close': [11, 9.5, 11.5, 13]
    })
    
    result = detect_engulfing(df)
    assert 'bullish_engulfing' in result.columns
    assert 'bearish_engulfing' in result.columns
    
    # Index 2 should be a bullish engulfing (opens < prev close, closes > prev open)
    assert result['bullish_engulfing'][2] == 1
    
    # Other candles should not be bullish engulfing
    assert result['bullish_engulfing'][[0, 1, 3]].sum() == 0
    
    # No bearish engulfing in this example
    assert result['bearish_engulfing'].sum() == 0


def test_detect_three_white_soldiers():
    """Test three white soldiers pattern"""
    df = pd.DataFrame({
        'open': [10, 11, 11.5, 12.5, 13.5],
        'high': [13, 14, 15, 16, 17],
        'low': [9, 10, 11, 12, 13],
        'close': [11, 13, 14, 15, 16]
    })
    
    result = detect_three_white_soldiers(df)
    assert 'three_white_soldiers' in result.columns
    
    # Index 4 should detect the pattern (need 3 bullish candles in a row with specific criteria)
    assert result['three_white_soldiers'][4] == 1
    
    # Other indices shouldn't have the pattern
    assert result['three_white_soldiers'][[0, 1, 2, 3]].sum() == 0


def test_detect_three_black_crows():
    """Test three black crows pattern"""
    df = pd.DataFrame({
        'open': [15, 14, 13, 12, 11],
        'high': [17, 14.5, 13.5, 12.5, 11.5],
        'low': [13, 12, 11, 10, 9],
        'close': [14, 12.5, 11.5, 10.5, 9.5]
    })
    
    result = detect_three_black_crows(df)
    assert 'three_black_crows' in result.columns
    
    # Index 4 should detect the pattern (need 3 bearish candles in a row with specific criteria)
    assert result['three_black_crows'][4] == 1
    
    # Other indices shouldn't have the pattern
    assert result['three_black_crows'][[0, 1, 2, 3]].sum() == 0


def test_detect_harami():
    """Test bullish and bearish harami patterns"""
    df = pd.DataFrame({
        # Bullish harami setup (large bearish, small bullish inside)
        'open': [15, 12, 10.5, 10],
        'high': [16, 12.5, 11.5, 11],
        'low': [14, 9, 10, 9],
        'close': [12, 9.5, 11, 10.5]
    })
    
    result = detect_harami(df)
    assert 'bullish_harami' in result.columns
    assert 'bearish_harami' in result.columns
    
    # Index 2 should be a bullish harami
    assert result['bullish_harami'][2] == 1
    
    # Other candles should not be bullish harami
    assert result['bullish_harami'][[0, 1, 3]].sum() == 0
    
    # No bearish harami in this example
    assert result['bearish_harami'].sum() == 0


def test_compute_candle_stats():
    """Test calculation of candlestick statistics"""
    df = pd.DataFrame({
        'open': [10, 12, 14, 15],
        'high': [15, 14, 16, 17],
        'low': [8, 10, 13, 14],
        'close': [12, 14, 15, 16]
    })
    
    result = compute_candle_stats(df)
    
    # Check that basic stats are calculated correctly
    assert 'body_size' in result.columns
    assert 'range_size' in result.columns
    assert 'rel_body_size' in result.columns
    assert 'upper_shadow' in result.columns
    assert 'lower_shadow' in result.columns
    
    # Check a few specific calculations
    assert result['body_size'][0] == 2  # |12 - 10| = 2
    assert result['range_size'][0] == 7  # 15 - 8 = 7
    assert result['rel_body_size'][0] == 2/7  # body_size / range_size
    assert result['upper_shadow'][0] == 3  # 15 - 12 = 3
    assert result['lower_shadow'][0] == 2  # 10 - 8 = 2
    
    # Check that MA columns are created
    for w in [5, 10, 20]:
        assert f'avg_rel_body_{w}' in result.columns
        assert f'avg_upper_shadow_{w}' in result.columns
        assert f'avg_lower_shadow_{w}' in result.columns
        assert f'avg_body_pos_{w}' in result.columns
        assert f'body_momentum_{w}' in result.columns


def test_compatibility_with_original_detectors():
    """Test that the imported detectors match behavior of the original ones"""
    df = pd.DataFrame({
        'open': [10, 12, 14, 15, 16],
        'high': [15, 14, 16, 17, 18],
        'low': [8, 10, 13, 14, 15],
        'close': [12, 14, 15, 16, 15]
    })
    
    # Test doji detection matches
    result1 = features_detect_doji(df.copy())
    result2 = detect_doji(df.copy())
    assert (result1['doji'] == result2['doji']).all()
    
    # Test hammer detection matches
    result1 = features_detect_hammer(df.copy())
    result2 = detect_hammer(df.copy())
    assert (result1['hammer'] == result2['hammer']).all()
    assert (result1['hanging_man'] == result2['hanging_man']).all()
    
    # Test engulfing detection matches
    result1 = features_detect_engulfing(df.copy())
    result2 = detect_engulfing(df.copy())
    assert (result1['bullish_engulfing'] == result2['bullish_engulfing']).all()
    assert (result1['bearish_engulfing'] == result2['bearish_engulfing']).all()


def test_complex_pattern_combination():
    """Test that multiple patterns can be detected in the same dataset"""
    # Create a more complex dataset that should contain multiple patterns
    df = pd.DataFrame({
        # day 0: normal candle
        'open':  [10.0, 11.0, 10.0, 9.0, 10.0, 12.0, 15.0, 14.5, 13.0, 12.0],
        'high':  [15.0, 12.0, 12.0, 10.0, 13.0, 15.0, 16.0, 15.0, 13.5, 12.5],
        'low':   [8.0,  9.0,  8.0,  7.0, 9.0, 11.5, 14.0, 13.0, 11.0, 11.0],
        'close': [11.0, 10.0, 11.5, 9.5, 12.0, 14.0, 15.0, 13.0, 11.5, 12.0]
    })
    
    # Apply all pattern detection functions
    from src.data.candle_patterns import compute_all_candle_patterns
    result = compute_all_candle_patterns(df)
    
    # Check that at least some patterns were detected
    pattern_cols = [col for col in result.columns if col in [
        'doji', 'hammer', 'hanging_man', 'bullish_engulfing', 'bearish_engulfing',
        'shooting_star', 'morning_star', 'evening_star', 'inside_bar', 'outside_bar',
        'tweezer_top', 'tweezer_bottom', 'three_white_soldiers', 'three_black_crows',
        'bullish_harami', 'bearish_harami', 'dark_cloud_cover', 'piercing_line'
    ]]
    
    # Sum detected patterns
    pattern_count = result[pattern_cols].sum().sum()
    
    # At least some patterns should be detected
    assert pattern_count > 0, "No patterns were detected in the test data"
    
    # Check that statistical features were also calculated
    stat_cols = ['body_size', 'range_size', 'rel_body_size', 'upper_shadow', 'lower_shadow']
    for col in stat_cols:
        assert col in result.columns
