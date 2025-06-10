"""
Integration tests for the data pipeline with advanced candle patterns.
"""
import pandas as pd
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.integration

from src.data.features import generate_features
@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(42)
    torch.manual_seed(42)

from src.data.synthetic import generate_gbm_prices


def test_generate_features_with_advanced_candles():
    """Test generating features with advanced candle patterns enabled"""
    # Create synthetic price data
    n_days = 100
    df = generate_gbm_prices(
        n_days=n_days,
        mu=0.0001,
        sigma=0.01,
        s0=100.0
    )
    
    # Generate features with advanced candles
    result = generate_features(df, advanced_candles=True)
    
    # Verify basic technical indicators
    assert 'log_return' in result.columns
    assert 'rsi_14' in result.columns
    assert 'vol_20' in result.columns
    
    # Verify standard candle patterns
    assert 'doji' in result.columns
    assert 'hammer' in result.columns
    assert 'bullish_engulfing' in result.columns
    
    # Verify advanced candle patterns
    assert 'inside_bar' in result.columns
    assert 'outside_bar' in result.columns
    assert 'tweezer_top' in result.columns
    assert 'tweezer_bottom' in result.columns
    assert 'three_white_soldiers' in result.columns
    assert 'three_black_crows' in result.columns
    
    # Verify candle statistics
    assert 'body_size' in result.columns
    assert 'range_size' in result.columns
    assert 'rel_body_size' in result.columns
    assert 'upper_shadow' in result.columns
    assert 'lower_shadow' in result.columns
    
    # Verify candle statistics moving averages
    assert 'avg_rel_body_5' in result.columns
    assert 'body_momentum_20' in result.columns
    
    # Check that we have more rows than the warmup period
    assert len(result) > 0
    assert len(result) < n_days  # Some rows should be dropped due to warmup
    
    # At least some patterns should be detected
    pattern_cols = [col for col in result.columns if col in [
        'doji', 'hammer', 'hanging_man', 'bullish_engulfing', 'bearish_engulfing',
        'inside_bar', 'outside_bar', 'tweezer_top', 'tweezer_bottom', 
        'three_white_soldiers', 'three_black_crows', 'bullish_harami', 'bearish_harami'
    ]]
    
    assert sum(result[pattern_cols].sum()) > 0, "No patterns detected in synthetic data"


def test_generate_features_with_basic_candles():
    """Test generating features with just basic candle patterns"""
    # Create synthetic price data
    n_days = 100
    df = generate_gbm_prices(
        n_days=n_days,
        mu=0.0001,
        sigma=0.01,
        s0=100.0
    )
    
    # Generate features with only basic candles
    result = generate_features(df, advanced_candles=False)
    
    # Verify basic technical indicators
    assert 'log_return' in result.columns
    assert 'rsi_14' in result.columns
    assert 'vol_20' in result.columns
    
    # Verify standard candle patterns
    assert 'doji' in result.columns
    assert 'hammer' in result.columns
    assert 'bullish_engulfing' in result.columns
    
    # Advanced candle patterns should not be present
    assert 'inside_bar' not in result.columns
    assert 'outside_bar' not in result.columns
    assert 'tweezer_top' not in result.columns
    
    # Candle statistics should not be present
    assert 'body_size' not in result.columns
    assert 'range_size' not in result.columns
    assert 'rel_body_size' not in result.columns
    
    # Check that we have more rows than the warmup period
    assert len(result) > 0
    assert len(result) < n_days  # Some rows should be dropped due to warmup


def test_pipeline_memory_usage():
    """Test that the feature generation pipeline doesn't use excessive memory"""
    import psutil
    import os
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create a large synthetic dataset
    n_days = 200
    df = generate_gbm_prices(
        n_days=n_days,
        mu=0.0001,
        sigma=0.01,
        s0=100.0
    )
    
    # Generate features with advanced candles
    result = generate_features(df, advanced_candles=True)
    
    # Get final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable
    assert memory_increase < 500, f"Memory usage increased by {memory_increase}MB"
