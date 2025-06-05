"""Tests for live data fetching functionality."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.data.live import fetch_live_data


class TestFetchLiveData:
    """Test the fetch_live_data function."""
    
    def test_fetch_live_data_basic(self):
        """Test basic functionality of fetch_live_data."""
        # Mock yfinance Ticker and its history method
        mock_history_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [95.0, 96.0, 97.0],
            'Close': [101.0, 102.0, 103.0],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history_data
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            result = fetch_live_data("AAPL", "2023-01-01", "2023-01-03", "day")
            
            # Check that yfinance was called correctly
            mock_ticker.history.assert_called_once_with(
                start="2023-01-01",
                end="2023-01-03",
                interval="1d"
            )
            
            # Check result structure
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            expected_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            assert list(result.columns) == expected_columns
            
            # Check data types
            assert result['open'].dtype in [np.float64, np.float32]
            assert result['volume'].dtype in [np.int64, np.float64]
            assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
    
    def test_fetch_live_data_timestep_mapping(self):
        """Test timestep parameter mapping to yfinance intervals."""
        mock_history_data = pd.DataFrame({
            'Open': [100.0], 'High': [105.0], 'Low': [95.0],
            'Close': [101.0], 'Volume': [1000]
        }, index=pd.date_range('2023-01-01', periods=1, freq='D'))
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history_data
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            # Test day mapping
            fetch_live_data("AAPL", "2023-01-01", "2023-01-02", "day")
            mock_ticker.history.assert_called_with(
                start="2023-01-01", end="2023-01-02", interval="1d"
            )
            
            # Test hour mapping
            fetch_live_data("AAPL", "2023-01-01", "2023-01-02", "hour")
            mock_ticker.history.assert_called_with(
                start="2023-01-01", end="2023-01-02", interval="1h"
            )
            
            # Test minute mapping
            fetch_live_data("AAPL", "2023-01-01", "2023-01-02", "minute")
            mock_ticker.history.assert_called_with(
                start="2023-01-01", end="2023-01-02", interval="1m"
            )
    
    def test_fetch_live_data_custom_interval(self):
        """Test fetch_live_data with custom interval string."""
        mock_history_data = pd.DataFrame({
            'Open': [100.0], 'High': [105.0], 'Low': [95.0],
            'Close': [101.0], 'Volume': [1000]
        }, index=pd.date_range('2023-01-01', periods=1, freq='D'))
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history_data
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            # Test custom interval (not in mapping)
            fetch_live_data("AAPL", "2023-01-01", "2023-01-02", "5m")
            mock_ticker.history.assert_called_with(
                start="2023-01-01", end="2023-01-02", interval="5m"
            )
    
    def test_fetch_live_data_empty_response(self):
        """Test fetch_live_data when yfinance returns empty DataFrame."""
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()  # Empty DataFrame
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            result = fetch_live_data("INVALID", "2023-01-01", "2023-01-02", "day")
            
            # Should return DataFrame with correct columns but no data
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            assert list(result.columns) == expected_columns
    
    def test_fetch_live_data_timezone_handling(self):
        """Test that timezone information is properly handled."""
        # Create data with timezone
        tz_index = pd.date_range('2023-01-01', periods=2, freq='D', tz='US/Eastern')
        mock_history_data = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [95.0, 96.0],
            'Close': [101.0, 102.0],
            'Volume': [1000, 1100]
        }, index=tz_index)
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history_data
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            result = fetch_live_data("AAPL", "2023-01-01", "2023-01-02", "day")
            
            # Timezone should be removed
            assert result['timestamp'].dt.tz is None
    
    def test_fetch_live_data_timezone_already_none(self):
        """Test timezone handling when index already has no timezone."""
        # Create data without timezone
        naive_index = pd.date_range('2023-01-01', periods=2, freq='D')
        mock_history_data = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [95.0, 96.0],
            'Close': [101.0, 102.0],
            'Volume': [1000, 1100]
        }, index=naive_index)
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history_data
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            result = fetch_live_data("AAPL", "2023-01-01", "2023-01-02", "day")
            
            # Should handle gracefully when no timezone info
            assert result['timestamp'].dt.tz is None
            assert len(result) == 2
    
    def test_fetch_live_data_column_selection(self):
        """Test that only OHLCV columns are selected from yfinance data."""
        # Mock data with extra columns
        mock_history_data = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [95.0, 96.0],
            'Close': [101.0, 102.0],
            'Volume': [1000, 1100],
            'Dividends': [0.0, 0.0],  # Extra column
            'Stock Splits': [0.0, 0.0]  # Extra column
        }, index=pd.date_range('2023-01-01', periods=2, freq='D'))
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history_data
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            result = fetch_live_data("AAPL", "2023-01-01", "2023-01-02", "day")
            
            # Should only have OHLCV + timestamp columns
            expected_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            assert list(result.columns) == expected_columns
            assert 'dividends' not in result.columns
            assert 'stock_splits' not in result.columns
    
    def test_fetch_live_data_index_reset(self):
        """Test that the index is properly reset in the result."""
        mock_history_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [95.0, 96.0, 97.0],
            'Close': [101.0, 102.0, 103.0],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history_data
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            result = fetch_live_data("AAPL", "2023-01-01", "2023-01-03", "day")
            
            # Index should be reset to integer index
            assert isinstance(result.index, pd.RangeIndex)
            assert list(result.index) == [0, 1, 2]
            
            # Timestamp should be a regular column
            assert 'timestamp' in result.columns
            assert len(result['timestamp']) == 3
    
    def test_fetch_live_data_different_symbols(self):
        """Test fetch_live_data with different stock symbols."""
        symbols_to_test = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        
        mock_history_data = pd.DataFrame({
            'Open': [100.0], 'High': [105.0], 'Low': [95.0],
            'Close': [101.0], 'Volume': [1000]
        }, index=pd.date_range('2023-01-01', periods=1, freq='D'))
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history_data
        
        with patch('yfinance.Ticker', return_value=mock_ticker) as mock_yf:
            for symbol in symbols_to_test:
                result = fetch_live_data(symbol, "2023-01-01", "2023-01-02", "day")
                
                # Verify the symbol was passed to yfinance
                mock_yf.assert_called_with(symbol)
                
                # Verify result structure is consistent
                assert len(result) == 1
                assert 'timestamp' in result.columns
    
    def test_fetch_live_data_default_timestep(self):
        """Test fetch_live_data with default timestep parameter."""
        mock_history_data = pd.DataFrame({
            'Open': [100.0], 'High': [105.0], 'Low': [95.0],
            'Close': [101.0], 'Volume': [1000]
        }, index=pd.date_range('2023-01-01', periods=1, freq='D'))
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history_data
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            # Call without timestep parameter (should default to "day")
            result = fetch_live_data("AAPL", "2023-01-01", "2023-01-02")
            
            mock_ticker.history.assert_called_with(
                start="2023-01-01", end="2023-01-02", interval="1d"
            )
            assert len(result) == 1


class TestFetchLiveDataErrorHandling:
    """Test error handling in fetch_live_data."""
    
    def test_fetch_live_data_yfinance_exception(self):
        """Test fetch_live_data when yfinance raises an exception."""
        mock_ticker = Mock()
        mock_ticker.history.side_effect = Exception("Network error")
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            with pytest.raises(Exception, match="Network error"):
                fetch_live_data("AAPL", "2023-01-01", "2023-01-02", "day")
    
    def test_fetch_live_data_invalid_dates(self):
        """Test fetch_live_data with invalid date formats."""
        mock_ticker = Mock()
        mock_ticker.history.side_effect = ValueError("Invalid date format")
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            with pytest.raises(ValueError, match="Invalid date format"):
                fetch_live_data("AAPL", "invalid-date", "2023-01-02", "day")
    
    def test_fetch_live_data_missing_columns(self):
        """Test fetch_live_data when yfinance returns data with missing columns."""
        # Mock data missing some OHLCV columns
        mock_history_data = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            # Missing 'Low', 'Close', 'Volume'
        }, index=pd.date_range('2023-01-01', periods=2, freq='D'))
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history_data
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            with pytest.raises(KeyError):
                fetch_live_data("AAPL", "2023-01-01", "2023-01-02", "day")


class TestFetchLiveDataDataTypes:
    """Test data types and validation in fetch_live_data."""
    
    def test_fetch_live_data_numeric_types(self):
        """Test that numeric columns have appropriate data types."""
        mock_history_data = pd.DataFrame({
            'Open': [100.5, 101.7],
            'High': [105.2, 106.8],
            'Low': [95.1, 96.3],
            'Close': [101.4, 102.9],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2023-01-01', periods=2, freq='D'))
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history_data
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            result = fetch_live_data("AAPL", "2023-01-01", "2023-01-02", "day")
            
            # Check that price columns are numeric
            for col in ['open', 'high', 'low', 'close']:
                assert pd.api.types.is_numeric_dtype(result[col])
                assert result[col].notna().all()  # No NaN values
            
            # Volume should be numeric (integer or float)
            assert pd.api.types.is_numeric_dtype(result['volume'])
            
            # Timestamp should be datetime
            assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
    
    def test_fetch_live_data_positive_values(self):
        """Test that price and volume values are positive."""
        mock_history_data = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [95.0, 96.0],
            'Close': [101.0, 102.0],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2023-01-01', periods=2, freq='D'))
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history_data
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            result = fetch_live_data("AAPL", "2023-01-01", "2023-01-02", "day")
            
            # All price values should be positive
            for col in ['open', 'high', 'low', 'close']:
                assert (result[col] > 0).all()
            
            # Volume should be non-negative
            assert (result['volume'] >= 0).all()


@pytest.mark.integration
class TestFetchLiveDataIntegration:
    """Integration tests that use real yfinance calls (skipped by default)."""
    
    @pytest.mark.skip(reason="Requires internet connection - enable for integration testing")
    def test_fetch_live_data_real_call(self):
        """Test fetch_live_data with actual yfinance call."""
        # This test would make real API calls to Yahoo Finance
        result = fetch_live_data("AAPL", "2023-01-01", "2023-01-05", "day")
        
        # Basic validations for real data
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume', 'timestamp'])
        
        # Data quality checks
        assert (result['high'] >= result['low']).all()
        assert (result['high'] >= result[['open', 'close']].max(axis=1)).all()
        assert (result['low'] <= result[['open', 'close']].min(axis=1)).all()
    
    @pytest.mark.skip(reason="Requires internet connection - enable for integration testing")
    def test_fetch_live_data_real_minute_data(self):
        """Test fetch_live_data with real minute-level data."""
        # Get recent minute data (only available for recent dates)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        result = fetch_live_data("AAPL", start_date, end_date, "minute")
        
        # Should have minute-level granularity
        assert isinstance(result, pd.DataFrame)
        if len(result) > 1:
            # Check that timestamps are minute-level
            time_diffs = result['timestamp'].diff().dropna()
            # Most differences should be 1 minute (allowing for market hours)
            minute_diffs = time_diffs.dt.total_seconds() / 60
            assert (minute_diffs == 1.0).any()  # At least some 1-minute intervals


class TestFetchLiveDataCompatibility:
    """Test compatibility with other data fetching functions."""
    
    def test_fetch_live_data_schema_compatibility(self):
        """Test that fetch_live_data returns same schema as other fetch functions."""
        mock_history_data = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [95.0, 96.0],
            'Close': [101.0, 102.0],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2023-01-01', periods=2, freq='D'))
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history_data
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            result = fetch_live_data("AAPL", "2023-01-01", "2023-01-02", "day")
            
            # Should match schema of fetch_historical_data and fetch_synthetic_data
            expected_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            assert list(result.columns) == expected_columns
            
            # Column order should be consistent
            assert result.columns[0] == 'open'
            assert result.columns[-1] == 'timestamp'
