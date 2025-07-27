-- Database initialization script for Trading RL Platform
-- Creates necessary tables for storing trading data, models, and performance metrics

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;

-- Create enum types
CREATE TYPE model_status AS ENUM ('training', 'trained', 'deployed', 'archived');
CREATE TYPE trade_side AS ENUM ('buy', 'sell');
CREATE TYPE order_status AS ENUM ('pending', 'filled', 'cancelled', 'rejected');

-- Market data table (using TimescaleDB for time-series optimization)
CREATE TABLE market_data (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume BIGINT,
    source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable for better time-series performance
SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);

-- Create indexes for better query performance
CREATE INDEX idx_market_data_symbol_timestamp ON market_data (symbol, timestamp DESC);
CREATE INDEX idx_market_data_timestamp ON market_data (timestamp DESC);

-- Models registry table
CREATE TABLE models (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    algorithm VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    status model_status DEFAULT 'training',
    file_path TEXT,
    config JSONB,
    training_metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, version)
);

-- Trading sessions table
CREATE TABLE trading_sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    model_id UUID REFERENCES models(id),
    broker VARCHAR(50) NOT NULL,
    symbols TEXT[] NOT NULL,
    initial_capital DECIMAL(15,2) NOT NULL,
    paper_trading BOOLEAN DEFAULT TRUE,
    risk_profile VARCHAR(50),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    final_portfolio_value DECIMAL(15,2),
    total_return DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    config JSONB
);

-- Trades table
CREATE TABLE trades (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES trading_sessions(id),
    symbol VARCHAR(20) NOT NULL,
    side trade_side NOT NULL,
    quantity DECIMAL(15,6) NOT NULL,
    price DECIMAL(12,4) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    commission DECIMAL(8,4) DEFAULT 0,
    slippage DECIMAL(8,4) DEFAULT 0,
    pnl DECIMAL(12,4),
    portfolio_value DECIMAL(15,2),
    signal_data JSONB
);

-- Orders table
CREATE TABLE orders (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES trading_sessions(id),
    symbol VARCHAR(20) NOT NULL,
    side trade_side NOT NULL,
    quantity DECIMAL(15,6) NOT NULL,
    order_type VARCHAR(20) DEFAULT 'market',
    limit_price DECIMAL(12,4),
    stop_price DECIMAL(12,4),
    status order_status DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    filled_at TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,
    filled_quantity DECIMAL(15,6) DEFAULT 0,
    avg_fill_price DECIMAL(12,4),
    broker_order_id VARCHAR(255),
    error_message TEXT
);

-- Performance metrics table (time-series data)
CREATE TABLE performance_metrics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES trading_sessions(id),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    portfolio_value DECIMAL(15,2) NOT NULL,
    cash DECIMAL(15,2) NOT NULL,
    positions_value DECIMAL(15,2) NOT NULL,
    daily_return DECIMAL(8,4),
    cumulative_return DECIMAL(8,4),
    drawdown DECIMAL(8,4),
    volatility DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    positions JSONB
);

-- Convert to hypertable
SELECT create_hypertable('performance_metrics', 'timestamp', if_not_exists => TRUE);

-- Risk events table
CREATE TABLE risk_events (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES trading_sessions(id),
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    description TEXT NOT NULL,
    triggered_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    metadata JSONB
);

-- System alerts table
CREATE TABLE system_alerts (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    alert_type VARCHAR(100) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    triggered_at TIMESTAMPTZ DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    metadata JSONB
);

-- Backtests table
CREATE TABLE backtests (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    model_id UUID REFERENCES models(id),
    name VARCHAR(255) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(15,2) NOT NULL,
    final_value DECIMAL(15,2),
    total_return DECIMAL(8,4),
    annualized_return DECIMAL(8,4),
    volatility DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    sortino_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    win_rate DECIMAL(5,2),
    profit_factor DECIMAL(8,4),
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    config JSONB,
    results JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Data quality metrics table
CREATE TABLE data_quality_metrics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    source VARCHAR(50) NOT NULL,
    completeness_score DECIMAL(5,4),
    accuracy_score DECIMAL(5,4),
    timeliness_score DECIMAL(5,4),
    consistency_score DECIMAL(5,4),
    overall_score DECIMAL(5,4),
    issues JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, date, source)
);

-- Feature engineering metadata
CREATE TABLE feature_sets (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    feature_list TEXT[] NOT NULL,
    config JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model-feature mapping
CREATE TABLE model_features (
    model_id UUID REFERENCES models(id),
    feature_set_id UUID REFERENCES feature_sets(id),
    PRIMARY KEY (model_id, feature_set_id)
);

-- Create indexes for better performance
CREATE INDEX idx_trades_session_timestamp ON trades (session_id, timestamp DESC);
CREATE INDEX idx_trades_symbol_timestamp ON trades (symbol, timestamp DESC);
CREATE INDEX idx_orders_session_status ON orders (session_id, status);
CREATE INDEX idx_performance_session_timestamp ON performance_metrics (session_id, timestamp DESC);
CREATE INDEX idx_risk_events_session_severity ON risk_events (session_id, severity, triggered_at DESC);
CREATE INDEX idx_system_alerts_type_severity ON system_alerts (alert_type, severity, triggered_at DESC);
CREATE INDEX idx_backtests_model_created ON backtests (model_id, created_at DESC);

-- Create views for common queries
CREATE VIEW active_trading_sessions AS
SELECT
    ts.*,
    m.name as model_name,
    m.algorithm,
    COUNT(t.id) as total_trades,
    MAX(pm.portfolio_value) as current_portfolio_value,
    MAX(pm.cumulative_return) as current_return
FROM trading_sessions ts
LEFT JOIN models m ON ts.model_id = m.id
LEFT JOIN trades t ON ts.id = t.session_id
LEFT JOIN performance_metrics pm ON ts.id = pm.session_id
WHERE ts.ended_at IS NULL
GROUP BY ts.id, m.name, m.algorithm;

CREATE VIEW model_performance_summary AS
SELECT
    m.id,
    m.name,
    m.algorithm,
    m.status,
    COUNT(DISTINCT ts.id) as total_sessions,
    COUNT(DISTINCT b.id) as total_backtests,
    COALESCE(AVG(ts.total_return), 0) as avg_live_return,
    COALESCE(AVG(ts.sharpe_ratio), 0) as avg_live_sharpe,
    COALESCE(AVG(b.total_return), 0) as avg_backtest_return,
    COALESCE(AVG(b.sharpe_ratio), 0) as avg_backtest_sharpe,
    m.created_at
FROM models m
LEFT JOIN trading_sessions ts ON m.id = ts.model_id
LEFT JOIN backtests b ON m.id = b.model_id
GROUP BY m.id, m.name, m.algorithm, m.status, m.created_at;

-- Create materialized view for daily performance summary
CREATE MATERIALIZED VIEW daily_performance_summary AS
SELECT
    ts.id as session_id,
    ts.broker,
    ts.paper_trading,
    DATE(pm.timestamp) as date,
    FIRST(pm.portfolio_value ORDER BY pm.timestamp) as opening_value,
    LAST(pm.portfolio_value ORDER BY pm.timestamp) as closing_value,
    MAX(pm.portfolio_value) as high_value,
    MIN(pm.portfolio_value) as low_value,
    LAST(pm.daily_return ORDER BY pm.timestamp) as daily_return,
    LAST(pm.cumulative_return ORDER BY pm.timestamp) as cumulative_return,
    LAST(pm.drawdown ORDER BY pm.timestamp) as drawdown,
    COUNT(t.id) as trades_count,
    SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(t.pnl) as total_pnl
FROM trading_sessions ts
JOIN performance_metrics pm ON ts.id = pm.session_id
LEFT JOIN trades t ON ts.id = t.session_id AND DATE(t.timestamp) = DATE(pm.timestamp)
GROUP BY ts.id, ts.broker, ts.paper_trading, DATE(pm.timestamp);

-- Create index on materialized view
CREATE INDEX idx_daily_performance_session_date ON daily_performance_summary (session_id, date DESC);

-- Create function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_daily_performance_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW daily_performance_summary;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to update model updated_at timestamp
CREATE OR REPLACE FUNCTION update_model_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_models_timestamp
    BEFORE UPDATE ON models
    FOR EACH ROW
    EXECUTE FUNCTION update_model_timestamp();

-- Create function for calculating portfolio metrics
CREATE OR REPLACE FUNCTION calculate_portfolio_metrics(
    p_session_id UUID,
    p_start_date TIMESTAMPTZ DEFAULT NULL,
    p_end_date TIMESTAMPTZ DEFAULT NULL
)
RETURNS TABLE (
    total_return DECIMAL(8,4),
    annualized_return DECIMAL(8,4),
    volatility DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    win_rate DECIMAL(5,2)
) AS $$
BEGIN
    RETURN QUERY
    WITH daily_returns AS (
        SELECT
            daily_return,
            drawdown
        FROM performance_metrics
        WHERE session_id = p_session_id
          AND (p_start_date IS NULL OR timestamp >= p_start_date)
          AND (p_end_date IS NULL OR timestamp <= p_end_date)
          AND daily_return IS NOT NULL
        ORDER BY timestamp
    ),
    trade_stats AS (
        SELECT
            COUNT(*) as total_trades_count,
            COUNT(*) FILTER (WHERE pnl > 0) as winning_trades_count
        FROM trades
        WHERE session_id = p_session_id
          AND (p_start_date IS NULL OR timestamp >= p_start_date)
          AND (p_end_date IS NULL OR timestamp <= p_end_date)
    )
    SELECT
        -- Total return (last cumulative return)
        COALESCE((
            SELECT cumulative_return
            FROM performance_metrics
            WHERE session_id = p_session_id
            ORDER BY timestamp DESC
            LIMIT 1
        ), 0)::DECIMAL(8,4) as total_return,

        -- Annualized return
        (COALESCE(AVG(dr.daily_return), 0) * 252)::DECIMAL(8,4) as annualized_return,

        -- Volatility (annualized)
        (COALESCE(STDDEV(dr.daily_return), 0) * SQRT(252))::DECIMAL(8,4) as volatility,

        -- Sharpe ratio (assuming 2% risk-free rate)
        CASE
            WHEN COALESCE(STDDEV(dr.daily_return), 0) > 0 THEN
                ((COALESCE(AVG(dr.daily_return), 0) - 0.02/252) / STDDEV(dr.daily_return) * SQRT(252))::DECIMAL(8,4)
            ELSE 0::DECIMAL(8,4)
        END as sharpe_ratio,

        -- Maximum drawdown
        COALESCE(MIN(dr.drawdown), 0)::DECIMAL(8,4) as max_drawdown,

        -- Win rate
        CASE
            WHEN ts.total_trades_count > 0 THEN
                (ts.winning_trades_count::DECIMAL / ts.total_trades_count * 100)::DECIMAL(5,2)
            ELSE 0::DECIMAL(5,2)
        END as win_rate

    FROM daily_returns dr
    CROSS JOIN trade_stats ts;
END;
$$ LANGUAGE plpgsql;

-- Insert initial data
INSERT INTO feature_sets (name, description, feature_list, config) VALUES
('basic_technical', 'Basic technical indicators', ARRAY['sma_20', 'sma_50', 'rsi', 'macd', 'bollinger_bands'], '{"indicators": ["sma", "rsi", "macd", "bollinger"]}'),
('advanced_technical', 'Advanced technical indicators', ARRAY['sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'macd', 'bollinger_bands', 'atr', 'stochastic', 'williams_r'], '{"indicators": ["sma", "ema", "rsi", "macd", "bollinger", "atr", "stochastic", "williams"]}'),
('fundamental', 'Fundamental analysis features', ARRAY['pe_ratio', 'pb_ratio', 'debt_to_equity', 'roe', 'revenue_growth'], '{"data_sources": ["financial_statements", "earnings_reports"]}'),
('sentiment', 'Market sentiment features', ARRAY['news_sentiment', 'social_sentiment', 'vix', 'put_call_ratio'], '{"sources": ["news", "twitter", "reddit", "options_data"]}');

-- Create scheduled job to refresh materialized views (requires pg_cron extension)
-- SELECT cron.schedule('refresh-daily-performance', '0 1 * * *', 'SELECT refresh_daily_performance_summary();');

COMMENT ON DATABASE trading_rl IS 'Trading RL Platform - Production database for storing market data, models, trading sessions, and performance metrics';
