"""
Real-time P&L and Performance Dashboard.

This module provides a comprehensive web-based dashboard for monitoring:
- Real-time portfolio P&L and performance metrics
- Risk metrics and position information
- Interactive charts and visualizations
- Real-time data updates and streaming
- Customizable dashboard layouts
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .dashboard import Dashboard
from .metrics_collector import MetricsCollector


class PerformanceDashboard:
    """Real-time P&L and Performance Dashboard using Streamlit."""

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        dashboard: Optional[Dashboard] = None,
        update_interval: float = 1.0,
        max_data_points: int = 1000,
    ) -> None:
        """Initialize the performance dashboard.

        Args:
            metrics_collector: MetricsCollector instance for data
            dashboard: Dashboard instance for system metrics
            update_interval: Update interval in seconds
            max_data_points: Maximum data points to keep in memory
        """
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.dashboard = dashboard or Dashboard(metrics_collector)
        self.update_interval = update_interval
        self.max_data_points = max_data_points
        
        # Initialize session state for dashboard configuration
        if 'dashboard_config' not in st.session_state:
            st.session_state.dashboard_config = self._get_default_config()
        
        # Initialize data storage
        if 'performance_data' not in st.session_state:
            st.session_state.performance_data = {
                'timestamps': [],
                'pnl': [],
                'cumulative_return': [],
                'sharpe_ratio': [],
                'max_drawdown': [],
                'volatility': [],
                'var_95': [],
                'position_count': [],
                'win_rate': [],
            }
        
        if 'position_data' not in st.session_state:
            st.session_state.position_data = []
        
        if 'alert_data' not in st.session_state:
            st.session_state.alert_data = []

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default dashboard configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            'layout': 'grid',  # 'grid', 'single_column', 'custom'
            'theme': 'light',  # 'light', 'dark'
            'auto_refresh': True,
            'refresh_interval': 1.0,
            'charts': {
                'pnl_chart': True,
                'risk_metrics': True,
                'position_overview': True,
                'performance_metrics': True,
                'system_health': True,
                'alerts': True,
            },
            'time_range': '24h',  # '1h', '6h', '24h', '7d', '30d'
            'metrics_display': {
                'show_percentages': True,
                'show_currency': True,
                'currency_symbol': '$',
                'decimal_places': 2,
            }
        }

    def run_dashboard(self) -> None:
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="Trading Performance Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        self._add_custom_css()
        
        # Sidebar configuration
        self._render_sidebar()
        
        # Main dashboard
        self._render_main_dashboard()
        
        # Auto-refresh logic
        if st.session_state.dashboard_config['auto_refresh']:
            time.sleep(st.session_state.dashboard_config['refresh_interval'])
            st.rerun()

    def _add_custom_css(self) -> None:
        """Add custom CSS styling."""
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin: 0.5rem 0;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #1f77b4;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            text-transform: uppercase;
        }
        .positive-change {
            color: #28a745;
        }
        .negative-change {
            color: #dc3545;
        }
        .alert-critical {
            background-color: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
        }
        .alert-warning {
            background-color: #fff3cd;
            border-color: #ffeaa7;
            color: #856404;
        }
        .alert-info {
            background-color: #d1ecf1;
            border-color: #bee5eb;
            color: #0c5460;
        }
        </style>
        """, unsafe_allow_html=True)

    def _render_sidebar(self) -> None:
        """Render the sidebar with configuration options."""
        st.sidebar.title("📊 Dashboard Settings")
        
        # Layout configuration
        st.sidebar.subheader("Layout")
        layout = st.sidebar.selectbox(
            "Dashboard Layout",
            ["grid", "single_column", "custom"],
            index=0
        )
        st.session_state.dashboard_config['layout'] = layout
        
        # Theme configuration
        theme = st.sidebar.selectbox(
            "Theme",
            ["light", "dark"],
            index=0
        )
        st.session_state.dashboard_config['theme'] = theme
        
        # Auto-refresh configuration
        auto_refresh = st.sidebar.checkbox(
            "Auto Refresh",
            value=st.session_state.dashboard_config['auto_refresh']
        )
        st.session_state.dashboard_config['auto_refresh'] = auto_refresh
        
        if auto_refresh:
            refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)",
                min_value=0.5,
                max_value=10.0,
                value=st.session_state.dashboard_config['refresh_interval'],
                step=0.5
            )
            st.session_state.dashboard_config['refresh_interval'] = refresh_interval
        
        # Time range configuration
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["1h", "6h", "24h", "7d", "30d"],
            index=2
        )
        st.session_state.dashboard_config['time_range'] = time_range
        
        # Chart visibility configuration
        st.sidebar.subheader("Charts")
        charts_config = st.session_state.dashboard_config['charts']
        
        for chart_name, is_visible in charts_config.items():
            charts_config[chart_name] = st.sidebar.checkbox(
                chart_name.replace('_', ' ').title(),
                value=is_visible
            )
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            self._update_data()
            st.rerun()

    def _render_main_dashboard(self) -> None:
        """Render the main dashboard content."""
        st.title("📈 Real-Time Trading Performance Dashboard")
        
        # Update data
        self._update_data()
        
        # Header metrics
        self._render_header_metrics()
        
        # Main content based on layout
        if st.session_state.dashboard_config['layout'] == 'grid':
            self._render_grid_layout()
        elif st.session_state.dashboard_config['layout'] == 'single_column':
            self._render_single_column_layout()
        else:
            self._render_custom_layout()

    def _render_header_metrics(self) -> None:
        """Render header metrics row."""
        col1, col2, col3, col4 = st.columns(4)
        
        # Get latest metrics
        trading_metrics = self.dashboard.get_trading_metrics()
        risk_metrics = self.dashboard.get_risk_metrics()
        
        with col1:
            self._render_metric_card(
                "Total P&L",
                f"${trading_metrics['pnl']:,.2f}",
                trading_metrics['pnl'] >= 0
            )
        
        with col2:
            self._render_metric_card(
                "Daily P&L",
                f"${trading_metrics['daily_pnl']:,.2f}",
                trading_metrics['daily_pnl'] >= 0
            )
        
        with col3:
            self._render_metric_card(
                "Sharpe Ratio",
                f"{trading_metrics['sharpe_ratio']:.2f}",
                trading_metrics['sharpe_ratio'] >= 0
            )
        
        with col4:
            self._render_metric_card(
                "Max Drawdown",
                f"{trading_metrics['max_drawdown']:.2%}",
                trading_metrics['max_drawdown'] >= 0
            )

    def _render_metric_card(self, label: str, value: str, is_positive: bool) -> None:
        """Render a metric card.

        Args:
            label: Metric label
            value: Metric value
            is_positive: Whether the value is positive
        """
        color_class = "positive-change" if is_positive else "negative-change"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color_class}">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    def _render_grid_layout(self) -> None:
        """Render dashboard in grid layout."""
        config = st.session_state.dashboard_config['charts']
        
        # First row
        col1, col2 = st.columns(2)
        
        with col1:
            if config['pnl_chart']:
                self._render_pnl_chart()
        
        with col2:
            if config['risk_metrics']:
                self._render_risk_metrics()
        
        # Second row
        col3, col4 = st.columns(2)
        
        with col3:
            if config['position_overview']:
                self._render_position_overview()
        
        with col4:
            if config['performance_metrics']:
                self._render_performance_metrics()
        
        # Third row
        col5, col6 = st.columns(2)
        
        with col5:
            if config['system_health']:
                self._render_system_health()
        
        with col6:
            if config['alerts']:
                self._render_alerts()

    def _render_single_column_layout(self) -> None:
        """Render dashboard in single column layout."""
        config = st.session_state.dashboard_config['charts']
        
        if config['pnl_chart']:
            self._render_pnl_chart()
        
        if config['risk_metrics']:
            self._render_risk_metrics()
        
        if config['position_overview']:
            self._render_position_overview()
        
        if config['performance_metrics']:
            self._render_performance_metrics()
        
        if config['system_health']:
            self._render_system_health()
        
        if config['alerts']:
            self._render_alerts()

    def _render_custom_layout(self) -> None:
        """Render dashboard in custom layout."""
        st.info("Custom layout configuration coming soon!")
        self._render_grid_layout()

    def _render_pnl_chart(self) -> None:
        """Render P&L chart."""
        st.subheader("📊 Portfolio P&L")
        
        if not st.session_state.performance_data['timestamps']:
            st.warning("No P&L data available")
            return
        
        # Create P&L chart
        df = pd.DataFrame({
            'timestamp': st.session_state.performance_data['timestamps'],
            'pnl': st.session_state.performance_data['pnl'],
            'cumulative_return': st.session_state.performance_data['cumulative_return']
        })
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio P&L', 'Cumulative Return'),
            vertical_spacing=0.1
        )
        
        # P&L line
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['pnl'],
                mode='lines',
                name='P&L',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        # Cumulative return
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['cumulative_return'],
                mode='lines',
                name='Cumulative Return',
                line=dict(color='#2ca02c', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _render_risk_metrics(self) -> None:
        """Render risk metrics."""
        st.subheader("⚠️ Risk Metrics")
        
        risk_metrics = self.dashboard.get_risk_metrics()
        
        # Create risk metrics visualization
        metrics_data = {
            'VaR (95%)': risk_metrics['var_95'],
            'CVaR (95%)': risk_metrics['cvar_95'],
            'Volatility': risk_metrics['volatility'],
            'Beta': risk_metrics['beta'],
            'Exposure': risk_metrics['current_exposure'],
            'Concentration': risk_metrics['position_concentration']
        }
        
        # Create gauge charts
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=list(metrics_data.keys())
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (metric_name, value) in enumerate(metrics_data.items()):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=abs(value),
                    title={'text': metric_name},
                    gauge={
                        'axis': {'range': [None, max(abs(value) * 1.2, 1)]},
                        'bar': {'color': colors[i]},
                        'steps': [
                            {'range': [0, abs(value) * 0.5], 'color': "lightgray"},
                            {'range': [abs(value) * 0.5, abs(value)], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': abs(value)
                        }
                    }
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    def _render_position_overview(self) -> None:
        """Render position overview."""
        st.subheader("📋 Position Overview")
        
        if not st.session_state.position_data:
            st.info("No position data available")
            return
        
        # Create position table
        df = pd.DataFrame(st.session_state.position_data)
        
        if not df.empty:
            # Display position table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
            
            # Position summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Positions", len(df))
            
            with col2:
                total_value = df['value'].sum() if 'value' in df.columns else 0
                st.metric("Total Value", f"${total_value:,.2f}")
            
            with col3:
                avg_size = df['size'].mean() if 'size' in df.columns else 0
                st.metric("Avg Position Size", f"{avg_size:.2f}")

    def _render_performance_metrics(self) -> None:
        """Render performance metrics."""
        st.subheader("📈 Performance Metrics")
        
        trading_metrics = self.dashboard.get_trading_metrics()
        
        # Create performance metrics visualization
        metrics = {
            'Total Return': trading_metrics['total_return'],
            'Win Rate': trading_metrics['win_rate'],
            'Total Trades': trading_metrics['total_trades'],
            'Open Positions': trading_metrics['open_positions']
        }
        
        # Create bar chart
        fig = px.bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            title="Performance Metrics",
            color=list(metrics.values()),
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sharpe Ratio", f"{trading_metrics['sharpe_ratio']:.2f}")
            st.metric("Max Drawdown", f"{trading_metrics['max_drawdown']:.2%}")
        
        with col2:
            st.metric("Win Rate", f"{trading_metrics['win_rate']:.1%}")
            st.metric("Total Trades", trading_metrics['total_trades'])

    def _render_system_health(self) -> None:
        """Render system health metrics."""
        st.subheader("🏥 System Health")
        
        health_metrics = self.dashboard.get_system_health()
        
        # Create system health visualization
        metrics = {
            'CPU Usage': health_metrics['cpu_usage'],
            'Memory Usage': health_metrics['memory_usage'],
            'Disk Usage': health_metrics['disk_usage'],
            'Network Latency': health_metrics['network_latency'],
            'Error Rate': health_metrics['error_rate'],
            'Response Time': health_metrics['response_time']
        }
        
        # Create gauge charts
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=list(metrics.keys())
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (metric_name, value) in enumerate(metrics.items()):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            # Determine color based on value
            if metric_name in ['CPU Usage', 'Memory Usage', 'Disk Usage']:
                color = 'red' if value > 80 else 'orange' if value > 60 else 'green'
            else:
                color = 'red' if value > 100 else 'orange' if value > 50 else 'green'
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': metric_name},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    def _render_alerts(self) -> None:
        """Render alerts."""
        st.subheader("🚨 Alerts")
        
        alerts = self.dashboard.get_recent_alerts(limit=10)
        
        if not alerts:
            st.success("No active alerts")
            return
        
        for alert in alerts:
            severity = alert['severity']
            alert_class = f"alert-{severity.lower()}"
            
            st.markdown(f"""
            <div class="{alert_class}" style="padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;">
                <strong>{alert['title']}</strong><br>
                {alert['message']}<br>
                <small>Source: {alert['source']} | {datetime.fromtimestamp(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)

    def _update_data(self) -> None:
        """Update dashboard data."""
        current_time = time.time()
        
        # Get latest metrics
        trading_metrics = self.dashboard.get_trading_metrics()
        risk_metrics = self.dashboard.get_risk_metrics()
        
        # Update performance data
        st.session_state.performance_data['timestamps'].append(current_time)
        st.session_state.performance_data['pnl'].append(trading_metrics['pnl'])
        st.session_state.performance_data['cumulative_return'].append(trading_metrics['total_return'])
        st.session_state.performance_data['sharpe_ratio'].append(trading_metrics['sharpe_ratio'])
        st.session_state.performance_data['max_drawdown'].append(trading_metrics['max_drawdown'])
        st.session_state.performance_data['volatility'].append(risk_metrics['volatility'])
        st.session_state.performance_data['var_95'].append(risk_metrics['var_95'])
        st.session_state.performance_data['position_count'].append(trading_metrics['open_positions'])
        st.session_state.performance_data['win_rate'].append(trading_metrics['win_rate'])
        
        # Limit data points
        for key in st.session_state.performance_data:
            if len(st.session_state.performance_data[key]) > self.max_data_points:
                st.session_state.performance_data[key] = st.session_state.performance_data[key][-self.max_data_points:]
        
        # Update position data (mock data for demonstration)
        self._update_position_data()
        
        # Update alert data
        st.session_state.alert_data = self.dashboard.get_recent_alerts(limit=10)

    def _update_position_data(self) -> None:
        """Update position data (mock implementation)."""
        # This would typically come from your trading system
        # For demonstration, we'll create mock position data
        mock_positions = [
            {
                'symbol': 'AAPL',
                'side': 'long',
                'size': 100,
                'entry_price': 150.0,
                'current_price': 152.5,
                'pnl': 250.0,
                'value': 15250.0
            },
            {
                'symbol': 'GOOGL',
                'side': 'short',
                'size': 50,
                'entry_price': 2800.0,
                'current_price': 2750.0,
                'pnl': 2500.0,
                'value': 137500.0
            },
            {
                'symbol': 'TSLA',
                'side': 'long',
                'size': 200,
                'entry_price': 200.0,
                'current_price': 195.0,
                'pnl': -1000.0,
                'value': 39000.0
            }
        ]
        
        st.session_state.position_data = mock_positions

    def save_configuration(self, filepath: str) -> None:
        """Save dashboard configuration to file.

        Args:
            filepath: Path to save configuration
        """
        with open(filepath, 'w') as f:
            json.dump(st.session_state.dashboard_config, f, indent=2)

    def load_configuration(self, filepath: str) -> None:
        """Load dashboard configuration from file.

        Args:
            filepath: Path to configuration file
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
            st.session_state.dashboard_config.update(config)

    def export_data(self, filepath: str) -> None:
        """Export dashboard data to file.

        Args:
            filepath: Path to save data
        """
        export_data = {
            'performance_data': st.session_state.performance_data,
            'position_data': st.session_state.position_data,
            'alert_data': st.session_state.alert_data,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)


def run_performance_dashboard(
    metrics_collector: Optional[MetricsCollector] = None,
    dashboard: Optional[Dashboard] = None,
    config_file: Optional[str] = None
) -> None:
    """Run the performance dashboard.

    Args:
        metrics_collector: Optional MetricsCollector instance
        dashboard: Optional Dashboard instance
        config_file: Optional configuration file path
    """
    dashboard_instance = PerformanceDashboard(
        metrics_collector=metrics_collector,
        dashboard=dashboard
    )
    
    if config_file:
        dashboard_instance.load_configuration(config_file)
    
    dashboard_instance.run_dashboard()


if __name__ == "__main__":
    run_performance_dashboard()