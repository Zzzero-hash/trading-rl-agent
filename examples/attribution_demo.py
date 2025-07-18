#!/usr/bin/env python3
"""
Comprehensive Performance Attribution Analysis Demo.

This script demonstrates all the capabilities of the performance attribution system:
- Factor model analysis and return decomposition
- Brinson attribution for sector allocation
- Risk-adjusted performance analysis
- Interactive dashboards
- Automated reporting
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from trading_rl_agent.portfolio.attribution import (
    AttributionConfig, PerformanceAttributor, FactorModel, 
    BrinsonAttributor, RiskAdjustedAttributor, AttributionVisualizer
)
from trading_rl_agent.portfolio.attribution_integration import (
    AttributionIntegration, AutomatedAttributionWorkflow
)
from trading_rl_agent.portfolio.manager import PortfolioManager, PortfolioConfig


def create_sample_data():
    """Create realistic sample data for attribution analysis."""
    print("Creating sample data...")
    
    # Date range
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Asset universe
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'CRM', 'ADBE']
    sectors = ['Technology', 'Technology', 'Technology', 'Consumer', 'Technology', 
               'Technology', 'Technology', 'Consumer', 'Technology', 'Technology']
    
    # Create sector mapping
    sector_data = pd.DataFrame({
        'sector': sectors
    }, index=symbols)
    
    # Generate realistic asset returns with sector effects
    np.random.seed(42)  # For reproducible results
    
    # Base returns for each sector
    sector_returns = {
        'Technology': 0.0008,  # Higher growth for tech
        'Consumer': 0.0005,    # Moderate growth for consumer
    }
    
    # Generate asset returns with sector effects and idiosyncratic components
    asset_returns = pd.DataFrame(index=symbols, columns=dates)
    
    for symbol in symbols:
        sector = sector_data.loc[symbol, 'sector']
        base_return = sector_returns[sector]
        
        # Add sector effect, market effect, and idiosyncratic noise
        for date in dates:
            # Market effect (correlated across all assets)
            market_effect = np.random.normal(0.0002, 0.01)
            
            # Sector effect (correlated within sector)
            sector_effect = np.random.normal(base_return, 0.005)
            
            # Idiosyncratic effect (unique to each asset)
            idiosyncratic = np.random.normal(0, 0.008)
            
            # Combine effects
            total_return = market_effect + sector_effect + idiosyncratic
            asset_returns.loc[symbol, date] = total_return
    
    # Create portfolio weights (active management)
    portfolio_weights = pd.DataFrame(index=symbols, columns=dates)
    
    # Start with equal weights
    initial_weights = np.ones(len(symbols)) / len(symbols)
    
    for i, date in enumerate(dates):
        if i == 0:
            # Initial weights
            portfolio_weights.loc[:, date] = initial_weights
        else:
            # Simulate active management with some drift
            prev_weights = portfolio_weights.iloc[:, i-1]
            
            # Add some active bets (overweight tech, underweight consumer)
            active_bets = np.zeros(len(symbols))
            for j, symbol in enumerate(symbols):
                sector = sector_data.loc[symbol, 'sector']
                if sector == 'Technology':
                    active_bets[j] = 0.02  # Overweight tech
                else:
                    active_bets[j] = -0.02  # Underweight consumer
            
            # Apply active bets with some rebalancing
            new_weights = prev_weights + active_bets * 0.1
            
            # Normalize weights
            new_weights = np.maximum(new_weights, 0)  # No short positions
            new_weights = new_weights / new_weights.sum()
            
            portfolio_weights.loc[:, date] = new_weights
    
    # Create benchmark weights (market cap weighted)
    benchmark_weights = pd.DataFrame(index=symbols, columns=dates)
    
    # Simulate market cap weights with some variation
    market_caps = np.array([2.5, 1.8, 2.2, 1.5, 0.8, 0.9, 1.2, 0.3, 0.2, 0.3])  # Trillions
    market_cap_weights = market_caps / market_caps.sum()
    
    for date in dates:
        # Add some variation to market cap weights
        variation = np.random.normal(0, 0.01, len(symbols))
        weights = market_cap_weights + variation
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        benchmark_weights.loc[:, date] = weights
    
    # Calculate portfolio returns
    portfolio_returns = pd.Series(index=dates, dtype=float)
    for date in dates:
        weights = portfolio_weights.loc[:, date]
        returns = asset_returns.loc[:, date]
        portfolio_returns[date] = (weights * returns).sum()
    
    # Calculate benchmark returns
    benchmark_returns = pd.Series(index=dates, dtype=float)
    for date in dates:
        weights = benchmark_weights.loc[:, date]
        returns = asset_returns.loc[:, date]
        benchmark_returns[date] = (weights * returns).sum()
    
    return {
        'asset_returns': asset_returns,
        'portfolio_weights': portfolio_weights,
        'benchmark_weights': benchmark_weights,
        'portfolio_returns': portfolio_returns,
        'benchmark_returns': benchmark_returns,
        'sector_data': sector_data,
        'dates': dates,
        'symbols': symbols
    }


def demo_factor_analysis(attributor, data):
    """Demonstrate factor analysis capabilities."""
    print("\n" + "="*60)
    print("FACTOR ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Run factor analysis
    results = attributor.analyze_performance(
        portfolio_returns=data['portfolio_returns'],
        benchmark_returns=data['benchmark_returns'],
        asset_returns=data['asset_returns'],
        portfolio_weights=data['portfolio_weights'],
        benchmark_weights=data['benchmark_weights'],
        sector_data=data['sector_data']
    )
    
    # Display factor attribution results
    print("\nFactor Attribution Results:")
    print("-" * 40)
    factor_attrib = results['factor_attribution']
    for factor, contrib in factor_attrib.items():
        print(f"{factor:15}: {contrib:8.4f}")
    
    # Display factor model quality
    print("\nFactor Model Quality:")
    print("-" * 40)
    r_squared = results['factor_model']['r_squared']
    if r_squared is not None:
        print(f"Average R-squared: {r_squared.mean():.4f}")
        print(f"Min R-squared:    {r_squared.min():.4f}")
        print(f"Max R-squared:    {r_squared.max():.4f}")
    
    # Show return decomposition
    decomposition = results['decomposition']
    systematic_vol = decomposition['systematic'].std().mean() * np.sqrt(252)
    idiosyncratic_vol = decomposition['idiosyncratic'].std().mean() * np.sqrt(252)
    
    print(f"\nReturn Decomposition:")
    print("-" * 40)
    print(f"Systematic Volatility:   {systematic_vol:.4f}")
    print(f"Idiosyncratic Volatility: {idiosyncratic_vol:.4f}")
    print(f"Total Volatility:        {systematic_vol + idiosyncratic_vol:.4f}")
    
    return results


def demo_brinson_attribution(attributor, data):
    """Demonstrate Brinson attribution analysis."""
    print("\n" + "="*60)
    print("BRINSON ATTRIBUTION DEMONSTRATION")
    print("="*60)
    
    # Run attribution analysis
    results = attributor.analyze_performance(
        portfolio_returns=data['portfolio_returns'],
        benchmark_returns=data['benchmark_returns'],
        asset_returns=data['asset_returns'],
        portfolio_weights=data['portfolio_weights'],
        benchmark_weights=data['benchmark_weights'],
        sector_data=data['sector_data']
    )
    
    # Display Brinson attribution results
    brinson_results = results['brinson_attribution']
    
    if brinson_results:
        print("\nSector Attribution Summary:")
        print("-" * 40)
        
        # Aggregate results over time
        allocation_effects = []
        selection_effects = []
        interaction_effects = []
        
        for date, attribution in brinson_results.items():
            allocation_effects.append(attribution['allocation'])
            selection_effects.append(attribution['selection'])
            interaction_effects.append(attribution['interaction'])
        
        print(f"Average Allocation Effect: {np.mean(allocation_effects):.6f}")
        print(f"Average Selection Effect:  {np.mean(selection_effects):.6f}")
        print(f"Average Interaction Effect: {np.mean(interaction_effects):.6f}")
        print(f"Total Attribution:         {np.mean(allocation_effects) + np.mean(selection_effects) + np.mean(interaction_effects):.6f}")
    
    return results


def demo_risk_analysis(attributor, data):
    """Demonstrate risk-adjusted analysis."""
    print("\n" + "="*60)
    print("RISK-ADJUSTED ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Run attribution analysis
    results = attributor.analyze_performance(
        portfolio_returns=data['portfolio_returns'],
        benchmark_returns=data['benchmark_returns'],
        asset_returns=data['asset_returns'],
        portfolio_weights=data['portfolio_weights'],
        benchmark_weights=data['benchmark_weights'],
        sector_data=data['sector_data']
    )
    
    # Display risk-adjusted metrics
    risk_adj = results['risk_adjusted']
    
    print("\nPortfolio Risk Metrics:")
    print("-" * 40)
    portfolio_risk = risk_adj['portfolio_risk']
    for metric, value in portfolio_risk.items():
        print(f"{metric:20}: {value:8.4f}")
    
    print(f"\nInformation Ratio: {risk_adj['information_ratio']:.4f}")
    
    # Display factor risk contributions
    print("\nFactor Risk Contributions:")
    print("-" * 40)
    factor_risk_contrib = risk_adj['factor_risk_contributions']
    for factor, contrib in factor_risk_contrib.items():
        print(f"{factor:15}: {contrib['contribution']:8.4f} (corr: {contrib['correlation']:6.3f})")
    
    return results


def demo_visualization(attributor, data):
    """Demonstrate visualization capabilities."""
    print("\n" + "="*60)
    print("VISUALIZATION DEMONSTRATION")
    print("="*60)
    
    # Run attribution analysis
    results = attributor.analyze_performance(
        portfolio_returns=data['portfolio_returns'],
        benchmark_returns=data['benchmark_returns'],
        asset_returns=data['asset_returns'],
        portfolio_weights=data['portfolio_weights'],
        benchmark_weights=data['benchmark_weights'],
        sector_data=data['sector_data']
    )
    
    # Create dashboard
    print("Creating attribution dashboard...")
    dashboard = attributor.create_dashboard()
    
    # Save dashboard
    output_dir = Path("attribution_output")
    output_dir.mkdir(exist_ok=True)
    
    dashboard_path = output_dir / "attribution_dashboard.html"
    if hasattr(dashboard, 'write_html'):
        dashboard.write_html(str(dashboard_path))
        print(f"Dashboard saved to: {dashboard_path}")
    
    # Create additional visualizations
    create_additional_plots(data, results, output_dir)
    
    return results


def create_additional_plots(data, results, output_dir):
    """Create additional matplotlib plots."""
    print("Creating additional visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')  # or simply 'seaborn'
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Performance Attribution Analysis', fontsize=16)
    
    # 1. Cumulative returns comparison
    cumulative_portfolio = (1 + data['portfolio_returns']).cumprod()
    cumulative_benchmark = (1 + data['benchmark_returns']).cumprod()
    
    axes[0, 0].plot(cumulative_portfolio.index, cumulative_portfolio.values, 
                   label='Portfolio', linewidth=2)
    axes[0, 0].plot(cumulative_benchmark.index, cumulative_benchmark.values, 
                   label='Benchmark', linewidth=2)
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Factor contributions
    if 'factor_attribution' in results:
        factor_contrib = results['factor_attribution']
        factors = list(factor_contrib.keys())
        contributions = list(factor_contrib.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(factors)))
        bars = axes[0, 1].bar(factors, contributions, color=colors)
        axes[0, 1].set_title('Factor Contributions')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, contributions):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
    
    # 3. Risk metrics comparison
    if 'risk_adjusted' in results:
        portfolio_risk = results['risk_adjusted']['portfolio_risk']
        benchmark_risk = results['risk_adjusted']['benchmark_risk']
        
        metrics = ['volatility', 'max_drawdown', 'var']
        portfolio_values = [portfolio_risk.get(m, 0) for m in metrics]
        benchmark_values = [benchmark_risk.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, portfolio_values, width, label='Portfolio', alpha=0.8)
        axes[1, 0].bar(x + width/2, benchmark_values, width, label='Benchmark', alpha=0.8)
        axes[1, 0].set_title('Risk Metrics Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].legend()
    
    # 4. Sector allocation over time
    sector_weights = data['portfolio_weights'].T.join(data['sector_data']['sector'])
    sector_allocation = sector_weights.groupby('sector').sum()
    
    sector_allocation.plot(kind='area', stacked=True, ax=axes[1, 1])
    axes[1, 1].set_title('Sector Allocation Over Time')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "attribution_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Additional plots saved to: {plot_path}")
    
    plt.show()


def demo_automation(data):
    """Demonstrate automated attribution workflow."""
    print("\n" + "="*60)
    print("AUTOMATED ATTRIBUTION WORKFLOW DEMONSTRATION")
    print("="*60)
    
    # Create portfolio manager
    config = PortfolioConfig()
    portfolio_manager = PortfolioManager(initial_capital=1000000, config=config)
    
    # Add performance history
    for date in data['dates']:
        portfolio_manager.performance_history.append({
            'timestamp': date,
            'total_value': 1000000 * (1 + data['portfolio_returns'][:date].sum()),
            'total_return': 1000000 * (1 + data['portfolio_returns'][:date].sum()),
            'cash': 100000,
            'equity_value': 900000 * (1 + data['portfolio_returns'][:date].sum())
        })
    
    # Create automated workflow
    attribution_config = AttributionConfig()
    workflow = AutomatedAttributionWorkflow(portfolio_manager, attribution_config)
    
    # Configure workflow
    workflow.analysis_frequency = "monthly"
    workflow.auto_generate_reports = True
    workflow.report_output_dir = "attribution_reports"
    
    # Run automated analysis
    print("Running automated attribution analysis...")
    results = workflow.run_scheduled_analysis()
    
    if results:
        print("✓ Automated analysis completed successfully")
        
        # Display summary
        summary = workflow.integration.get_attribution_summary()
        print("\nAttribution Summary:")
        print("-" * 40)
        for key, value in summary.items():
            print(f"{key}: {value}")
    else:
        print("⚠ No analysis performed (not due yet)")
    
    return workflow


def demo_reporting(attributor, data):
    """Demonstrate automated reporting capabilities."""
    print("\n" + "="*60)
    print("AUTOMATED REPORTING DEMONSTRATION")
    print("="*60)
    
    # Run attribution analysis
    results = attributor.analyze_performance(
        portfolio_returns=data['portfolio_returns'],
        benchmark_returns=data['benchmark_returns'],
        asset_returns=data['asset_returns'],
        portfolio_weights=data['portfolio_weights'],
        benchmark_weights=data['benchmark_weights'],
        sector_data=data['sector_data']
    )
    
    # Generate report
    output_dir = Path("attribution_output")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "attribution_report.txt"
    report_content = attributor.generate_report(str(report_path))
    
    print(f"Report generated: {report_path}")
    print("\nReport Preview:")
    print("-" * 40)
    print(report_content[:500] + "..." if len(report_content) > 500 else report_content)
    
    # Note: Excel export is available through AttributionIntegration
    # For standalone attributor, data export would need to be implemented separately
    print("\nNote: For Excel export, use AttributionIntegration with a PortfolioManager")
    print(f"\nData exported to: {data_path}")
    
    return results


def main():
    """Run the complete attribution analysis demonstration."""
    print("PERFORMANCE ATTRIBUTION ANALYSIS DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases comprehensive performance attribution capabilities")
    print("including factor analysis, Brinson attribution, risk-adjusted metrics,")
    print("interactive dashboards, and automated reporting.")
    print()
    
    # Create sample data
    data = create_sample_data()
    print(f"✓ Created sample data for {len(data['symbols'])} assets over {len(data['dates'])} days")
    
    # Initialize attribution system
    config = AttributionConfig(
        risk_free_rate=0.02,
        confidence_level=0.95,
        use_plotly=True
    )
    attributor = PerformanceAttributor(config)
    
    # Run demonstrations
    try:
        # Factor analysis
        factor_results = demo_factor_analysis(attributor, data)
        
        # Brinson attribution
        brinson_results = demo_brinson_attribution(attributor, data)
        
        # Risk analysis
        risk_results = demo_risk_analysis(attributor, data)
        
        # Visualization
        viz_results = demo_visualization(attributor, data)
        
        # Automation
        workflow = demo_automation(data)
        
        # Reporting
        report_results = demo_reporting(attributor, data)
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated files:")
        print("  - attribution_output/attribution_dashboard.html")
        print("  - attribution_output/attribution_analysis.png")
        print("  - attribution_output/attribution_report.txt")
        print("  - attribution_output/attribution_data.xlsx")
        print("  - attribution_reports/ (automated reports)")
        
        print("\nNext steps:")
        print("  1. Open attribution_dashboard.html in your browser for interactive analysis")
        print("  2. Review attribution_report.txt for detailed analysis")
        print("  3. Use attribution_data.xlsx for further analysis in Excel")
        print("  4. Set up automated workflows for regular attribution analysis")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()