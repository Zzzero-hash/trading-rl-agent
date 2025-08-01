#!/usr/bin/env python3
"""
Trade Agent MCP Server - Custom MCP server for trading-specific services
"""
import asyncio
import datetime
import os
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

# Create MCP server instance
server = Server("trade-agent-mcp")

@server.list_tools()
async def list_tools():
    """List available trade-specific tools"""
    return [
        Tool(
            name="memory_store",
            description="Store trading insights, strategies, or analysis for later retrieval",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Key to store data under (e.g., 'strategy_momentum', 'analysis_AAPL')"},
                    "value": {"type": "string", "description": "Value to store"},
                    "category": {"type": "string", "description": "Category: strategy, analysis, backtest, or insight", "enum": ["strategy", "analysis", "backtest", "insight"]}
                },
                "required": ["key", "value"]
            }
        ),
        Tool(
            name="memory_retrieve",
            description="Retrieve stored trading information",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Key to retrieve data for"},
                    "category": {"type": "string", "description": "Filter by category: strategy, analysis, backtest, or insight", "enum": ["strategy", "analysis", "backtest", "insight"]}
                },
                "required": ["key"]
            }
        ),
        Tool(
            name="list_memories",
            description="List all stored trading memories by category",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Filter by category: strategy, analysis, backtest, or insight", "enum": ["strategy", "analysis", "backtest", "insight"]}
                }
            }
        ),
        Tool(
            name="trading_log",
            description="Log trading decisions, outcomes, or observations",
            inputSchema={
                "type": "object",
                "properties": {
                    "entry": {"type": "string", "description": "Log entry describing the trading event"},
                    "symbol": {"type": "string", "description": "Trading symbol (optional)"},
                    "action": {"type": "string", "description": "Type of action", "enum": ["buy", "sell", "hold", "analysis", "backtest", "observation"]},
                    "confidence": {"type": "number", "description": "Confidence level 0-100", "minimum": 0, "maximum": 100}
                },
                "required": ["entry", "action"]
            }
        ),
        Tool(
            name="get_trading_logs",
            description="Retrieve trading logs with optional filtering",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Filter by trading symbol"},
                    "action": {"type": "string", "description": "Filter by action type", "enum": ["buy", "sell", "hold", "analysis", "backtest", "observation"]},
                    "days": {"type": "number", "description": "Number of days to look back (default: 7)", "minimum": 1}
                }
            }
        ),
        Tool(
            name="market_time",
            description="Get current market time and status",
            inputSchema={
                "type": "object",
                "properties": {
                    "market": {"type": "string", "description": "Market timezone", "enum": ["NYSE", "NASDAQ", "LSE", "TSE", "UTC"], "default": "NYSE"}
                }
            }
        )
    ]

# Enhanced storage with categories
memory_store: dict[str, dict[str, Any]] = {}
trading_logs: list[dict[str, Any]] = []

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]):
    """Handle tool calls for trading operations"""

    if name == "memory_store":
        store_key: str = arguments["key"]
        store_value: str = arguments["value"]
        store_category: str = arguments.get("category", "insight")

        memory_store[store_key] = {
            "value": store_value,
            "category": store_category,
            "timestamp": datetime.datetime.now().isoformat()
        }
        return [TextContent(type="text", text=f"Stored '{store_value}' under key '{store_key}' in category '{store_category}'")]

    elif name == "memory_retrieve":
        retrieve_key: str = arguments["key"]
        retrieve_category_filter: str | None = arguments.get("category")

        if retrieve_key in memory_store:
            retrieve_entry = memory_store[retrieve_key]
            if retrieve_category_filter and retrieve_entry["category"] != retrieve_category_filter:
                return [TextContent(type="text", text=f"Key '{retrieve_key}' found but not in category '{retrieve_category_filter}'")]

            return [TextContent(type="text", text=f"Retrieved [{retrieve_entry['category']}]: {retrieve_entry['value']} (stored: {retrieve_entry['timestamp']})")]
        else:
            return [TextContent(type="text", text=f"Key '{retrieve_key}' not found")]

    elif name == "list_memories":
        list_category_filter: str | None = arguments.get("category")

        if list_category_filter:
            filtered = {k: v for k, v in memory_store.items() if v["category"] == list_category_filter}
            keys = list(filtered.keys())
        else:
            keys = list(memory_store.keys())

        if keys:
            return [TextContent(type="text", text=f"Stored memories: {', '.join(keys)}")]
        else:
            filter_text = f" in category '{list_category_filter}'" if list_category_filter else ""
            return [TextContent(type="text", text=f"No memories stored{filter_text}")]

    elif name == "trading_log":
        log_entry_text: str = arguments["entry"]
        log_symbol: str = arguments.get("symbol", "")
        log_action: str = arguments["action"]
        log_confidence: float = arguments.get("confidence", 0)

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "entry": log_entry_text,
            "symbol": log_symbol,
            "action": log_action,
            "confidence": log_confidence
        }

        trading_logs.append(log_entry)

        return [TextContent(type="text", text=f"Logged {log_action} for {log_symbol}: {log_entry_text} (confidence: {log_confidence}%)")]

    elif name == "get_trading_logs":
        logs_symbol_filter: str | None = arguments.get("symbol")
        logs_action_filter: str | None = arguments.get("action")
        logs_days: int = arguments.get("days", 7)

        # Filter by date
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=logs_days)
        recent_logs = [
            log for log in trading_logs
            if datetime.datetime.fromisoformat(log["timestamp"]) > cutoff_date
        ]

        # Apply filters
        if logs_symbol_filter:
            recent_logs = [log for log in recent_logs if log["symbol"] == logs_symbol_filter]
        if logs_action_filter:
            recent_logs = [log for log in recent_logs if log["action"] == logs_action_filter]

        if recent_logs:
            log_text = "\n".join([
                f"[{log['timestamp']}] {log['action'].upper()} {log['symbol']}: {log['entry']} (confidence: {log['confidence']}%)"
                for log in recent_logs[-10:]  # Last 10 entries
            ])
            return [TextContent(type="text", text=f"Recent trading logs:\n{log_text}")]
        else:
            return [TextContent(type="text", text="No trading logs found matching criteria")]

    elif name == "market_time":
        market: str = arguments.get("market", "NYSE") or "NYSE"

        market_timezones = {
            "NYSE": "America/New_York",
            "NASDAQ": "America/New_York",
            "LSE": "Europe/London",
            "TSE": "Asia/Tokyo",
            "UTC": "UTC"
        }

        tz_name = market_timezones.get(market, "UTC")
        now = datetime.datetime.now()

        # Simple market hours check (NYSE/NASDAQ: 9:30 AM - 4:00 PM ET, Mon-Fri)
        market_status = "Unknown"
        if market in ["NYSE", "NASDAQ"]:
            weekday = now.weekday()  # 0 = Monday, 6 = Sunday
            hour = now.hour
            if weekday < 5:  # Monday to Friday
                if 9 <= hour < 16:  # Simplified: 9 AM to 4 PM
                    market_status = "Open"
                else:
                    market_status = "Closed"
            else:
                market_status = "Closed (Weekend)"

        return [TextContent(type="text", text=f"{market} time: {now} ({tz_name}) - Status: {market_status}")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Run the Trade Agent MCP server"""
    from mcp.server.stdio import stdio_server

    # Create data directory if it doesn't exist
    os.makedirs("/workspaces/trade-agent/data", exist_ok=True)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
