"""
Real-time streaming dashboard with WebSocket support.

This module provides real-time data streaming capabilities for:
- Live P&L updates via WebSocket
- Real-time position tracking
- Live market data integration
- Streaming performance metrics
- WebSocket-based dashboard updates
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

from .performance_dashboard import PerformanceDashboard


class StreamingDashboard:
    """Real-time streaming dashboard with WebSocket support."""

    def __init__(
        self,
        performance_dashboard: PerformanceDashboard,
        host: str = "localhost",
        port: int = 8765,
        update_interval: float = 0.1,
    ) -> None:
        """Initialize the streaming dashboard.

        Args:
            performance_dashboard: PerformanceDashboard instance
            host: WebSocket server host
            port: WebSocket server port
            update_interval: Update interval in seconds
        """
        self.performance_dashboard = performance_dashboard
        self.host = host
        self.port = port
        self.update_interval = update_interval
        
        # WebSocket connections
        self.connections: Set[WebSocketServerProtocol] = set()
        
        # Data streams
        self.data_streams: Dict[str, List[Dict[str, Any]]] = {
            'pnl': [],
            'positions': [],
            'risk_metrics': [],
            'performance_metrics': [],
            'alerts': [],
        }
        
        # Callbacks for data updates
        self.update_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Streaming state
        self.is_streaming = False
        self.stream_task: Optional[asyncio.Task] = None

    async def start_server(self) -> None:
        """Start the WebSocket server."""
        server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port
        )
        
        print(f"🚀 Streaming dashboard server started on ws://{self.host}:{self.port}")
        
        # Start data streaming
        self.is_streaming = True
        self.stream_task = asyncio.create_task(self._stream_data())
        
        await server.wait_closed()

    async def stop_server(self) -> None:
        """Stop the WebSocket server."""
        self.is_streaming = False
        
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for connection in self.connections.copy():
            await connection.close()
        
        print("🛑 Streaming dashboard server stopped")

    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle WebSocket connection.

        Args:
            websocket: WebSocket connection
            path: Request path
        """
        self.connections.add(websocket)
        print(f"📡 New client connected. Total connections: {len(self.connections)}")
        
        try:
            # Send initial data
            await self._send_initial_data(websocket)
            
            # Handle incoming messages
            async for message in websocket:
                await self._handle_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connections.discard(websocket)
            print(f"📡 Client disconnected. Total connections: {len(self.connections)}")

    async def _send_initial_data(self, websocket: WebSocketServerProtocol) -> None:
        """Send initial data to new client.

        Args:
            websocket: WebSocket connection
        """
        initial_data = {
            'type': 'initial_data',
            'timestamp': time.time(),
            'data': {
                'pnl': self.data_streams['pnl'][-10:] if self.data_streams['pnl'] else [],
                'positions': self.data_streams['positions'][-10:] if self.data_streams['positions'] else [],
                'risk_metrics': self.data_streams['risk_metrics'][-10:] if self.data_streams['risk_metrics'] else [],
                'performance_metrics': self.data_streams['performance_metrics'][-10:] if self.data_streams['performance_metrics'] else [],
                'alerts': self.data_streams['alerts'][-10:] if self.data_streams['alerts'] else [],
            }
        }
        
        await websocket.send(json.dumps(initial_data))

    async def _handle_message(self, websocket: WebSocketServerProtocol, message: str) -> None:
        """Handle incoming WebSocket message.

        Args:
            websocket: WebSocket connection
            message: Received message
        """
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                await self._handle_subscribe(websocket, data)
            elif message_type == 'unsubscribe':
                await self._handle_unsubscribe(websocket, data)
            elif message_type == 'request_data':
                await self._handle_request_data(websocket, data)
            else:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                }))
                
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Invalid JSON format'
            }))

    async def _handle_subscribe(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """Handle subscription request.

        Args:
            websocket: WebSocket connection
            data: Subscription data
        """
        stream_type = data.get('stream_type', 'all')
        
        response = {
            'type': 'subscription_confirmed',
            'stream_type': stream_type,
            'timestamp': time.time()
        }
        
        await websocket.send(json.dumps(response))
        print(f"📡 Client subscribed to {stream_type} stream")

    async def _handle_unsubscribe(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """Handle unsubscription request.

        Args:
            websocket: WebSocket connection
            data: Unsubscription data
        """
        stream_type = data.get('stream_type', 'all')
        
        response = {
            'type': 'unsubscription_confirmed',
            'stream_type': stream_type,
            'timestamp': time.time()
        }
        
        await websocket.send(json.dumps(response))
        print(f"📡 Client unsubscribed from {stream_type} stream")

    async def _handle_request_data(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """Handle data request.

        Args:
            websocket: WebSocket connection
            data: Request data
        """
        data_type = data.get('data_type', 'all')
        limit = data.get('limit', 100)
        
        response_data = {}
        
        if data_type in ['all', 'pnl']:
            response_data['pnl'] = self.data_streams['pnl'][-limit:]
        
        if data_type in ['all', 'positions']:
            response_data['positions'] = self.data_streams['positions'][-limit:]
        
        if data_type in ['all', 'risk_metrics']:
            response_data['risk_metrics'] = self.data_streams['risk_metrics'][-limit:]
        
        if data_type in ['all', 'performance_metrics']:
            response_data['performance_metrics'] = self.data_streams['performance_metrics'][-limit:]
        
        if data_type in ['all', 'alerts']:
            response_data['alerts'] = self.data_streams['alerts'][-limit:]
        
        response = {
            'type': 'data_response',
            'data_type': data_type,
            'timestamp': time.time(),
            'data': response_data
        }
        
        await websocket.send(json.dumps(response))

    async def _stream_data(self) -> None:
        """Stream data to all connected clients."""
        while self.is_streaming:
            try:
                # Update dashboard data
                self.performance_dashboard._update_data()
                
                # Get latest data
                latest_data = self._get_latest_data()
                
                # Store in data streams
                self._update_data_streams(latest_data)
                
                # Broadcast to all connections
                await self._broadcast_data(latest_data)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                print(f"❌ Error in data streaming: {e}")
                await asyncio.sleep(1.0)

    def _get_latest_data(self) -> Dict[str, Any]:
        """Get latest data from performance dashboard.

        Returns:
            Dictionary with latest data
        """
        trading_metrics = self.performance_dashboard.dashboard.get_trading_metrics()
        risk_metrics = self.performance_dashboard.dashboard.get_risk_metrics()
        health_metrics = self.performance_dashboard.dashboard.get_system_health()
        alerts = self.performance_dashboard.dashboard.get_recent_alerts(limit=5)
        
        return {
            'timestamp': time.time(),
            'pnl': {
                'total_pnl': trading_metrics['pnl'],
                'daily_pnl': trading_metrics['daily_pnl'],
                'cumulative_return': trading_metrics['total_return'],
                'sharpe_ratio': trading_metrics['sharpe_ratio'],
                'max_drawdown': trading_metrics['max_drawdown'],
            },
            'positions': st.session_state.position_data if hasattr(st, 'session_state') else [],
            'risk_metrics': {
                'var_95': risk_metrics['var_95'],
                'cvar_95': risk_metrics['cvar_95'],
                'volatility': risk_metrics['volatility'],
                'beta': risk_metrics['beta'],
                'current_exposure': risk_metrics['current_exposure'],
                'position_concentration': risk_metrics['position_concentration'],
            },
            'performance_metrics': {
                'win_rate': trading_metrics['win_rate'],
                'total_trades': trading_metrics['total_trades'],
                'open_positions': trading_metrics['open_positions'],
            },
            'system_health': {
                'cpu_usage': health_metrics['cpu_usage'],
                'memory_usage': health_metrics['memory_usage'],
                'disk_usage': health_metrics['disk_usage'],
                'network_latency': health_metrics['network_latency'],
                'error_rate': health_metrics['error_rate'],
                'response_time': health_metrics['response_time'],
            },
            'alerts': alerts,
        }

    def _update_data_streams(self, data: Dict[str, Any]) -> None:
        """Update data streams with latest data.

        Args:
            data: Latest data dictionary
        """
        timestamp = data['timestamp']
        
        # Update P&L stream
        self.data_streams['pnl'].append({
            'timestamp': timestamp,
            'data': data['pnl']
        })
        
        # Update positions stream
        self.data_streams['positions'].append({
            'timestamp': timestamp,
            'data': data['positions']
        })
        
        # Update risk metrics stream
        self.data_streams['risk_metrics'].append({
            'timestamp': timestamp,
            'data': data['risk_metrics']
        })
        
        # Update performance metrics stream
        self.data_streams['performance_metrics'].append({
            'timestamp': timestamp,
            'data': data['performance_metrics']
        })
        
        # Update alerts stream
        self.data_streams['alerts'].append({
            'timestamp': timestamp,
            'data': data['alerts']
        })
        
        # Limit stream size
        max_stream_size = 1000
        for stream_name in self.data_streams:
            if len(self.data_streams[stream_name]) > max_stream_size:
                self.data_streams[stream_name] = self.data_streams[stream_name][-max_stream_size:]

    async def _broadcast_data(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected clients.

        Args:
            data: Data to broadcast
        """
        if not self.connections:
            return
        
        message = {
            'type': 'data_update',
            'timestamp': data['timestamp'],
            'data': data
        }
        
        message_json = json.dumps(message)
        
        # Broadcast to all connections
        disconnected = set()
        for connection in self.connections:
            try:
                await connection.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(connection)
            except Exception as e:
                print(f"❌ Error sending to client: {e}")
                disconnected.add(connection)
        
        # Remove disconnected clients
        self.connections -= disconnected

    def add_update_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for data updates.

        Args:
            callback: Callback function to call on data updates
        """
        self.update_callbacks.append(callback)

    def remove_update_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove callback for data updates.

        Args:
            callback: Callback function to remove
        """
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)

    def get_connection_count(self) -> int:
        """Get number of active connections.

        Returns:
            Number of active WebSocket connections
        """
        return len(self.connections)

    def get_stream_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics.

        Returns:
            Dictionary with streaming statistics
        """
        return {
            'active_connections': len(self.connections),
            'is_streaming': self.is_streaming,
            'update_interval': self.update_interval,
            'stream_sizes': {
                stream_name: len(stream_data)
                for stream_name, stream_data in self.data_streams.items()
            },
            'last_update': time.time(),
        }


class WebSocketClient:
    """WebSocket client for connecting to streaming dashboard."""

    def __init__(self, uri: str = "ws://localhost:8765"):
        """Initialize WebSocket client.

        Args:
            uri: WebSocket server URI
        """
        self.uri = uri
        self.websocket: Optional[WebSocketServerProtocol] = None
        self.is_connected = False
        self.message_handlers: Dict[str, Callable[[Dict[str, Any]], None]] = {}

    async def connect(self) -> None:
        """Connect to WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.is_connected = True
            print(f"🔗 Connected to streaming dashboard at {self.uri}")
        except Exception as e:
            print(f"❌ Failed to connect to streaming dashboard: {e}")
            self.is_connected = False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            print("🔗 Disconnected from streaming dashboard")

    async def subscribe(self, stream_type: str = "all") -> None:
        """Subscribe to data stream.

        Args:
            stream_type: Type of stream to subscribe to
        """
        if not self.is_connected or not self.websocket:
            return
        
        message = {
            'type': 'subscribe',
            'stream_type': stream_type,
            'timestamp': time.time()
        }
        
        await self.websocket.send(json.dumps(message))

    async def unsubscribe(self, stream_type: str = "all") -> None:
        """Unsubscribe from data stream.

        Args:
            stream_type: Type of stream to unsubscribe from
        """
        if not self.is_connected or not self.websocket:
            return
        
        message = {
            'type': 'unsubscribe',
            'stream_type': stream_type,
            'timestamp': time.time()
        }
        
        await self.websocket.send(json.dumps(message))

    async def request_data(self, data_type: str = "all", limit: int = 100) -> None:
        """Request specific data.

        Args:
            data_type: Type of data to request
            limit: Maximum number of data points
        """
        if not self.is_connected or not self.websocket:
            return
        
        message = {
            'type': 'request_data',
            'data_type': data_type,
            'limit': limit,
            'timestamp': time.time()
        }
        
        await self.websocket.send(json.dumps(message))

    async def listen(self) -> None:
        """Listen for incoming messages."""
        if not self.is_connected or not self.websocket:
            return
        
        try:
            async for message in self.websocket:
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            self.is_connected = False
            print("🔗 Connection closed by server")

    async def _handle_message(self, message: str) -> None:
        """Handle incoming message.

        Args:
            message: Received message
        """
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            # Call registered handler
            if message_type in self.message_handlers:
                self.message_handlers[message_type](data)
            else:
                print(f"📨 Received message: {message_type}")
                
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON message: {message}")

    def add_message_handler(self, message_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Add message handler.

        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        self.message_handlers[message_type] = handler

    def remove_message_handler(self, message_type: str) -> None:
        """Remove message handler.

        Args:
            message_type: Type of message to remove handler for
        """
        if message_type in self.message_handlers:
            del self.message_handlers[message_type]


async def run_streaming_dashboard(
    performance_dashboard: PerformanceDashboard,
    host: str = "localhost",
    port: int = 8765
) -> None:
    """Run the streaming dashboard.

    Args:
        performance_dashboard: PerformanceDashboard instance
        host: WebSocket server host
        port: WebSocket server port
    """
    streaming_dashboard = StreamingDashboard(
        performance_dashboard=performance_dashboard,
        host=host,
        port=port
    )
    
    try:
        await streaming_dashboard.start_server()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down streaming dashboard...")
        await streaming_dashboard.stop_server()


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    # Create performance dashboard
    from .performance_dashboard import PerformanceDashboard
    from .metrics_collector import MetricsCollector
    from .dashboard import Dashboard
    
    metrics_collector = MetricsCollector()
    dashboard = Dashboard(metrics_collector)
    performance_dashboard = PerformanceDashboard(
        metrics_collector=metrics_collector,
        dashboard=dashboard
    )
    
    # Run streaming dashboard
    asyncio.run(run_streaming_dashboard(performance_dashboard))