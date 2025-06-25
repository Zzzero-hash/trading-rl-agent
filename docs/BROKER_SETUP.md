# Trading RL Agent - Message Broker Setup

This directory contains Docker configurations for the message broker infrastructure used in the trading system.

## Quick Start

```bash
# Start the broker services
docker-compose -f docker-compose.broker.yml up -d

# Check status
docker-compose -f docker-compose.broker.yml ps

# View logs
docker-compose -f docker-compose.broker.yml logs -f nats

# Stop services
docker-compose -f docker-compose.broker.yml down
```

## Services

### NATS (Primary Message Broker)

- **Port**: 4222 (client connections)
- **Monitoring**: http://localhost:8222
- **Features**: JetStream enabled for persistence
- **Storage**: 1GB file store, 256MB memory

### Redis (Caching & Sessions)

- **Port**: 6379
- **Features**: Persistence enabled, LRU eviction
- **Memory Limit**: 256MB

### NATS Surveyor (Monitoring Dashboard)

- **Port**: http://localhost:7777
- **Purpose**: Real-time NATS metrics and monitoring

## Configuration

### Environment Variables

```bash
# NATS connection
NATS_URL=nats://localhost:4222

# Redis connection
REDIS_URL=redis://localhost:6379

# For containerized apps
NATS_URL=nats://trading-nats:4222
REDIS_URL=redis://trading-redis:6379
```

### Network Configuration

- **Network**: `trading_network` (172.20.0.0/16)
- **Internal DNS**: Services accessible by container name

## Development Integration

### Python Client Example

```python
import asyncio
import nats

async def connect_nats():
    nc = await nats.connect("nats://localhost:4222")

    # Subscribe to market data
    async def market_handler(msg):
        data = json.loads(msg.data.decode())
        print(f"Received: {data}")

    await nc.subscribe("market.data.*", cb=market_handler)

    # Publish test data
    await nc.publish("market.data.AAPL",
                    json.dumps({"price": 150.0, "volume": 1000}).encode())

    return nc
```

### Health Checks

```bash
# Check NATS health
curl http://localhost:8222/healthz

# Check Redis
redis-cli ping
```

## Kubernetes Migration Path

This Docker setup translates directly to Kubernetes:

1. **Services** → K8s Services
2. **Volumes** → PersistentVolumeClaims
3. **Networks** → K8s NetworkPolicies
4. **Health checks** → K8s Probes

### Helm Chart Structure (Future)

```
charts/trading-broker/
├── templates/
│   ├── nats-deployment.yaml
│   ├── redis-deployment.yaml
│   ├── services.yaml
│   └── pvcs.yaml
├── values.yaml
└── Chart.yaml
```

## Monitoring & Observability

### NATS Metrics

- Connection count: `curl http://localhost:8222/connz`
- Subject stats: `curl http://localhost:8222/subsz`
- Server info: `curl http://localhost:8222/varz`

### Resource Usage

```bash
# Container resource usage
docker stats trading-nats trading-redis

# Disk usage
docker system df
```

## Security Notes

- **Development**: No authentication required
- **Production**: Enable NATS auth, Redis password, TLS
- **Network**: Isolated docker network for security

## Troubleshooting

### Common Issues

1. **Port conflicts**

   ```bash
   # Check port usage
   netstat -tulpn | grep :4222
   ```

2. **Storage issues**

   ```bash
   # Clean up volumes
   docker-compose -f docker-compose.broker.yml down -v
   ```

3. **Connection refused**
   ```bash
   # Check service health
   docker-compose -f docker-compose.broker.yml logs nats
   ```

## Cost Analysis

### Resource Requirements

- **NATS**: ~50MB RAM, minimal CPU
- **Redis**: ~100MB RAM, minimal CPU
- **Total**: <200MB RAM, negligible storage

### Cloud Costs (estimated)

- **Local dev**: $0 (uses dev machine)
- **Cloud VM**: ~$5-10/month for t3.micro/small
- **Managed K8s**: ~$15-25/month including cluster costs

This is significantly cheaper than managed message brokers like AWS MSK ($100+/month).
