# Trading System - Kubernetes Deployment Orchestration

This directory contains a comprehensive Kubernetes deployment orchestration for the Trading RL Agent system, implementing a production-ready microservices architecture with horizontal scaling, monitoring, and automated deployment.

## 🏗️ Architecture Overview

The trading system is deployed as a collection of microservices:

- **API Service**: RESTful API gateway for external interactions
- **Trading Engine**: Core trading logic and order execution
- **ML Service**: Machine learning model inference and training (GPU-enabled)
- **Data Pipeline**: Data collection, processing, and storage
- **Infrastructure**: PostgreSQL, Redis, RabbitMQ
- **Monitoring**: Prometheus, Grafana

## 📁 File Structure

```
k8s/
├── namespace.yaml                    # Namespace and resource quotas
├── configmap.yaml                    # Application configuration
├── secrets.yaml                      # Sensitive data (API keys, passwords)
├── persistent-volumes.yaml           # Storage volumes
├── infrastructure-services.yaml      # Database, cache, message queue
├── api-service-deployment.yaml       # API service deployment
├── trading-engine-deployment.yaml    # Trading engine deployment
├── ml-service-deployment.yaml        # ML service deployment (GPU)
├── data-pipeline-deployment.yaml     # Data pipeline deployment
├── monitoring-stack.yaml             # Prometheus and Grafana
├── ingress.yaml                      # External access configuration
├── autoscaling.yaml                  # HPA and VPA configurations
├── ci-cd-pipeline.yaml               # CI/CD pipeline definitions
├── deploy.sh                         # Deployment automation script
├── training-job.yaml                 # Model training jobs
├── download-datasets-job.yaml        # Data download jobs
├── scheduled-backtest-cronjob.yaml   # Automated backtesting
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Docker registry access
- GPU nodes (for ML service)
- Ingress controller (nginx-ingress)
- Cert-manager (for SSL certificates)
- Metrics server (for HPA)

### 1. Prepare Secrets

Create your secrets file with actual values:

```bash
# Encode your secrets
echo -n "your-alpaca-api-key" | base64
echo -n "your-alpaca-secret-key" | base64
echo -n "your-postgres-password" | base64
echo -n "your-redis-password" | base64
echo -n "your-rabbitmq-password" | base64
echo -n "your-jwt-secret" | base64

# Edit secrets.yaml with the encoded values
```

### 2. Deploy the System

```bash
# Make deploy script executable
chmod +x deploy.sh

# Deploy everything
./deploy.sh deploy

# Or deploy step by step
./deploy.sh verify    # Check status
./deploy.sh status    # Show all resources
./deploy.sh logs      # View logs
```

### 3. Access Services

Add to your `/etc/hosts`:
```
127.0.0.1 api.trading-system.local
127.0.0.1 dashboard.trading-system.local
127.0.0.1 monitoring.trading-system.local
```

Access URLs:
- **API**: http://api.trading-system.local
- **Grafana**: http://dashboard.trading-system.local (admin/admin)
- **Prometheus**: http://monitoring.trading-system.local
- **RabbitMQ**: http://rabbitmq.internal:15672

## 🔧 Configuration

### Environment Variables

All services use these environment variables:

```yaml
# Trading API
TRADING_RL_AGENT_ALPACA_API_KEY
TRADING_RL_AGENT_ALPACA_SECRET_KEY
TRADING_RL_AGENT_ALPACA_BASE_URL

# Database
POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB
POSTGRES_USER, POSTGRES_PASSWORD

# Cache
REDIS_HOST, REDIS_PORT, REDIS_PASSWORD

# Message Queue
RABBITMQ_HOST, RABBITMQ_PORT
RABBITMQ_USER, RABBITMQ_PASS

# Security
JWT_SECRET
```

### Resource Requirements

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit | GPU |
|---------|-------------|-----------|----------------|--------------|-----|
| API Service | 500m | 1000m | 1Gi | 2Gi | - |
| Trading Engine | 1000m | 2000m | 2Gi | 4Gi | - |
| ML Service | 2000m | 4000m | 4Gi | 8Gi | 1 |
| Data Pipeline | 1000m | 2000m | 2Gi | 4Gi | - |
| PostgreSQL | 500m | 1000m | 1Gi | 2Gi | - |
| Redis | 250m | 500m | 512Mi | 1Gi | - |
| RabbitMQ | 250m | 500m | 512Mi | 1Gi | - |

### Storage Configuration

| Volume | Size | Access Mode | Purpose |
|--------|------|-------------|---------|
| Data | 100Gi | ReadWriteMany | Market data storage |
| Models | 50Gi | ReadWriteMany | ML models |
| Artifacts | 20Gi | ReadWriteMany | Training artifacts |
| Results | 10Gi | ReadWriteMany | Backtest results |
| Logs | 5Gi | ReadWriteMany | Application logs |
| MLRuns | 10Gi | ReadWriteMany | MLflow tracking |

## 📊 Monitoring & Observability

### Prometheus Metrics

All services expose Prometheus metrics on port 9090:

- **System metrics**: CPU, memory, disk usage
- **Application metrics**: Request rate, latency, error rate
- **Business metrics**: Trading volume, P&L, order success rate

### Grafana Dashboards

Pre-configured dashboards:
- System Overview
- Trading Performance
- ML Model Metrics
- Infrastructure Health

### Alerts

Configured alerts for:
- High resource usage (>80%)
- Service downtime
- High error rates
- Trading engine issues
- ML service latency

## 🔄 Autoscaling

### Horizontal Pod Autoscaler (HPA)

| Service | Min Replicas | Max Replicas | CPU Target | Memory Target |
|---------|--------------|--------------|------------|---------------|
| API Service | 3 | 10 | 70% | 80% |
| Trading Engine | 2 | 5 | 70% | 80% |
| ML Service | 2 | 4 | 70% | 80% |
| Data Pipeline | 2 | 6 | 70% | 80% |

### Vertical Pod Autoscaler (VPA)

ML Service uses VPA for automatic resource optimization:
- CPU: 1000m - 4000m
- Memory: 2Gi - 8Gi

## 🔒 Security

### Network Policies

- Pod-to-pod communication restricted
- External access through ingress only
- Database access limited to application pods

### RBAC

- Service accounts with minimal permissions
- Role-based access control
- Secrets management

### Security Context

- Non-root containers
- Read-only root filesystem where possible
- Dropped capabilities
- Security scanning in CI/CD

## 🚀 CI/CD Pipeline

### GitHub Actions

Automated pipeline with:
- **Testing**: Unit tests, integration tests
- **Security**: Vulnerability scanning
- **Building**: Multi-platform Docker images
- **Deployment**: Staging and production
- **Performance**: Load testing

### ArgoCD Integration

GitOps deployment with:
- Automated sync
- Self-healing
- Rollback capabilities
- Multi-environment support

## 📈 Performance Optimization

### Resource Management

- CPU and memory limits
- GPU allocation for ML workloads
- Storage optimization
- Network policies

### Caching Strategy

- Redis for session data
- Model prediction caching
- Database query caching

### Load Balancing

- Service mesh ready
- Health checks
- Circuit breakers
- Retry policies

## 🛠️ Operations

### Deployment Commands

```bash
# Deploy everything
./deploy.sh deploy

# Check status
./deploy.sh status

# View logs
./deploy.sh logs trading-api-service

# Port forward
./deploy.sh port-forward grafana 3000

# Rollback
./deploy.sh rollback
```

### Monitoring Commands

```bash
# Check pod status
kubectl get pods -n trading-system

# Check services
kubectl get services -n trading-system

# Check HPA
kubectl get hpa -n trading-system

# Check ingress
kubectl get ingress -n trading-system

# View logs
kubectl logs -f deployment/trading-api-service -n trading-system
```

### Troubleshooting

```bash
# Check events
kubectl get events -n trading-system --sort-by='.lastTimestamp'

# Describe resources
kubectl describe pod <pod-name> -n trading-system

# Exec into container
kubectl exec -it <pod-name> -n trading-system -- /bin/bash

# Check resource usage
kubectl top pods -n trading-system
```

## 🔄 Updates and Maintenance

### Rolling Updates

All deployments use rolling update strategy:
- Zero downtime deployments
- Health check validation
- Automatic rollback on failure

### Backup Strategy

- Database backups to persistent storage
- Model versioning with MLflow
- Configuration version control
- Disaster recovery procedures

### Scaling Operations

```bash
# Scale API service
kubectl scale deployment trading-api-service --replicas=5 -n trading-system

# Update image
kubectl set image deployment/trading-api-service api-service=new-image:tag -n trading-system

# Check rollout status
kubectl rollout status deployment/trading-api-service -n trading-system
```

## 📋 Best Practices

### Resource Management

- Set appropriate resource requests and limits
- Use HPA for automatic scaling
- Monitor resource usage
- Optimize container images

### Security

- Use secrets for sensitive data
- Implement network policies
- Regular security updates
- Access control and audit logging

### Monitoring

- Comprehensive metrics collection
- Alerting on critical issues
- Performance monitoring
- Business metrics tracking

### Deployment

- Use rolling updates
- Implement health checks
- Test in staging environment
- Automated rollback procedures

## 🆘 Support

### Common Issues

1. **Pod startup failures**: Check resource limits and health checks
2. **Service connectivity**: Verify network policies and service configuration
3. **Storage issues**: Check PVC status and storage class
4. **Scaling problems**: Verify HPA configuration and metrics server

### Getting Help

- Check logs: `kubectl logs -n trading-system`
- Monitor events: `kubectl get events -n trading-system`
- Verify configuration: `kubectl describe -n trading-system`
- Check metrics: Access Grafana dashboard

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/)
- [Helm Charts](https://helm.sh/docs/)
- [ArgoCD GitOps](https://argoproj.github.io/argo-cd/)

---

**Note**: This deployment is production-ready but should be customized for your specific environment, security requirements, and performance needs.
