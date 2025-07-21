# ArgoCD GitOps Setup for Trading RL Agent

This directory contains the ArgoCD configuration for implementing GitOps deployment for the Trading RL Agent system. ArgoCD automatically syncs your Kubernetes cluster with the desired state defined in your Git repository.

## 🚀 Quick Start

### 1. Install ArgoCD

```bash
# Run the setup script
./k8s/argocd/argocd-setup.sh
```

### 2. Access ArgoCD UI

- **URL**: https://argocd.trading-system.local
- **Username**: admin
- **Password**: Check `argocd-admin-password.txt` file

### 3. Deploy Trading System

```bash
# Deploy to production
./deploy-trading-system.sh production latest

# Deploy to staging
./deploy-trading-system.sh staging v1.2.3
```

## 📁 File Structure

```
k8s/argocd/
├── argocd-installation.yaml    # ArgoCD server installation
├── trading-system-app.yaml     # Main trading system application
├── application-set.yaml        # ApplicationSet for multi-env deployment
├── notifications.yaml          # Notification configuration
├── argocd-setup.sh            # Installation script
└── README.md                  # This file
```

## 🔧 Configuration

### Repository Configuration

Update the following files with your actual GitHub repository URL:

- `trading-system-app.yaml`
- `application-set.yaml`
- `notifications.yaml`

Replace `https://github.com/yourusername/trading-rl-agent.git` with your actual repository URL.

### Environment Configuration

The system supports multiple environments:

- **Production**: `trading-system` namespace
- **Staging**: `trading-system-staging` namespace

Each environment has different resource allocations and replica counts.

### Notification Setup

Configure Slack notifications by updating the secrets in `notifications.yaml`:

```yaml
stringData:
  slack-token: "xoxb-your-slack-bot-token"
  webhook-url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
```

## 🎯 How It Works

### GitOps Workflow

1. **Code Changes**: Developers push changes to Git repository
2. **Image Build**: CI/CD pipeline builds and pushes Docker images
3. **Manifest Updates**: Update Kubernetes manifests with new image tags
4. **Git Push**: Push manifest changes to Git repository
5. **ArgoCD Sync**: ArgoCD automatically detects changes and syncs cluster
6. **Deployment**: New version is deployed to Kubernetes cluster

### Application Structure

```
Trading System Application
├── Infrastructure
│   ├── Namespace
│   ├── Persistent Volumes
│   └── Infrastructure Services (Redis, PostgreSQL, RabbitMQ)
├── Core Services
│   ├── ConfigMaps & Secrets
│   ├── Data Pipeline
│   ├── ML Service
│   ├── Trading Engine
│   └── API Service
├── Networking
│   ├── Ingress
│   └── Autoscaling
├── Monitoring
│   ├── Prometheus Stack
│   └── Alerting Rules
└── Jobs
    ├── Training Jobs
    ├── Backtest Jobs
    └── Data Download Jobs
```

## 🔄 Deployment Process

### Manual Deployment

```bash
# Deploy specific version
./deploy-trading-system.sh production v2.1.0

# Deploy latest
./deploy-trading-system.sh production latest
```

### Automated Deployment

ArgoCD automatically syncs when:

- New commits are pushed to the main branch
- Image tags are updated in deployment manifests
- Configuration changes are made

### Rollback Process

```bash
# Rollback to previous version
argocd app rollback trading-system

# Rollback to specific revision
argocd app rollback trading-system 2
```

## 📊 Monitoring & Alerts

### ArgoCD Notifications

The system sends notifications for:

- ✅ Successful deployments
- ❌ Failed deployments
- ⚠️ Health degradation
- 🔄 Sync in progress

### Health Checks

ArgoCD monitors:

- Pod health status
- Service availability
- Resource quotas
- Network connectivity

## 🔐 Security

### RBAC Configuration

- **Admin**: Full access to all ArgoCD resources
- **Trading Team**: Limited access to trading system applications
- **Read-only**: View-only access for monitoring

### Secrets Management

- Secrets are stored in Kubernetes secrets
- ArgoCD can read secrets for deployment
- No secrets stored in Git repository

## 🛠️ Troubleshooting

### Common Issues

#### 1. ArgoCD Not Syncing

```bash
# Check application status
argocd app get trading-system

# Force sync
argocd app sync trading-system --force

# Check logs
kubectl logs -n argocd deployment/argocd-application-controller
```

#### 2. Image Pull Errors

```bash
# Check image pull secrets
kubectl get secrets -n trading-system

# Verify image exists
docker pull ghcr.io/yourusername/trading-rl-agent:latest
```

#### 3. Resource Quota Exceeded

```bash
# Check resource usage
kubectl describe resourcequota trading-system-quota -n trading-system

# Scale down if needed
kubectl scale deployment trading-engine --replicas=1 -n trading-system
```

### Debug Commands

```bash
# Get application details
argocd app get trading-system -o yaml

# Check sync status
argocd app sync-status trading-system

# View application logs
argocd app logs trading-system

# Check cluster connectivity
argocd cluster list
```

## 📈 Best Practices

### 1. Git Workflow

- Use feature branches for development
- Merge to main only after testing
- Tag releases for production deployments
- Keep manifests in sync with code

### 2. Resource Management

- Set appropriate resource limits
- Use horizontal pod autoscaling
- Monitor resource usage
- Clean up unused resources

### 3. Security

- Rotate secrets regularly
- Use least privilege access
- Monitor for security vulnerabilities
- Keep ArgoCD updated

### 4. Monitoring

- Set up comprehensive alerting
- Monitor application health
- Track deployment metrics
- Log all operations

## 🔄 Migration from Manual Deployments

### Before Migration

1. Backup current deployment:

```bash
kubectl get all -n trading-system -o yaml > backup.yaml
```

2. Verify current state:

```bash
kubectl get pods,services,deployments -n trading-system
```

### After Migration

1. Install ArgoCD:

```bash
./k8s/argocd/argocd-setup.sh
```

2. Deploy applications:

```bash
kubectl apply -f k8s/argocd/
```

3. Verify deployment:

```bash
argocd app list
argocd app get trading-system
```

## 📞 Support

For issues with ArgoCD setup:

1. Check ArgoCD logs: `kubectl logs -n argocd deployment/argocd-server`
2. Verify cluster connectivity: `argocd cluster list`
3. Check application status: `argocd app get trading-system`

For trading system issues:

1. Check application logs: `kubectl logs -n trading-system deployment/trading-engine`
2. Verify service health: `kubectl get endpoints -n trading-system`
3. Check resource usage: `kubectl top pods -n trading-system`
