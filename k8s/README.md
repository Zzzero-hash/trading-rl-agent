# Trading RL Agent - Kubernetes Job Templates

This directory contains Kubernetes YAML templates for running Trading RL Agent CLI commands as Jobs and CronJobs. These templates are **not production-ready** and should be customized for your specific environment.

## 📁 Files Overview

### Job Templates

- `download-datasets-job.yaml` - Downloads nightly datasets
- `scheduled-backtest-job.yaml` - Runs backtesting (one-time)
- `training-job.yaml` - Runs ad-hoc training with GPU support

### CronJob Templates

- `scheduled-backtest-cronjob.yaml` - Scheduled daily backtesting

### Supporting Resources

- `secrets.yaml` - API keys and sensitive data
- `configmap.yaml` - Non-sensitive configuration
- `persistent-volumes.yaml` - Storage volumes for data, models, artifacts, etc.

## 🚀 Quick Start

### 1. Prerequisites

- Kubernetes cluster with GPU support (for training jobs)
- Docker image `trading-rl-agent:latest` available in your registry
- Storage class configured for PersistentVolumeClaims

### 2. Setup Secrets

First, create your secrets with actual API keys:

```bash
# Encode your API keys
echo -n "your-alpaca-api-key" | base64
echo -n "your-alpaca-secret-key" | base64
echo -n "https://paper-api.alpaca.markets" | base64

# Edit secrets.yaml with the encoded values
kubectl apply -f k8s/secrets.yaml
```

### 3. Deploy Supporting Resources

```bash
# Create ConfigMap and PersistentVolumeClaims
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/persistent-volumes.yaml
```

### 4. Run Jobs

#### Download Datasets

```bash
kubectl apply -f k8s/download-datasets-job.yaml
```

#### Run Training

```bash
kubectl apply -f k8s/training-job.yaml
```

#### Run Backtesting

```bash
kubectl apply -f k8s/scheduled-backtest-job.yaml
```

#### Setup Scheduled Backtesting

```bash
kubectl apply -f k8s/scheduled-backtest-cronjob.yaml
```

## 📊 Job Types

### 1. Download Datasets Job

- **Purpose**: Downloads nightly market data
- **Command**: `trading-rl-agent data all`
- **Resources**: 2-4Gi memory, 0.5-1 CPU
- **Timeout**: 1 hour
- **Volumes**: Config (read-only), Data (read-write), Logs (read-write)

### 2. Training Job

- **Purpose**: Ad-hoc model training with GPU support
- **Command**: `trading-rl-agent train cnn-lstm`
- **Resources**: 8-16Gi memory, 2-4 CPU, 1 GPU
- **Timeout**: 4 hours
- **Volumes**: Config (read-only), Data (read-only), Models (read-write), Artifacts (read-write), Logs (read-write), MLRuns (read-write)

### 3. Backtesting Job

- **Purpose**: Strategy backtesting
- **Command**: `trading-rl-agent backtest strategy`
- **Resources**: 4-8Gi memory, 1-2 CPU
- **Timeout**: 2 hours
- **Volumes**: Config (read-only), Data (read-only), Models (read-only), Results (read-write), Logs (read-write)

### 4. Scheduled Backtesting CronJob

- **Purpose**: Daily automated backtesting
- **Schedule**: Daily at 2 AM UTC
- **Concurrency**: Forbid (prevents overlapping runs)
- **History**: 7 successful, 3 failed jobs

## 🔧 Customization

### Environment Variables

All jobs use these environment variables from secrets:

- `TRADING_RL_AGENT_ALPACA_API_KEY`
- `TRADING_RL_AGENT_ALPACA_SECRET_KEY`
- `TRADING_RL_AGENT_ALPACA_BASE_URL`
- `TRADING_RL_AGENT_ALPHAVANTAGE_API_KEY` (optional)
- `TRADING_RL_AGENT_NEWSAPI_KEY` (optional)

### Resource Requirements

Adjust resource requests/limits based on your cluster:

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "1000m"
    nvidia.com/gpu: 1 # For training jobs
  limits:
    memory: "8Gi"
    cpu: "2000m"
    nvidia.com/gpu: 1
```

### Storage

Modify PVC sizes and storage classes:

```yaml
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi # Adjust based on needs
  storageClassName: standard # Use your cluster's storage class
```

### Scheduling

For CronJobs, modify the schedule using cron syntax:

```yaml
spec:
  schedule: "0 2 * * *" # Daily at 2 AM UTC
  # Other options:
  # "0 */6 * * *"     # Every 6 hours
  # "0 2 * * 1-5"     # Weekdays at 2 AM
  # "0 2 1 * *"       # Monthly on 1st at 2 AM
```

## 📝 Monitoring and Logs

### View Job Status

```bash
# List all jobs
kubectl get jobs -l app=trading-rl-agent

# Get job details
kubectl describe job download-datasets-job

# View job logs
kubectl logs job/download-datasets-job
```

### View CronJob Status

```bash
# List cronjobs
kubectl get cronjobs -l app=trading-rl-agent

# View cronjob details
kubectl describe cronjob scheduled-backtest-cronjob

# View recent job logs
kubectl logs job/scheduled-backtest-cronjob-1234567890
```

### Persistent Volume Status

```bash
# Check PVC status
kubectl get pvc -l app=trading-rl-agent

# View PVC details
kubectl describe pvc trading-rl-agent-data
```

## 🛠️ Troubleshooting

### Common Issues

1. **Image Pull Errors**

   ```bash
   # Check if image exists in registry
   kubectl describe pod <pod-name>
   # Look for "ImagePullBackOff" or "ErrImagePull"
   ```

2. **Resource Constraints**

   ```bash
   # Check if pods are pending due to resources
   kubectl get pods -l app=trading-rl-agent
   kubectl describe pod <pending-pod-name>
   ```

3. **Volume Mount Issues**

   ```bash
   # Check PVC status
   kubectl get pvc
   kubectl describe pvc trading-rl-agent-data
   ```

4. **Secret Issues**
   ```bash
   # Verify secrets exist
   kubectl get secrets trading-rl-agent-secrets
   kubectl describe secret trading-rl-agent-secrets
   ```

### Debug Commands

```bash
# Get job events
kubectl get events --sort-by='.lastTimestamp' | grep trading-rl-agent

# Check pod logs
kubectl logs -f job/download-datasets-job

# Exec into running pod
kubectl exec -it <pod-name> -- /bin/bash

# Check resource usage
kubectl top pods -l app=trading-rl-agent
```

## 🔒 Security Considerations

### Production Hardening

1. **Network Policies**: Restrict pod-to-pod communication
2. **RBAC**: Use service accounts with minimal permissions
3. **Pod Security Standards**: Enable pod security admission
4. **Image Scanning**: Scan container images for vulnerabilities
5. **Secret Management**: Use external secret management (e.g., HashiCorp Vault)

### Example Network Policy

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: trading-rl-agent-network-policy
spec:
  podSelector:
    matchLabels:
      app: trading-rl-agent
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 8080
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
      ports:
        - protocol: TCP
          port: 53
```

## 📚 Additional Resources

- [Kubernetes Jobs Documentation](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [Kubernetes CronJobs Documentation](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)
- [Persistent Volumes Documentation](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [Trading RL Agent CLI Documentation](../README_CLI_USAGE.md)
