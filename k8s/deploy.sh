#!/bin/bash

# Enhanced Deployment Script for Trading System
# Supports deployment, rollback, health checks, and monitoring

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="trading-system"
ENVIRONMENT="${ENVIRONMENT:-staging}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BACKUP_DIR="/tmp/trading-system-backups"
MAX_BACKUPS=5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
}

# Check if namespace exists
check_namespace() {
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE does not exist. Creating..."
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace $NAMESPACE created"
    fi
}

# Create backup of current deployment
create_backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="backup_${ENVIRONMENT}_${timestamp}"
    local backup_path="${BACKUP_DIR}/${backup_name}"
    
    log_info "Creating backup: $backup_name"
    
    mkdir -p "$backup_path"
    
    # Backup all resources in the namespace
    kubectl get all -n "$NAMESPACE" -o yaml > "$backup_path/all_resources.yaml" 2>/dev/null || true
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "$backup_path/configmaps.yaml" 2>/dev/null || true
    kubectl get secrets -n "$NAMESPACE" -o yaml > "$backup_path/secrets.yaml" 2>/dev/null || true
    kubectl get persistentvolumeclaims -n "$NAMESPACE" -o yaml > "$backup_path/pvcs.yaml" 2>/dev/null || true
    
    # Store current image tags
    kubectl get deployments -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.spec.template.spec.containers[0].image}{"\n"}{end}' > "$backup_path/image_tags.txt" 2>/dev/null || true
    
    log_success "Backup created: $backup_path"
    
    # Clean up old backups
    cleanup_old_backups
}

# Clean up old backups
cleanup_old_backups() {
    if [ -d "$BACKUP_DIR" ]; then
        local backup_count=$(find "$BACKUP_DIR" -name "backup_${ENVIRONMENT}_*" | wc -l)
        if [ "$backup_count" -gt "$MAX_BACKUPS" ]; then
            log_info "Cleaning up old backups..."
            find "$BACKUP_DIR" -name "backup_${ENVIRONMENT}_*" -type d -printf '%T@ %p\n' | sort -n | head -n $((backup_count - MAX_BACKUPS)) | cut -d' ' -f2- | xargs rm -rf
            log_success "Old backups cleaned up"
        fi
    fi
}

# Update image tags in deployment files
update_image_tags() {
    log_info "Updating image tags to: $IMAGE_TAG"
    
    # Update image tags in all deployment files
    find . -name "*-deployment.yaml" -type f | while read -r file; do
        if [ -f "$file" ]; then
            sed -i "s|image:.*trading-rl-agent.*|image: ghcr.io/\${GITHUB_REPOSITORY}:${IMAGE_TAG}|g" "$file"
            log_info "Updated image tag in $file"
        fi
    done
}

# Deploy to Kubernetes
deploy() {
    log_info "Starting deployment to $ENVIRONMENT environment"
    
    # Check prerequisites
    check_kubectl
    check_namespace
    
    # Create backup before deployment
    create_backup
    
    # Update image tags
    update_image_tags
    
    # Apply namespace
    log_info "Applying namespace configuration..."
    kubectl apply -f namespace.yaml
    
    # Apply infrastructure services first
    log_info "Deploying infrastructure services..."
    kubectl apply -f infrastructure-services.yaml
    
    # Wait for infrastructure to be ready
    log_info "Waiting for infrastructure services to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/redis -n "$NAMESPACE" || true
    kubectl wait --for=condition=available --timeout=300s deployment/postgres -n "$NAMESPACE" || true
    
    # Apply persistent volumes
    log_info "Applying persistent volumes..."
    kubectl apply -f persistent-volumes.yaml
    
    # Apply secrets and configmaps
    log_info "Applying secrets and configmaps..."
    kubectl apply -f secrets.yaml
    kubectl apply -f configmap.yaml
    
    # Apply core services
    log_info "Deploying core services..."
    kubectl apply -f data-pipeline-deployment.yaml
    kubectl apply -f ml-service-deployment.yaml
    kubectl apply -f trading-engine-deployment.yaml
    kubectl apply -f api-service-deployment.yaml
    
    # Apply ingress
    log_info "Applying ingress configuration..."
    kubectl apply -f ingress.yaml
    
    # Apply autoscaling
    log_info "Applying autoscaling configuration..."
    kubectl apply -f autoscaling.yaml
    
    log_success "Deployment completed successfully"
}

# Rollback to previous deployment
rollback() {
    log_info "Starting rollback for $ENVIRONMENT environment"
    
    # Find the most recent backup
    local latest_backup=$(find "$BACKUP_DIR" -name "backup_${ENVIRONMENT}_*" -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -z "$latest_backup" ] || [ ! -d "$latest_backup" ]; then
        log_error "No backup found for rollback"
        exit 1
    fi
    
    log_info "Rolling back to: $latest_backup"
    
    # Restore from backup
    if [ -f "$latest_backup/all_resources.yaml" ]; then
        kubectl apply -f "$latest_backup/all_resources.yaml" --force
    fi
    
    if [ -f "$latest_backup/configmaps.yaml" ]; then
        kubectl apply -f "$latest_backup/configmaps.yaml" --force
    fi
    
    if [ -f "$latest_backup/secrets.yaml" ]; then
        kubectl apply -f "$latest_backup/secrets.yaml" --force
    fi
    
    log_success "Rollback completed successfully"
}

# Verify deployment health
verify() {
    log_info "Verifying deployment health..."
    
    # Check if all pods are running
    local pod_status=$(kubectl get pods -n "$NAMESPACE" --no-headers | grep -v "Running\|Completed" | wc -l)
    if [ "$pod_status" -gt 0 ]; then
        log_error "Some pods are not in Running state"
        kubectl get pods -n "$NAMESPACE"
        return 1
    fi
    
    # Check service endpoints
    local services=("api-service" "trading-engine" "data-pipeline" "ml-service")
    for service in "${services[@]}"; do
        if ! kubectl get endpoints "$service" -n "$NAMESPACE" | grep -q "ENDPOINTS"; then
            log_warning "Service $service has no endpoints"
        else
            log_success "Service $service is healthy"
        fi
    done
    
    # Check ingress
    if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
        log_success "Ingress is configured"
    else
        log_warning "No ingress found"
    fi
    
    log_success "Deployment verification completed"
}

# Run smoke tests
smoke_test() {
    log_info "Running smoke tests..."
    
    # Get service URLs
    local api_url=""
    if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
        api_url=$(kubectl get ingress -n "$NAMESPACE" -o jsonpath='{.items[0].spec.rules[0].host}')
    else
        api_url="localhost:8000"
    fi
    
    # Test API health endpoint
    log_info "Testing API health endpoint..."
    if curl -f -s "http://$api_url/health" > /dev/null; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi
    
    # Test trading engine health
    log_info "Testing trading engine health..."
    if kubectl exec -n "$NAMESPACE" deployment/trading-engine -- python -c "import trading_rl_agent; print('Trading engine healthy')" 2>/dev/null; then
        log_success "Trading engine health check passed"
    else
        log_error "Trading engine health check failed"
        return 1
    fi
    
    # Test data pipeline
    log_info "Testing data pipeline..."
    if kubectl exec -n "$NAMESPACE" deployment/data-pipeline -- python -c "import trading_rl_agent.data; print('Data pipeline healthy')" 2>/dev/null; then
        log_success "Data pipeline health check passed"
    else
        log_error "Data pipeline health check failed"
        return 1
    fi
    
    log_success "All smoke tests passed"
}

# Monitor deployment
monitor() {
    log_info "Starting deployment monitoring..."
    
    # Monitor for 10 minutes
    local duration=600
    local interval=30
    local elapsed=0
    
    while [ $elapsed -lt $duration ]; do
        log_info "Monitoring deployment... ($elapsed/$duration seconds)"
        
        # Check pod status
        local failed_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers | grep -v "Running\|Completed" | wc -l)
        if [ "$failed_pods" -gt 0 ]; then
            log_error "Found $failed_pods failed pods"
            kubectl get pods -n "$NAMESPACE"
            return 1
        fi
        
        # Check resource usage
        kubectl top pods -n "$NAMESPACE" 2>/dev/null || true
        
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    
    log_success "Monitoring completed successfully"
}

# Clean up resources
cleanup() {
    log_info "Cleaning up resources..."
    
    # Delete all resources in namespace
    kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
    
    # Clean up backups
    if [ -d "$BACKUP_DIR" ]; then
        rm -rf "$BACKUP_DIR"
    fi
    
    log_success "Cleanup completed"
}

# Show deployment status
status() {
    log_info "Deployment status for $ENVIRONMENT environment:"
    
    echo ""
    echo "=== Pods ==="
    kubectl get pods -n "$NAMESPACE"
    
    echo ""
    echo "=== Services ==="
    kubectl get services -n "$NAMESPACE"
    
    echo ""
    echo "=== Deployments ==="
    kubectl get deployments -n "$NAMESPACE"
    
    echo ""
    echo "=== Ingress ==="
    kubectl get ingress -n "$NAMESPACE" 2>/dev/null || echo "No ingress found"
    
    echo ""
    echo "=== Recent Events ==="
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
}

# Show available backups
list_backups() {
    log_info "Available backups for $ENVIRONMENT environment:"
    
    if [ -d "$BACKUP_DIR" ]; then
        find "$BACKUP_DIR" -name "backup_${ENVIRONMENT}_*" -type d -printf '%T@ %p\n' | sort -n | while read -r line; do
            local timestamp=$(echo "$line" | cut -d' ' -f1)
            local path=$(echo "$line" | cut -d' ' -f2-)
            local date=$(date -d "@$timestamp" '+%Y-%m-%d %H:%M:%S')
            echo "$date - $path"
        done
    else
        echo "No backups found"
    fi
}

# Main script logic
main() {
    local command="${1:-help}"
    
    case "$command" in
        deploy)
            deploy
            verify
            smoke_test
            ;;
        rollback)
            rollback
            verify
            smoke_test
            ;;
        verify)
            verify
            ;;
        smoke-test)
            smoke_test
            ;;
        monitor)
            monitor
            ;;
        status)
            status
            ;;
        backup)
            create_backup
            ;;
        list-backups)
            list_backups
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  deploy         - Deploy the application"
            echo "  rollback       - Rollback to previous deployment"
            echo "  verify         - Verify deployment health"
            echo "  smoke-test     - Run smoke tests"
            echo "  monitor        - Monitor deployment for issues"
            echo "  status         - Show deployment status"
            echo "  backup         - Create backup of current deployment"
            echo "  list-backups   - List available backups"
            echo "  cleanup        - Clean up all resources"
            echo "  help           - Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  ENVIRONMENT    - Deployment environment (default: staging)"
            echo "  IMAGE_TAG      - Docker image tag (default: latest)"
            echo "  NAMESPACE      - Kubernetes namespace (default: trading-system)"
            ;;
        *)
            log_error "Unknown command: $command"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
