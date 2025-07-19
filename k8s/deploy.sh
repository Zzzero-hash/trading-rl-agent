#!/bin/bash

# Trading System Kubernetes Deployment Script
# This script deploys the complete trading system with all microservices

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="trading-system"
REGISTRY="your-registry.com"
IMAGE_TAG="latest"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    print_success "kubectl found"
}

# Function to check if namespace exists
check_namespace() {
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        print_warning "Namespace $NAMESPACE already exists"
    else
        print_status "Creating namespace $NAMESPACE"
        kubectl apply -f namespace.yaml
    fi
}

# Function to deploy secrets
deploy_secrets() {
    print_status "Deploying secrets..."
    
    # Check if secrets file exists
    if [ ! -f "secrets.yaml" ]; then
        print_error "secrets.yaml not found. Please create it with your actual secrets."
        exit 1
    fi
    
    kubectl apply -f secrets.yaml
    print_success "Secrets deployed"
}

# Function to deploy infrastructure
deploy_infrastructure() {
    print_status "Deploying infrastructure services..."
    
    # Deploy persistent volumes
    kubectl apply -f persistent-volumes.yaml
    
    # Deploy infrastructure services
    kubectl apply -f infrastructure-services.yaml
    
    # Wait for infrastructure to be ready
    print_status "Waiting for infrastructure services to be ready..."
    kubectl wait --for=condition=ready pod -l app=trading-db -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=trading-redis -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=trading-rabbitmq -n $NAMESPACE --timeout=300s
    
    print_success "Infrastructure services deployed"
}

# Function to deploy monitoring
deploy_monitoring() {
    print_status "Deploying monitoring stack..."
    
    kubectl apply -f monitoring-stack.yaml
    
    # Wait for monitoring to be ready
    print_status "Waiting for monitoring services to be ready..."
    kubectl wait --for=condition=ready pod -l app=prometheus -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=grafana -n $NAMESPACE --timeout=300s
    
    print_success "Monitoring stack deployed"
}

# Function to deploy microservices
deploy_microservices() {
    print_status "Deploying microservices..."
    
    # Deploy API service
    kubectl apply -f api-service-deployment.yaml
    
    # Deploy trading engine
    kubectl apply -f trading-engine-deployment.yaml
    
    # Deploy ML service
    kubectl apply -f ml-service-deployment.yaml
    
    # Deploy data pipeline
    kubectl apply -f data-pipeline-deployment.yaml
    
    # Wait for microservices to be ready
    print_status "Waiting for microservices to be ready..."
    kubectl wait --for=condition=ready pod -l app=trading-api-service -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=trading-engine -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=trading-ml-service -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=trading-data-pipeline -n $NAMESPACE --timeout=300s
    
    print_success "Microservices deployed"
}

# Function to deploy autoscaling
deploy_autoscaling() {
    print_status "Deploying autoscaling configurations..."
    
    kubectl apply -f autoscaling.yaml
    
    print_success "Autoscaling configurations deployed"
}

# Function to deploy ingress
deploy_ingress() {
    print_status "Deploying ingress configurations..."
    
    kubectl apply -f ingress.yaml
    
    print_success "Ingress configurations deployed"
}

# Function to deploy batch jobs
deploy_batch_jobs() {
    print_status "Deploying batch job configurations..."
    
    # Update existing job files to use the new namespace
    sed -i "s/namespace: default/namespace: $NAMESPACE/g" *.yaml 2>/dev/null || true
    
    kubectl apply -f training-job.yaml
    kubectl apply -f download-datasets-job.yaml
    kubectl apply -f scheduled-backtest-job.yaml
    kubectl apply -f scheduled-backtest-cronjob.yaml
    
    print_success "Batch job configurations deployed"
}

# Function to verify deployment
verify_deployment() {
    print_status "Verifying deployment..."
    
    # Check all pods are running
    echo "Checking pod status..."
    kubectl get pods -n $NAMESPACE
    
    # Check services
    echo "Checking services..."
    kubectl get services -n $NAMESPACE
    
    # Check deployments
    echo "Checking deployments..."
    kubectl get deployments -n $NAMESPACE
    
    # Check ingress
    echo "Checking ingress..."
    kubectl get ingress -n $NAMESPACE
    
    # Check HPA
    echo "Checking horizontal pod autoscalers..."
    kubectl get hpa -n $NAMESPACE
    
    print_success "Deployment verification completed"
}

# Function to show access information
show_access_info() {
    print_success "Deployment completed successfully!"
    echo ""
    echo "Access Information:"
    echo "=================="
    echo "API Service: http://api.trading-system.local"
    echo "Grafana Dashboard: http://dashboard.trading-system.local"
    echo "Prometheus: http://monitoring.trading-system.local"
    echo "RabbitMQ Management: http://rabbitmq.internal:15672"
    echo ""
    echo "To access services, add the following to your /etc/hosts:"
    echo "127.0.0.1 api.trading-system.local"
    echo "127.0.0.1 dashboard.trading-system.local"
    echo "127.0.0.1 monitoring.trading-system.local"
    echo ""
    echo "Useful commands:"
    echo "kubectl get pods -n $NAMESPACE"
    echo "kubectl logs -f deployment/trading-api-service -n $NAMESPACE"
    echo "kubectl port-forward service/grafana 3000:3000 -n $NAMESPACE"
    echo "kubectl port-forward service/prometheus 9090:9090 -n $NAMESPACE"
}

# Function to rollback deployment
rollback_deployment() {
    print_warning "Rolling back deployment..."
    
    # Delete all resources in reverse order
    kubectl delete -f autoscaling.yaml --ignore-not-found=true
    kubectl delete -f ingress.yaml --ignore-not-found=true
    kubectl delete -f api-service-deployment.yaml --ignore-not-found=true
    kubectl delete -f trading-engine-deployment.yaml --ignore-not-found=true
    kubectl delete -f ml-service-deployment.yaml --ignore-not-found=true
    kubectl delete -f data-pipeline-deployment.yaml --ignore-not-found=true
    kubectl delete -f monitoring-stack.yaml --ignore-not-found=true
    kubectl delete -f infrastructure-services.yaml --ignore-not-found=true
    kubectl delete -f persistent-volumes.yaml --ignore-not-found=true
    kubectl delete -f secrets.yaml --ignore-not-found=true
    kubectl delete -f namespace.yaml --ignore-not-found=true
    
    print_success "Rollback completed"
}

# Main deployment function
main() {
    print_status "Starting Trading System deployment..."
    print_status "Environment: $ENVIRONMENT"
    print_status "Namespace: $NAMESPACE"
    print_status "Image tag: $IMAGE_TAG"
    
    # Check prerequisites
    check_kubectl
    
    # Deploy in order
    check_namespace
    deploy_secrets
    deploy_infrastructure
    deploy_monitoring
    deploy_microservices
    deploy_autoscaling
    deploy_ingress
    deploy_batch_jobs
    
    # Verify deployment
    verify_deployment
    
    # Show access information
    show_access_info
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback_deployment
        ;;
    "verify")
        verify_deployment
        ;;
    "status")
        kubectl get all -n $NAMESPACE
        ;;
    "logs")
        SERVICE="${2:-trading-api-service}"
        kubectl logs -f deployment/$SERVICE -n $NAMESPACE
        ;;
    "port-forward")
        SERVICE="${2:-grafana}"
        PORT="${3:-3000}"
        kubectl port-forward service/$SERVICE $PORT:$PORT -n $NAMESPACE
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy      - Deploy the complete trading system (default)"
        echo "  rollback    - Rollback the deployment"
        echo "  verify      - Verify the deployment status"
        echo "  status      - Show current status"
        echo "  logs [svc]  - Show logs for a service"
        echo "  port-forward [svc] [port] - Port forward a service"
        echo "  help        - Show this help message"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
