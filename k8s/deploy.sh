#!/bin/bash

# Trading RL Agent Kubernetes Deployment Script
# This script deploys all necessary resources for running Trading RL Agent jobs

set -e

echo "🚀 Deploying Trading RL Agent Kubernetes resources..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Check if we're connected to a cluster
if ! kubectl cluster-info &> /dev/null; then
    print_error "Not connected to a Kubernetes cluster"
    exit 1
fi

print_status "Connected to cluster: $(kubectl config current-context)"

# Create namespace if it doesn't exist
NAMESPACE=${NAMESPACE:-default}
if [ "$NAMESPACE" != "default" ]; then
    print_status "Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
fi

# Function to apply resources with namespace
apply_resource() {
    local file=$1
    local description=$2

    print_status "Applying $description..."
    if [ "$NAMESPACE" != "default" ]; then
        kubectl apply -f "$file" -n "$NAMESPACE"
    else
        kubectl apply -f "$file"
    fi
}

# Apply resources in order
print_status "Step 1: Creating secrets..."
apply_resource "secrets.yaml" "API secrets"

print_status "Step 2: Creating ConfigMap..."
apply_resource "configmap.yaml" "configuration"

print_status "Step 3: Creating PersistentVolumeClaims..."
apply_resource "persistent-volumes.yaml" "storage volumes"

# Wait for PVCs to be bound
print_status "Waiting for PersistentVolumeClaims to be bound..."
if [ "$NAMESPACE" != "default" ]; then
    kubectl wait --for=condition=Bound pvc -l app=trading-rl-agent -n "$NAMESPACE" --timeout=300s
else
    kubectl wait --for=condition=Bound pvc -l app=trading-rl-agent --timeout=300s
fi

print_status "✅ All resources deployed successfully!"

echo ""
print_status "Available job templates:"
echo "  • kubectl apply -f download-datasets-job.yaml"
echo "  • kubectl apply -f training-job.yaml"
echo "  • kubectl apply -f scheduled-backtest-job.yaml"
echo "  • kubectl apply -f scheduled-backtest-cronjob.yaml"

echo ""
print_warning "Remember to:"
echo "  1. Update secrets.yaml with your actual API keys"
echo "  2. Adjust resource limits based on your cluster"
echo "  3. Modify storage class names if needed"
echo "  4. Update the Docker image tag if using a different registry"

echo ""
print_status "To monitor deployments:"
echo "  kubectl get jobs -l app=trading-rl-agent"
echo "  kubectl get cronjobs -l app=trading-rl-agent"
echo "  kubectl get pvc -l app=trading-rl-agent"
