#!/bin/bash

# ArgoCD Setup Script for Trading RL Agent
# This script installs and configures ArgoCD for GitOps deployment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGOCD_NAMESPACE="argocd"
TRADING_NAMESPACE="trading-system"
GITHUB_REPO="https://github.com/yourusername/trade-agent.git"
GITHUB_BRANCH="main"

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi

    # Check if helm is available
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi

    # Check if argocd CLI is available
    if ! command -v argocd &> /dev/null; then
        log_warning "argocd CLI is not installed. Installing..."
        install_argocd_cli
    fi

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Install ArgoCD CLI
install_argocd_cli() {
    log_info "Installing ArgoCD CLI..."

    # Detect OS and install appropriate version
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)

    if [ "$ARCH" = "x86_64" ]; then
        ARCH="amd64"
    elif [ "$ARCH" = "aarch64" ]; then
        ARCH="arm64"
    fi

    # Download and install ArgoCD CLI
    curl -sSL -o argocd-linux-$ARCH https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-$ARCH
    sudo install -m 555 argocd-linux-$ARCH /usr/local/bin/argocd
    rm argocd-linux-$ARCH

    log_success "ArgoCD CLI installed"
}

# Install ArgoCD using Helm
install_argocd() {
    log_info "Installing ArgoCD using Helm..."

    # Add ArgoCD Helm repository
    helm repo add argo https://argoproj.github.io/argo-helm
    helm repo update

    # Create namespace if it doesn't exist
    kubectl create namespace $ARGOCD_NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

    # Install ArgoCD
    helm upgrade --install argocd argo/argo-cd \
        --namespace $ARGOCD_NAMESPACE \
        --set server.ingress.enabled=true \
        --set server.ingress.hosts[0]=argocd.trading-system.local \
        --set server.ingress.annotations."kubernetes\.io/ingress\.class"=nginx \
        --set server.ingress.annotations."cert-manager\.io/cluster-issuer"=letsencrypt-prod \
        --set server.ingress.tls[0].secretName=argocd-server-tls \
        --set server.ingress.tls[0].hosts[0]=argocd.trading-system.local \
        --set server.extraArgs[0]=--insecure \
        --set ha.enabled=true \
        --set ha.replicas=3 \
        --set applicationSet.enabled=true \
        --set notifications.enabled=true \
        --set rbac.defaultPolicy=role:readonly \
        --set rbac.pspEnabled=false \
        --wait --timeout=10m

    log_success "ArgoCD installed successfully"
}

# Wait for ArgoCD to be ready
wait_for_argocd() {
    log_info "Waiting for ArgoCD to be ready..."

    # Wait for ArgoCD server to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n $ARGOCD_NAMESPACE

    # Wait for ArgoCD application controller to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/argocd-application-controller -n $ARGOCD_NAMESPACE

    # Wait for ArgoCD repo server to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/argocd-repo-server -n $ARGOCD_NAMESPACE

    log_success "ArgoCD is ready"
}

# Get ArgoCD admin password
get_admin_password() {
    log_info "Getting ArgoCD admin password..."

    # Get the initial admin password
    ADMIN_PASSWORD=$(kubectl -n $ARGOCD_NAMESPACE get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)

    echo "=========================================="
    echo "ArgoCD Admin Password: $ADMIN_PASSWORD"
    echo "ArgoCD URL: https://argocd.trading-system.local"
    echo "Username: admin"
    echo "=========================================="

    # Store password in a file for later use
    echo "$ADMIN_PASSWORD" > argocd-admin-password.txt
    chmod 600 argocd-admin-password.txt

    log_success "Admin password saved to argocd-admin-password.txt"
}

# Login to ArgoCD
login_to_argocd() {
    log_info "Logging in to ArgoCD..."

    # Get admin password
    ADMIN_PASSWORD=$(cat argocd-admin-password.txt)

    # Login to ArgoCD
    argocd login argocd.trading-system.local \
        --username admin \
        --password "$ADMIN_PASSWORD" \
        --insecure \
        --grpc-web

    log_success "Logged in to ArgoCD"
}

# Create ArgoCD projects
create_projects() {
    log_info "Creating ArgoCD projects..."

    # Create default project
    argocd proj create default \
        --description "Default project for trading system" \
        --dest https://kubernetes.default.svc,https://kubernetes.default.svc \
        --src https://github.com/yourusername/trade-agent.git \
        --allow-cluster-resource "*" \
        --allow-namespace-resource "*" \
        --orphaned-resources warn

    # Create trading project
    argocd proj create trading \
        --description "Trading system specific project" \
        --dest https://kubernetes.default.svc,trading-system \
        --dest https://kubernetes.default.svc,trading-system-staging \
        --src https://github.com/yourusername/trade-agent.git \
        --allow-cluster-resource "*" \
        --allow-namespace-resource "*" \
        --orphaned-resources warn

    log_success "ArgoCD projects created"
}

# Deploy trading system applications
deploy_applications() {
    log_info "Deploying trading system applications..."

    # Apply ArgoCD applications
    kubectl apply -f "$SCRIPT_DIR/argocd-installation.yaml"
    kubectl apply -f "$SCRIPT_DIR/trading-system-app.yaml"
    kubectl apply -f "$SCRIPT_DIR/application-set.yaml"
    kubectl apply -f "$SCRIPT_DIR/notifications.yaml"

    # Wait for applications to be created
    sleep 10

    # Sync the main trading system application
    argocd app sync trading-system --prune --force

    log_success "Trading system applications deployed"
}

# Configure notifications
configure_notifications() {
    log_info "Configuring ArgoCD notifications..."

    # Update notification configuration
    kubectl patch configmap argocd-notifications-cm -n $ARGOCD_NAMESPACE --patch-file "$SCRIPT_DIR/notifications.yaml"

    # Restart ArgoCD server to pick up notification changes
    kubectl rollout restart deployment/argocd-server -n $ARGOCD_NAMESPACE

    log_success "Notifications configured"
}

# Create RBAC for trading team
create_rbac() {
    log_info "Creating RBAC for trading team..."

    # Create trading team role
    kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: trading-team-role
rules:
  - apiGroups: ["argoproj.io"]
    resources: ["applications", "applicationsets"]
    verbs: ["get", "list", "watch", "update", "patch", "delete", "create"]
  - apiGroups: [""]
    resources: ["pods", "services", "configmaps", "secrets"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments", "statefulsets", "daemonsets"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: trading-team-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: trading-team-role
subjects:
  - kind: ServiceAccount
    name: trading-team-sa
    namespace: trading-system
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: trading-team-sa
  namespace: trading-system
EOF

    log_success "RBAC created for trading team"
}

# Setup monitoring and alerting
setup_monitoring() {
    log_info "Setting up monitoring and alerting..."

    # Create monitoring namespace
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

    # Install Prometheus and Grafana
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update

    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --set grafana.enabled=true \
        --set prometheus.prometheusSpec.retention=7d \
        --wait --timeout=10m

    log_success "Monitoring stack installed"
}

# Create deployment script
create_deployment_script() {
    log_info "Creating deployment script..."

    cat > deploy-trading-system.sh <<'EOF'
#!/bin/bash

# Trading System Deployment Script using ArgoCD
# This script triggers deployment through ArgoCD GitOps

set -euo pipefail

ENVIRONMENT="${1:-production}"
IMAGE_TAG="${2:-latest}"

echo "Deploying trading system to $ENVIRONMENT with image tag $IMAGE_TAG"

# Update image tag in deployment files
find k8s -name "*-deployment.yaml" -type f -exec sed -i "s|image:.*trade-agent.*|image: ghcr.io/\${GITHUB_REPOSITORY}:${IMAGE_TAG}|g" {} \;

# Commit and push changes
git add .
git commit -m "Deploy trading system to $ENVIRONMENT with image $IMAGE_TAG"
git push origin main

echo "Deployment triggered. Check ArgoCD UI for status:"
echo "https://argocd.trading-system.local"
EOF

    chmod +x deploy-trading-system.sh

    log_success "Deployment script created: deploy-trading-system.sh"
}

# Main installation function
main() {
    log_info "Starting ArgoCD setup for Trading RL Agent..."

    check_prerequisites
    install_argocd
    wait_for_argocd
    get_admin_password
    login_to_argocd
    create_projects
    deploy_applications
    configure_notifications
    create_rbac
    setup_monitoring
    create_deployment_script

    log_success "ArgoCD setup completed successfully!"

    echo ""
    echo "=========================================="
    echo "ArgoCD Setup Complete!"
    echo "=========================================="
    echo "ArgoCD URL: https://argocd.trading-system.local"
    echo "Username: admin"
    echo "Password: $(cat argocd-admin-password.txt)"
    echo ""
    echo "Next steps:"
    echo "1. Update your DNS to point argocd.trading-system.local to your cluster"
    echo "2. Configure Slack notifications in k8s/argocd/notifications.yaml"
    echo "3. Update the GitHub repository URL in the ArgoCD applications"
    echo "4. Use deploy-trading-system.sh for future deployments"
    echo "=========================================="
}

# Run main function
main "$@"
