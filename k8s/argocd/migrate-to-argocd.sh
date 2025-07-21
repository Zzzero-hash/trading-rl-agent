#!/bin/bash

# Migration Script: Manual K8s Deployments to ArgoCD GitOps
# This script helps migrate from manual kubectl deployments to ArgoCD

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRADING_NAMESPACE="trading-system"
ARGOCD_NAMESPACE="argocd"
BACKUP_DIR="/tmp/trading-system-migration-backup"

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
    log_info "Checking migration prerequisites..."

    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi

    # Check if argocd CLI is available
    if ! command -v argocd &> /dev/null; then
        log_error "argocd CLI is not installed. Please install it first."
        exit 1
    fi

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check if trading system namespace exists
    if ! kubectl get namespace "$TRADING_NAMESPACE" &> /dev/null; then
        log_error "Trading system namespace does not exist"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Create backup of current deployment
create_backup() {
    log_info "Creating backup of current deployment..."

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="trading-system-backup-${timestamp}"
    local backup_path="${BACKUP_DIR}/${backup_name}"

    mkdir -p "$backup_path"

    # Backup all resources in the trading system namespace
    kubectl get all -n "$TRADING_NAMESPACE" -o yaml > "$backup_path/all_resources.yaml" 2>/dev/null || true
    kubectl get configmaps -n "$TRADING_NAMESPACE" -o yaml > "$backup_path/configmaps.yaml" 2>/dev/null || true
    kubectl get secrets -n "$TRADING_NAMESPACE" -o yaml > "$backup_path/secrets.yaml" 2>/dev/null || true
    kubectl get persistentvolumeclaims -n "$TRADING_NAMESPACE" -o yaml > "$backup_path/pvcs.yaml" 2>/dev/null || true
    kubectl get ingresses -n "$TRADING_NAMESPACE" -o yaml > "$backup_path/ingresses.yaml" 2>/dev/null || true
    kubectl get horizontalpodautoscalers -n "$TRADING_NAMESPACE" -o yaml > "$backup_path/hpas.yaml" 2>/dev/null || true

    # Store current image tags
    kubectl get deployments -n "$TRADING_NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.spec.template.spec.containers[0].image}{"\n"}{end}' > "$backup_path/image_tags.txt" 2>/dev/null || true

    # Create a restore script
    cat > "$backup_path/restore.sh" <<EOF
#!/bin/bash
# Restore script for trading system backup
echo "Restoring trading system from backup..."
kubectl apply -f all_resources.yaml
kubectl apply -f configmaps.yaml
kubectl apply -f secrets.yaml
kubectl apply -f pvcs.yaml
kubectl apply -f ingresses.yaml
kubectl apply -f hpas.yaml
echo "Restore completed"
EOF

    chmod +x "$backup_path/restore.sh"

    log_success "Backup created: $backup_path"
    echo "Backup location: $backup_path"
    echo "To restore: cd $backup_path && ./restore.sh"
}

# Check current deployment status
check_current_status() {
    log_info "Checking current deployment status..."

    echo "Current pods in $TRADING_NAMESPACE:"
    kubectl get pods -n "$TRADING_NAMESPACE"

    echo ""
    echo "Current services in $TRADING_NAMESPACE:"
    kubectl get services -n "$TRADING_NAMESPACE"

    echo ""
    echo "Current deployments in $TRADING_NAMESPACE:"
    kubectl get deployments -n "$TRADING_NAMESPACE"

    echo ""
    echo "Current image tags:"
    kubectl get deployments -n "$TRADING_NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.spec.template.spec.containers[0].image}{"\n"}{end}'
}

# Install ArgoCD
install_argocd() {
    log_info "Installing ArgoCD..."

    # Check if ArgoCD is already installed
    if kubectl get namespace "$ARGOCD_NAMESPACE" &> /dev/null; then
        log_warning "ArgoCD namespace already exists. Skipping installation."
        return 0
    fi

    # Run the ArgoCD setup script
    if [ -f "$SCRIPT_DIR/argocd-setup.sh" ]; then
        log_info "Running ArgoCD setup script..."
        "$SCRIPT_DIR/argocd-setup.sh"
    else
        log_error "ArgoCD setup script not found at $SCRIPT_DIR/argocd-setup.sh"
        exit 1
    fi

    log_success "ArgoCD installation completed"
}

# Configure ArgoCD applications
configure_argocd_apps() {
    log_info "Configuring ArgoCD applications..."

    # Apply ArgoCD application manifests
    kubectl apply -f "$SCRIPT_DIR/argocd-installation.yaml"
    kubectl apply -f "$SCRIPT_DIR/trading-system-app.yaml"
    kubectl apply -f "$SCRIPT_DIR/application-set.yaml"
    kubectl apply -f "$SCRIPT_DIR/notifications.yaml"

    # Wait for applications to be created
    sleep 10

    log_success "ArgoCD applications configured"
}

# Verify ArgoCD setup
verify_argocd_setup() {
    log_info "Verifying ArgoCD setup..."

    # Check if ArgoCD server is running
    if ! kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n "$ARGOCD_NAMESPACE"; then
        log_error "ArgoCD server is not ready"
        return 1
    fi

    # Check if ArgoCD application controller is running
    if ! kubectl wait --for=condition=available --timeout=300s deployment/argocd-application-controller -n "$ARGOCD_NAMESPACE"; then
        log_error "ArgoCD application controller is not ready"
        return 1
    fi

    # Get ArgoCD admin password
    ADMIN_PASSWORD=$(kubectl -n "$ARGOCD_NAMESPACE" get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)

    echo "=========================================="
    echo "ArgoCD Setup Verification"
    echo "=========================================="
    echo "ArgoCD Server: Available"
    echo "ArgoCD Controller: Available"
    echo "Admin Password: $ADMIN_PASSWORD"
    echo "=========================================="

    log_success "ArgoCD setup verified"
}

# Sync applications with ArgoCD
sync_applications() {
    log_info "Syncing applications with ArgoCD..."

    # Get ArgoCD admin password
    ADMIN_PASSWORD=$(kubectl -n "$ARGOCD_NAMESPACE" get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)

    # Login to ArgoCD
    argocd login argocd.trading-system.local \
        --username admin \
        --password "$ADMIN_PASSWORD" \
        --insecure \
        --grpc-web

    # Sync the main trading system application
    argocd app sync trading-system --prune --force

    # Wait for sync to complete
    argocd app wait trading-system --health --timeout=300

    log_success "Applications synced with ArgoCD"
}

# Verify migration
verify_migration() {
    log_info "Verifying migration..."

    # Check if all applications are synced
    echo "ArgoCD application status:"
    argocd app list

    echo ""
    echo "Trading system pods after migration:"
    kubectl get pods -n "$TRADING_NAMESPACE"

    echo ""
    echo "Trading system services after migration:"
    kubectl get services -n "$TRADING_NAMESPACE"

    # Check application health
    echo ""
    echo "Application health status:"
    argocd app get trading-system --output yaml | grep -A 5 "health:"

    log_success "Migration verification completed"
}

# Create post-migration script
create_post_migration_script() {
    log_info "Creating post-migration script..."

    cat > post-migration-commands.sh <<'EOF'
#!/bin/bash

# Post-Migration Commands for Trading System
# Use these commands to manage your ArgoCD deployment

echo "Post-Migration Commands for Trading System"
echo "=========================================="

# Get ArgoCD admin password
ADMIN_PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)

echo "1. Access ArgoCD UI:"
echo "   URL: https://argocd.trading-system.local"
echo "   Username: admin"
echo "   Password: $ADMIN_PASSWORD"
echo ""

echo "2. List all applications:"
echo "   argocd app list"
echo ""

echo "3. Get application status:"
echo "   argocd app get trading-system"
echo ""

echo "4. Sync application:"
echo "   argocd app sync trading-system"
echo ""

echo "5. Rollback application:"
echo "   argocd app rollback trading-system"
echo ""

echo "6. View application logs:"
echo "   argocd app logs trading-system"
echo ""

echo "7. Check application health:"
echo "   argocd app get trading-system --output yaml | grep -A 5 'health:'"
echo ""

echo "8. Deploy new version:"
echo "   ./deploy-trading-system.sh production v2.1.0"
echo ""

echo "9. View cluster resources:"
echo "   kubectl get all -n trading-system"
echo ""

echo "10. Check application events:"
echo "    kubectl get events -n trading-system --sort-by='.lastTimestamp'"
echo ""

echo "=========================================="
echo "Migration completed successfully!"
echo "Your trading system is now managed by ArgoCD GitOps"
echo "=========================================="
EOF

    chmod +x post-migration-commands.sh

    log_success "Post-migration script created: post-migration-commands.sh"
}

# Main migration function
main() {
    log_info "Starting migration from manual K8s deployments to ArgoCD GitOps..."

    echo "=========================================="
    echo "Trading System Migration to ArgoCD"
    echo "=========================================="
    echo "This script will:"
    echo "1. Create a backup of current deployment"
    echo "2. Install and configure ArgoCD"
    echo "3. Migrate applications to ArgoCD management"
    echo "4. Verify the migration"
    echo "=========================================="

    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Migration cancelled by user"
        exit 0
    fi

    check_prerequisites
    create_backup
    check_current_status

    echo ""
    read -p "Current deployment status shown above. Continue with ArgoCD installation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Migration cancelled by user"
        exit 0
    fi

    install_argocd
    configure_argocd_apps
    verify_argocd_setup

    echo ""
    read -p "ArgoCD is ready. Continue with application migration? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Migration cancelled by user"
        exit 0
    fi

    sync_applications
    verify_migration
    create_post_migration_script

    log_success "Migration completed successfully!"

    echo ""
    echo "=========================================="
    echo "Migration Complete!"
    echo "=========================================="
    echo "Your trading system is now managed by ArgoCD GitOps"
    echo ""
    echo "Next steps:"
    echo "1. Run: ./post-migration-commands.sh"
    echo "2. Access ArgoCD UI: https://argocd.trading-system.local"
    echo "3. Update your CI/CD pipeline to use ArgoCD"
    echo "4. Remove the old deploy.sh script"
    echo "5. Update your documentation"
    echo "=========================================="
}

# Run main function
main "$@"
