# 🚀 **Quick Start - Dependency Management**

## **For New Team Members**

### **1. Initial Setup**

```bash
# Clone the repository
git clone <repository-url>
cd trading-rl-agent

# Install core dependencies
pip install -r requirements-core.txt

# For development (includes testing tools)
pip install -r requirements-dev.txt

# For full features (includes ML)
pip install -r requirements-full.txt
```

### **2. Validate Your Environment**

```bash
# Run comprehensive validation
python scripts/validate_dependencies.py

# Expected output: "✅ All validations passed successfully!"
```

### **3. If Issues Found**

```bash
# Run automated resolution
python scripts/resolve_dependencies.py

# Re-validate
python scripts/validate_dependencies.py
```

---

## **For Different Environments**

### **Development**

```bash
pip install -r requirements-dev.txt
```

### **Testing Only**

```bash
pip install -r requirements-test.txt
```

### **CI/CD Pipeline**

```bash
pip install -r requirements-ci.txt
```

### **Production**

```bash
pip install -r requirements-production.txt
```

---

## **Common Issues & Solutions**

### **Issue: "No module named 'structlog'"**

```bash
pip install structlog>=23.1.0
```

### **Issue: "Ray initialization failed"**

```bash
pip install "ray[rllib,tune]>=2.6.0,<3.0.0"
```

### **Issue: "pytest not found"**

```bash
pip install -r requirements-test.txt
```

### **Issue: "ML dependencies missing"**

```bash
pip install -r requirements-ml.txt
```

---

## **Validation Commands**

### **Check All Dependencies**

```bash
python scripts/validate_dependencies.py
```

### **Check Specific Component**

```bash
# Test Ray compatibility
python -c "import ray; print('Ray version:', ray.__version__)"

# Test structlog
python -c "import structlog; print('Structlog version:', structlog.__version__)"

# Test ML packages
python -c "import torch, sklearn; print('PyTorch:', torch.__version__, 'Sklearn:', sklearn.__version__)"
```

---

## **Troubleshooting**

### **If Validation Fails**

1. Run `python scripts/resolve_dependencies.py`
2. Check the generated report: `dependency_validation_report.json`
3. Follow recommendations in the report

### **If Tests Fail**

1. Ensure you have test dependencies: `pip install -r requirements-test.txt`
2. Run validation: `python scripts/validate_dependencies.py`
3. Check test logs for specific import errors

### **If Ray Issues Persist**

1. Check Ray version: `python -c "import ray; print(ray.__version__)"`
2. Reinstall Ray: `pip install "ray[rllib,tune]>=2.6.0,<3.0.0" --force-reinstall`
3. Test Ray initialization: `python -c "import ray; ray.init(); ray.shutdown()"`

---

## **Best Practices**

### **Before Committing**

```bash
# Run validation
python scripts/validate_dependencies.py

# Run tests
python -m pytest tests/unit/ -v

# Check code quality
ruff check src/
black --check src/
```

### **When Adding New Dependencies**

1. Add to appropriate requirements file
2. Update version constraints
3. Run validation: `python scripts/validate_dependencies.py`
4. Test in clean environment

### **For Production Deployment**

1. Use `requirements-production.txt`
2. Run validation in production environment
3. Test all critical paths
4. Monitor for dependency conflicts

---

## **Quick Reference**

| Environment   | Requirements File             | Purpose                |
| ------------- | ----------------------------- | ---------------------- |
| Core          | `requirements-core.txt`       | Minimal functionality  |
| Development   | `requirements-dev.txt`        | Full development tools |
| Testing       | `requirements-test.txt`       | Test dependencies only |
| CI/CD         | `requirements-ci.txt`         | Optimized for CI       |
| Production    | `requirements-production.txt` | Production deployment  |
| Full Features | `requirements-full.txt`       | Complete system        |

| Script                     | Purpose                  | Usage                                     |
| -------------------------- | ------------------------ | ----------------------------------------- |
| `validate_dependencies.py` | Check all dependencies   | `python scripts/validate_dependencies.py` |
| `resolve_dependencies.py`  | Fix common issues        | `python scripts/resolve_dependencies.py`  |
| `analyze-deps.py`          | Analyze dependency sizes | `python analyze-deps.py`                  |

---

**Need Help?** Check the full documentation: `DEPENDENCY_RESOLUTION_PLAN.md`
