import sys

import coverage
import pytest

files = [
    "src/agents/configs.py",
    "src/agents/trainer.py",
    "src/data/preprocessing.py",
    "src/data/features.py",
    "src/optimization/model_utils.py",
]

cov = coverage.Coverage(source=["src"])

cov.start()
result = pytest.main(["tests/unit", "-v"])
cov.stop()

cov.save()
cov.report(include=files, show_missing=True)

sys.exit(result)
