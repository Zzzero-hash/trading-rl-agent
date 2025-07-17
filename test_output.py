#!/usr/bin/env python3

import sys
import io
from unittest.mock import patch

# Add the src directory to the path
sys.path.insert(0, 'src')

# Import the console module directly
sys.path.insert(0, 'src/trading_rl_agent')
from console import print_metrics_table

# Test data
results = [
    {
        "strategy": "momentum",
        "total_return": 0.15,
        "sharpe_ratio": 1.25
    }
]

print("Testing print_metrics_table with single result:")
print("=" * 50)

with patch('sys.stdout', new=io.StringIO()) as fake_out:
    print_metrics_table(results)
    output = fake_out.getvalue()
    print("Actual output:")
    print(repr(output))
    print("\nFormatted output:")
    print(output)