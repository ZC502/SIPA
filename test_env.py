"""
SIPA Environment Sanity Check
Run before executing audits.
"""

import sys

print("🔎 SIPA Environment Check\n")

try:
    import numpy
    print(f"numpy OK ({numpy.__version__})")
except ImportError:
    print("numpy NOT installed")
    sys.exit(1)

try:
    import pandas
    print(f"pandas OK ({pandas.__version__})")
except ImportError:
    print("pandas NOT installed")
    sys.exit(1)

try:
    import matplotlib
    print(f"matplotlib OK ({matplotlib.__version__})")
except ImportError:
    print("matplotlib NOT installed")
    sys.exit(1)

print("\n✅ Environment ready for SIPA.")
