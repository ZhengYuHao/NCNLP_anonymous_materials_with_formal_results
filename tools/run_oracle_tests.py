"""Run the recovered Oracle unit tests without any model-service calls."""
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "code" / "experiments" / "benchmark_v2"
sys.path.insert(0, str(BENCH))
suite = unittest.defaultTestLoader.discover(str(BENCH), pattern="test_n8n_h1_oracle_runtime.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
