"""Make `src/` importable when the scripts are run directly (no installation needed)."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

# The models fit many small matrices (inner cross-validation): one BLAS thread per process is faster than
# several threads competing for few cores.
from threadpoolctl import threadpool_limits  # noqa: E402

threadpool_limits(1)
