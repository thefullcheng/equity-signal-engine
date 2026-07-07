"""Minimal offline stand-in for pytest.

Why this exists: the sandbox this repo is developed in has no network and no
pytest. This runner lets the SAME test files execute there. On your machine,
ignore this file and just run ``pytest`` -- the tests are ordinary pytest
tests. Supports only what our suite needs: @pytest.fixture (no scopes),
plain assert-based tests, and a no-op ``caplog``.

Usage: python tools/dev_test_runner.py tests/test_universe.py [...]
"""

import importlib.util
import inspect
import sys
import traceback
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---- fake pytest module ----------------------------------------------------
if "pytest" not in sys.modules:
    fake = types.ModuleType("pytest")

    def fixture(func=None, **kwargs):
        def deco(f):
            f._is_fixture = True
            return f
        return deco(func) if func is not None else deco

    class _Raises:
        def __init__(self, exc):
            self.exc = exc
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise AssertionError(f"expected {self.exc}")
            return issubclass(exc_type, self.exc)

    fake.fixture = fixture
    fake.raises = _Raises
    sys.modules["pytest"] = fake

class _CapLog:
    records: list = []
    text = ""

def run_file(path: Path) -> tuple[int, int]:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fixtures = {
        name: obj for name, obj in vars(mod).items()
        if callable(obj) and getattr(obj, "_is_fixture", False)
    }
    passed = failed = 0
    for name, obj in sorted(vars(mod).items()):
        if not (name.startswith("test_") and callable(obj)):
            continue
        kwargs = {}
        try:
            for p in inspect.signature(obj).parameters:
                if p in fixtures:
                    kwargs[p] = fixtures[p]()
                elif p == "caplog":
                    kwargs[p] = _CapLog()
            obj(**kwargs)
            print(f"  PASS {name}")
            passed += 1
        except Exception:
            print(f"  FAIL {name}")
            traceback.print_exc()
            failed += 1
    return passed, failed


if __name__ == "__main__":
    files = [Path(a) for a in sys.argv[1:]] or sorted(
        (REPO_ROOT / "tests").glob("test_*.py")
    )
    tp = tf = 0
    for f in files:
        print(f"== {f}")
        p, x = run_file(f)
        tp += p
        tf += x
    print(f"\n{tp} passed, {tf} failed")
    sys.exit(1 if tf else 0)
