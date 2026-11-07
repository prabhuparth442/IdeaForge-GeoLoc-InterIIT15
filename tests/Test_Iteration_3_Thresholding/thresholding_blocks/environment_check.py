"""Is the notebook running in the right Python, and are the needed packages and files there? Returns answers; the notebook prints them."""
import importlib
import sys
from pathlib import Path


def package_versions(names):
    """{package name: version text, or None when it cannot be imported}."""
    found = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            found[name] = str(getattr(module, "__version__", "installed"))
        except Exception:
            found[name] = None
    return found


def missing_files(paths):
    """The paths (as text) that do not exist."""
    return [str(path) for path in paths if not Path(path).exists()]


def python_in_use():
    """The path of the Python running this notebook (it should contain .venv-geoloc)."""
    return sys.executable


def file_report(entries):
    """entries: list of (what it is, path). Returns a list of (found yes/no, what it is, path) so the notebook can print ok / MISSING lines."""
    return [(Path(path).exists(), what, str(path)) for what, path in entries]
