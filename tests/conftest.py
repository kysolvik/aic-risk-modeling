import json
import os

import pytest

CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs")


@pytest.fixture
def repo_config():
    """Load configs/<name>.json; skip when absent (configs/ is gitignored, so CI lacks it)."""
    def load(name):
        path = os.path.join(CONFIGS, f"{name}.json")
        if not os.path.exists(path):
            pytest.skip(f"configs/{name}.json not present")
        with open(path) as f:
            return json.load(f)
    return load
