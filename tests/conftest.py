"""
Pytest Configuration and Environment Setup.

Purpose:
  Configures the test execution runtime and module import search paths.

Role in Project:
  - Dynamically registers the project root directory into sys.path.
  - Ensures clean, absolute imports across all test suites without requiring
    editable package installation (`pip install -e .`).
  - Serves as the central location for shared pytest fixtures across the test suite.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
