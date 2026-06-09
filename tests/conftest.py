import sys
from unittest.mock import MagicMock

# Mock paramiko
sys.modules["paramiko"] = MagicMock()
