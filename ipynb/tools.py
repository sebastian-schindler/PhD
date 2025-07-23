"""
Backwards compatibility module for existing notebooks.

This module allows existing code using 'from tools import *' to continue working
while the actual package has been renamed to 'bastiastro'.

For new code, prefer: from bastiastro import *
"""

# Re-export everything from bastiastro for backwards compatibility
from bastiastro import *
