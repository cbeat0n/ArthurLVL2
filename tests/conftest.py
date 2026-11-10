### conftest.py lets the tests import Arthur's modules no matter where pytest is started from.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
