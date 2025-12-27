#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from production_monitor import main

if __name__ == "__main__":
    main()
