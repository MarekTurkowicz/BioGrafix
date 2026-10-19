"""Entry point: ``python main.py [image ...]``."""

import sys

from biografix.app import run

if __name__ == "__main__":
    sys.exit(run())
