"""Thin entry point for configured CLAHE enhancement."""

import sys

from petct.cli import legacy_preprocess_main

if __name__ == "__main__":
    legacy_preprocess_main("clahe", sys.argv[1:])
