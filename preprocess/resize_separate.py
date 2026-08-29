"""Thin entry point for validated paired volume resampling."""

import sys

from petct.cli import legacy_preprocess_main

if __name__ == "__main__":
    legacy_preprocess_main("resize", sys.argv[1:])
