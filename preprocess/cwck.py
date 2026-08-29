"""Thin entry point for CT window width/level normalization."""

import sys

from petct.cli import legacy_preprocess_main

if __name__ == "__main__":
    legacy_preprocess_main("window", sys.argv[1:])
