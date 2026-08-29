"""Thin entry point for paired fixed-size CT/PET slicing."""

import sys

from petct.cli import legacy_preprocess_main

if __name__ == "__main__":
    legacy_preprocess_main("slice", sys.argv[1:])
