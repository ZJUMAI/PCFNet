"""Legacy thin entry point for a single PET/CT training run."""

import sys

from petct.cli import legacy_main

if __name__ == "__main__":
    legacy_main("train", sys.argv[1:])
