"""Legacy thin entry point for registered-model cross-validation."""

import sys

from petct.cli import legacy_main

if __name__ == "__main__":
    legacy_main("crossval", sys.argv[1:])
