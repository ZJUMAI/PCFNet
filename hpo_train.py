"""Legacy thin entry point for in-process Optuna optimization."""

import sys

from petct.cli import legacy_main

if __name__ == "__main__":
    legacy_main("hpo", sys.argv[1:])
