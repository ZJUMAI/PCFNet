"""Legacy thin entry point for the CNN/tree two-stage workflow."""

import sys

from petct.cli import legacy_main

if __name__ == "__main__":
    legacy_main("two-stage", sys.argv[1:])
