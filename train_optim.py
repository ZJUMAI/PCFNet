"""Train the final model from an HPO-generated best configuration."""

import sys

from petct.cli import legacy_main

if __name__ == "__main__":
    legacy_main("train", sys.argv[1:])
