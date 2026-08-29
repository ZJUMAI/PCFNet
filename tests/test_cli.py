import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["train", "--help"],
        ["hpo", "--help"],
        ["crossval", "--help"],
        ["two-stage", "--help"],
        ["preprocess", "--help"],
        ["evaluate", "--help"],
    ],
)
def test_cli_help(arguments: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "petct", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
