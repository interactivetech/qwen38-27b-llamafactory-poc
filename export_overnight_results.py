"""Deprecated exporter; preserves the historical overnight_* output filenames."""

import sys

from export_sft_results import export as _export
from export_sft_results import main as _main


def export(run, output, include_responses=False):
    return _export(run, output, include_responses, prefix="overnight")


def main():
    print(
        "DEPRECATED: use export_sft_results.py; historical filename prefix retained.",
        file=sys.stderr,
    )
    return _main(prefix="overnight")


if __name__ == "__main__":
    main()
