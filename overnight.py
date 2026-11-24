"""Deprecated compatibility entrypoint; use sft_train.py or setup.sh --run sft."""

import sys

import sft_train

if __name__ == "__main__":
    print(
        "DEPRECATED: use ./setup.sh --run sft (or sft_train.py).",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(sft_train.main())
else:
    # Preserve imported function identity and patching behavior for existing callers.
    sys.modules[__name__] = sft_train
