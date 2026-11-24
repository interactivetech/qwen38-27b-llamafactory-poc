"""Isolated Harbor worker, cancellable by the evaluation supervisor."""

import asyncio
import os
import signal
import sys
from pathlib import Path


async def run(config_path):
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial

    config = TrialConfig.model_validate_json(Path(config_path).read_text())
    trial = await Trial.create(config)
    task = asyncio.create_task(trial.run())
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await task
    except asyncio.CancelledError:
        pass  # Harbor finalizes and saves interrupted verifier/agent work.


def main():
    # Deliberately do not load dotenv or forward API credentials to Apptainer.
    for key in list(os.environ):
        if any(
            word in key.upper() for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")
        ):
            os.environ.pop(key, None)
    asyncio.run(run(sys.argv[1]))


if __name__ == "__main__":
    main()
