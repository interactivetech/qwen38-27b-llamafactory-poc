"""Export only allowlisted evaluation aggregates; transcripts always remain local."""

import argparse
from pathlib import Path

from terminalbench import read, write
from terminalbench_tracking import metrics


def export(run_dir, destination):
    summary = read(Path(run_dir) / "summary.json")
    if not summary:
        raise ValueError("Evaluation summary is missing")
    result = dict(
        status=summary["status"],
        metrics=metrics(summary),
        models={
            model: {
                key: summary["models"][model][key]
                for key in ("solved", "scored", "planned")
            }
            for model in ("base", "sft")
        },
        paired={
            key: summary["paired"][key] for key in ("base_wins", "sft_wins", "ties")
        },
        limitations=summary["limitations"],
        resource_note=summary["resource_note"],
    )
    write(Path(destination) / "terminalbench21.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    export(args.run_dir, args.output_dir)
