"""Run synthetic documents against backends explicitly configured by the caller."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from lev.client import LevClient
from lev_examples.request import annotate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file")
    parser.add_argument("--inputs", type=Path, required=True, help="JSON array of document objects")
    parser.add_argument("--mode", choices=("strict", "eager", "speculative"), default="speculative")
    options = parser.parse_args()
    with LevClient.from_env(env_file=options.env_file, mode=options.mode) as client:
        with client.submit(annotate, json.loads(options.inputs.read_text()), ordered=True) as results:
            for result in results:
                print(json.dumps(asdict(result), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
