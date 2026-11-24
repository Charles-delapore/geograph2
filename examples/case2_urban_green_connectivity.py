"""Standalone offline reproducible Case Study 2: Urban Green Space Connectivity."""

from __future__ import annotations

import json
from pathlib import Path

from geograph2.cases import run_case_urban_green_connectivity


def main() -> None:
    out_dir = Path("output_case2")
    result = run_case_urban_green_connectivity(output_dir=out_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
