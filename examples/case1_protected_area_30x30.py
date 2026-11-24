"""Standalone offline reproducible Case Study 1: Protected Area GBF 30x30 Policy Audit."""

from __future__ import annotations

import json
from pathlib import Path

from geograph2.cases import run_case_protected_area_30x30


def main() -> None:
    out_dir = Path("output_case1")
    result = run_case_protected_area_30x30(output_dir=out_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
