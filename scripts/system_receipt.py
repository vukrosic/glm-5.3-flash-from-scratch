#!/usr/bin/env python3
"""Capture the exact runtime and model configuration used for the experiment."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm53_flash import GLM53FlashFromScratch, ModelConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    config = ModelConfig()
    model = GLM53FlashFromScratch(config)
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    name, driver, memory_mib = [part.strip() for part in query.split(",")]
    payload = {
        "schema_version": "1.0",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": {"name": name, "driver": driver, "memory_mib": int(memory_mib)},
        "model_config": config.to_dict(),
        "parameter_counts": model.parameter_counts(),
        "attention_pattern": [
            "sparse" if (index + 1) % 4 == 0 else "linear"
            for index in range(config.layers)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
