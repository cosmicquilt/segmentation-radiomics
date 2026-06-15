"""config-driven entry point for the segmentation->radiomics pipeline

    python -m seg_radiomics.cli smoke
    python -m seg_radiomics.cli run --config configs/default.yaml

synthetic run exercises every stage end-to-end (no download) the learned segmenter
(monai) and pyradiomics are opt-in see the readme build plan
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger("seg_radiomics.cli")


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _cmd_smoke(args: argparse.Namespace) -> int:
    from .pipeline import format_results, run_synthetic_pipeline

    results = run_synthetic_pipeline({"seed": 0, "data": {"n": 24, "shape": [32, 48, 48]}})
    print(format_results(results))
    print("\nsmoke OK")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import format_results, run_synthetic_pipeline

    cfg = _load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    results = run_synthetic_pipeline(cfg)
    print(format_results(results))

    out_dir = Path(cfg.get("output", {}).get("dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / 'results.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seg_radiomics", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_smoke = sub.add_parser("smoke", help="quick synthetic end-to-end check")
    p_smoke.set_defaults(func=_cmd_smoke)

    p_run = sub.add_parser("run", help="run the synthetic pipeline from a config")
    p_run.add_argument("--config", default=None, help="path to a YAML config")
    p_run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
