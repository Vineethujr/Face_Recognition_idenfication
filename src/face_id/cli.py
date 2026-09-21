"""Command-line interface (optional; the GUI is the main way to start)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from face_id.config import COSINE_THRESHOLD, SCORE_MARGIN
from face_id.evaluate import run_folder_evaluation, run_synthetic_evaluation
from face_id.models import ensure_models
from face_id.pipeline import FaceIdentificationSystem


def _print_identify(result) -> None:
    decision = result.decision
    payload = {
        "identity": decision.identity,
        "score": round(decision.score, 4),
        "is_unknown": decision.is_unknown,
        "reason": decision.reason,
        "faces_detected": len(result.detections),
        "top_candidates": [
            {"identity": c.identity, "score": round(c.score, 4)}
            for c in decision.candidates[:5]
        ],
    }
    print(json.dumps(payload, indent=2))


def cmd_download(_args: argparse.Namespace) -> int:
    yunet, sface = ensure_models()
    print(f"YuNet: {yunet}")
    print(f"SFace: {sface}")
    return 0


def cmd_enroll(args: argparse.Namespace) -> int:
    system = FaceIdentificationSystem(threshold=args.threshold, margin=args.margin)
    count = system.enroll_image(args.name, args.image, face_index=args.face)
    print(f"Enrolled '{args.name}' ({count} template(s) stored)")
    return 0


def cmd_identify(args: argparse.Namespace) -> int:
    system = FaceIdentificationSystem(threshold=args.threshold, margin=args.margin)
    result = system.identify_image(args.image, face_index=args.face)
    _print_identify(result)
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    system = FaceIdentificationSystem()
    identities = system.gallery.identities()
    if not identities:
        print("Gallery is empty.")
        return 0
    for name in identities:
        n = system.gallery.template_count(name)
        print(f"{name}\t{n} template(s)")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    if args.dataset:
        report = run_folder_evaluation(
            Path(args.dataset),
            threshold=args.threshold,
            margin=args.margin,
        )
    else:
        report = run_synthetic_evaluation(
            threshold=args.threshold,
            margin=args.margin,
            seed=args.seed,
        )
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Face recognition identification system"
    )
    parser.add_argument("--threshold", type=float, default=COSINE_THRESHOLD)
    parser.add_argument("--margin", type=float, default=SCORE_MARGIN)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download-models")
    p_dl.set_defaults(func=cmd_download)

    p_en = sub.add_parser("enroll")
    p_en.add_argument("--name", required=True)
    p_en.add_argument("--image", required=True)
    p_en.add_argument("--face", type=int, default=0)
    p_en.set_defaults(func=cmd_enroll)

    p_id = sub.add_parser("identify")
    p_id.add_argument("--image", required=True)
    p_id.add_argument("--face", type=int, default=0)
    p_id.set_defaults(func=cmd_identify)

    p_ls = sub.add_parser("list")
    p_ls.set_defaults(func=cmd_list)

    p_ev = sub.add_parser("evaluate")
    p_ev.add_argument("--dataset")
    p_ev.add_argument("--seed", type=int, default=7)
    p_ev.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(code)
