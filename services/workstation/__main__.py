"""Command-line entry point for ``python -m services.workstation``."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from .server import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m services.workstation",
        description="Serve the synthetic CMR Studio workstation web UI.",
    )
    parser.add_argument(
        "--study",
        required=True,
        help="study root containing dicom/<explicit-series-uid>/ and optional cmr.sqlite3",
    )
    parser.add_argument("--host", default="127.0.0.1", help="listen address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8787, help="listen port (default: 8787)")
    parser.add_argument(
        "--allow-authority",
        action="append",
        default=None,
        metavar="HOST:PORT",
        help="browser Host/Origin authority; repeat for additional authorities",
    )
    parser.add_argument("--user", default="workstation", help="physician identity for actions")
    parser.add_argument(
        "--x1-weights-root",
        default=None,
        help=(
            "root containing the two pinned X1 checkpoint artifacts "
            "(overrides CMR_X1_WEIGHTS_ROOT)"
        ),
    )
    parser.add_argument(
        "--frontend-dist",
        default=None,
        help="directory containing the built webapp (serves / + /assets/ same-origin)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(
            args.study,
            host=args.host,
            port=args.port,
            allow_authorities=args.allow_authority,
            user=args.user,
            x1_weights_root=(
                args.x1_weights_root
                if args.x1_weights_root is not None
                else os.environ.get("CMR_X1_WEIGHTS_ROOT")
            ),
            frontend_dist=args.frontend_dist,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"workstation startup failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
