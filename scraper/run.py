"""CLI entry point: python -m scraper.run --source bbc --limit 20"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from pipeline.ingest import run_ingest
from scraper.registry import list_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape, clean, chunk, and store news articles.")
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Source name (repeatable). Defaults to all configured sources.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum articles per source (default: 50).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scrape and process without writing to MongoDB.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = args.sources or list_sources()

    result = asyncio.run(
        run_ingest(
            sources=sources,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    )

    print(json.dumps(result, indent=2))
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    sys.exit(main())
