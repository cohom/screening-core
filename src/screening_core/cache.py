"""Append-only JSONL cache for seen item IDs.

ListingCache tracks which items have been processed in previous cycles.
Items are identified by (org, id) pairs and stored per-source.

The cache accepts any object with .id, .org, .title, .url attributes
(duck-typed — no concrete schema import required).
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class ListingCache:
    """append-only JSONL wrapper for per-source item ID tracking."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, org: str) -> Path:
        return self.cache_dir / f"{org}.jsonl"

    def known_ids(self, org: str) -> set[str]:
        path = self._path(org)
        if not path.exists():
            return set()
        ids: set[str] = set()
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids.add(json.loads(line)["id"])
        return ids

    def has_entries(self, org: str) -> bool:
        """Return True if at least one item was cached for this source in a prior cycle."""
        path = self._path(org)
        if not path.exists():
            return False
        return any(line.strip() for line in path.open(encoding="utf-8"))

    def diff(self, org: str, listings: Iterable[Any]) -> list[Any]:
        """Return only the items whose id is not yet in the cache."""
        known = self.known_ids(org)
        return [l for l in listings if l.id not in known]

    def add(self, listings: Iterable[Any]) -> None:
        """Append new items to the cache, grouped by org. No duplicate check within one call."""
        by_org: dict[str, list[Any]] = {}
        for l in listings:
            by_org.setdefault(l.org, []).append(l)
        for org, items in by_org.items():
            with self._path(org).open("a", encoding="utf-8") as f:
                for l in items:
                    record = {"id": l.id, "title": l.title, "url": l.url}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
