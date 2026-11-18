"""Source-independent helpers for importing real recordings (Italian adapter).

Source-specific parsing lives in ``scripts/import_italian_dataset.py``; this
module holds what any source needs: transcript normalisation for duplicate
detection, audio validation, duplicate grouping and deterministic,
order-independent split assignment. No network access.
"""

from collections import defaultdict
import hashlib
import math
import re
import unicodedata

import numpy as np

HOP = 256
MIN_FRAMES, MAX_FRAMES = 32, 3000      # prepare.py admission limits at hop 256 (~0.34-32 s at 24 kHz)


def dedup_text(text):
    """Normalised transcript used ONLY to detect duplicates (original text is kept)."""
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text)).strip()


def audio_hash(wave):
    """Hash of the decoded samples quantised to 16 bits (container-independent)."""
    pcm = np.clip(np.round(np.asarray(wave, dtype=np.float64) * 32767), -32768, 32767).astype("<i2")
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def check_audio(wave, sample_rate, target_rate=24000):
    """Raise ``ValueError`` for audio prepare.py would reject or must not see."""
    wave = np.asarray(wave)
    if wave.ndim != 1:
        raise ValueError("audio must be mono after conversion")
    if sample_rate != target_rate:
        raise ValueError(f"expected {target_rate} Hz after resampling, got {sample_rate}")
    if not wave.size or not np.isfinite(wave).all():
        raise ValueError("empty or non-finite audio")
    if float(np.abs(wave).max()) > 1.0:
        raise ValueError("audio out of range")
    if float(np.sqrt(np.mean(np.square(wave, dtype=np.float64)))) < 1e-4:
        raise ValueError("audio is effectively silent")
    frames = math.ceil(len(wave) / HOP)
    if not MIN_FRAMES <= frames <= MAX_FRAMES:
        raise ValueError(f"duration outside admission limits ({frames} frames)")


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def connected_groups(records):
    """Group ids so that records sharing a normalised transcript, decoded-audio
    hash or source group (book/session) end in the same group.

    ``records``: dicts with ``id``, ``text``, ``audio_sha256`` and ``group``.
    Returns ``{id: group_key}``; the key is the smallest member id, so it does
    not depend on input order.
    """
    union = _UnionFind()
    owners = defaultdict(list)
    for record in records:
        union.find(record["id"])
        for key in (("text", dedup_text(record["text"])), ("audio", record["audio_sha256"]), ("group", record["group"])):
            owners[key].append(record["id"])
    for members in owners.values():
        for other in members[1:]:
            union.union(members[0], other)
    return {record["id"]: union.find(record["id"]) for record in records}


def assign_splits(records, groups, seed, fractions=(("test", 0.05), ("validation", 0.05))):
    """Deterministic split by usable duration over whole groups.

    Groups are ordered by a seeded hash of their key (input order is
    irrelevant) and dealt to each held-out split until it reaches its share of
    the total duration; everything else is training data. Fails when the groups
    are too coarse to give every split something.
    """
    seconds = defaultdict(float)
    for record in records:
        seconds[groups[record["id"]]] += record["seconds"]
    total = sum(seconds.values())
    order = sorted(seconds, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    assignment, position = {}, 0
    for split, fraction in fractions:
        filled = 0.0
        while position < len(order) and filled < fraction * total:
            assignment[order[position]] = split
            filled += seconds[order[position]]
            position += 1
        if filled == 0:
            raise ValueError(f"not enough independent groups for a {split} split")
    if position >= len(order):
        raise ValueError("no groups left for training; the held-out fractions are too large for this grouping")
    for key in order[position:]:
        assignment[key] = "train"
    return {record["id"]: assignment[groups[record["id"]]] for record in records}


def keep_held_out(records, groups, kept):
    """Splits that keep an earlier import's held-out clips exactly.

    A clip listed as validation/test before keeps that split. A new clip that
    shares a group (book section, transcript or audio) with a held-out clip is
    dropped (``None``): it can join neither the held-out set, which must stay
    identical, nor training, which would leak. Everything else is training.
    A held-out clip that is missing now is an error.
    """
    missing = sorted(set(kept) - {r["id"] for r in records})
    if missing:
        raise ValueError(f"held-out clips of the earlier import are missing now: {missing[:5]}")
    held = {}
    for record in records:
        if record["id"] in kept:
            group = groups[record["id"]]
            if held.setdefault(group, kept[record["id"]]) != kept[record["id"]]:
                raise ValueError("a group now joins validation and test clips; the earlier split cannot be kept")
    return {r["id"]: kept.get(r["id"], None if groups[r["id"]] in held else "train") for r in records}


def split_report(records, splits, sample_rate=24000):
    report = {}
    for split in ("train", "validation", "test"):
        chosen = [r["seconds"] for r in records if splits[r["id"]] == split]
        report[split] = {"clips": len(chosen), "hours": round(sum(chosen) / 3600, 3),
                         "seconds_quantiles": {str(q): round(float(np.percentile(chosen, q)), 2) for q in (0, 5, 50, 95, 100)} if chosen else None}
    return report
