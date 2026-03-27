from __future__ import annotations

import hashlib
import re
import string
import unicodedata
from collections import defaultdict
from pathlib import Path

import mmh3


def exact_line_deduplication(
    input_files: list[Path],
    output_directory: Path,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    line_counts: dict[bytes, int] = {}
    for path in input_files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                key = hashlib.md5(raw_line.rstrip("\n").encode("utf-8", errors="replace")).digest()
                line_counts[key] = line_counts.get(key, 0) + 1

    for path in input_files:
        out_path = output_directory / Path(path).name
        with (
            open(path, encoding="utf-8", errors="replace") as f_in,
            open(out_path, "w", encoding="utf-8") as f_out,
        ):
            for raw_line in f_in:
                key = hashlib.md5(raw_line.rstrip("\n").encode("utf-8", errors="replace")).digest()
                if line_counts[key] == 1:
                    f_out.write(raw_line)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_ngrams(text: str, n: int) -> set[str]:
    words = text.split()
    if len(words) < n:
        return {text} if text else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _minhash_signature(ngram_set: set[str], num_hashes: int) -> list[int]:
    sig = [0xFFFFFFFF] * num_hashes
    for ngram in ngram_set:
        encoded = ngram.encode("utf-8")
        for seed in range(num_hashes):
            h = mmh3.hash(encoded, seed=seed, signed=False)
            if h < sig[seed]:
                sig[seed] = h
    return sig


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def minhash_deduplication(
    input_files: list[Path],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: Path,
) -> None:
    assert num_hashes % num_bands == 0, "num_hashes must be divisible by num_bands"
    rows_per_band = num_hashes // num_bands

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    n_docs = len(input_files)

    raw_docs: list[str] = []
    ngram_sets: list[set[str]] = []
    for path in input_files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        raw_docs.append(text)
        ngram_sets.append(_word_ngrams(_normalize(text), ngrams))

    signatures: list[list[int]] = [
        _minhash_signature(ng, num_hashes) for ng in ngram_sets
    ]

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for doc_idx, sig in enumerate(signatures):
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band_bytes = b"".join(v.to_bytes(4, "big") for v in sig[start:end])
            band_hash = mmh3.hash128(band_bytes, seed=band_idx)
            buckets[(band_idx, band_hash)].append(doc_idx)

    uf = _UnionFind(n_docs)
    candidate_pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                candidate_pairs.add((min(a, b), max(a, b)))

    for a, b in candidate_pairs:
        if _jaccard(ngram_sets[a], ngram_sets[b]) >= jaccard_threshold:
            uf.union(a, b)

    clusters: dict[int, list[int]] = defaultdict(list)
    for doc_idx in range(n_docs):
        clusters[uf.find(doc_idx)].append(doc_idx)

    to_keep: set[int] = {min(members) for members in clusters.values()}

    for doc_idx in sorted(to_keep):
        out_path = output_directory / Path(input_files[doc_idx]).name
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(raw_docs[doc_idx])
