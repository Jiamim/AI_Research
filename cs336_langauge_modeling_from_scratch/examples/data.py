"""Data-pipeline primitives: rule filters, contact redaction, exact and near-duplicate removal.

The rule set follows the Gopher subset discussed in the course data lectures. These are
teaching references with interfaces of their own (in-memory documents, different return
values and placeholders); the 2026 A4 asks you to write your own implementations, and the
coursework restrictions are explained in chapter 28.
Hashes are stable across processes (SHA-256 / keyed BLAKE2), unlike hash().
"""
from collections import Counter, defaultdict
import hashlib
import re
import unicodedata
import numpy as np
import torch

WORD = re.compile(r"\S+")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")
# North American style numbers with optional country code and common separators. A heuristic:
# any bare 10-digit run (order numbers, Unix timestamps such as 1700000000) is also redacted.
PHONE = re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)")
# Not part of a longer dotted number on either side, but a sentence-final period is allowed.
IPV4 = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\.?\d)")


def rule_quality_check(text):
    """Return the names of the failed rules (empty list = keep). Rules: 50-100,000 words;
    mean word length 3-10; at most 30% of lines ending in '...'; at least 80% of words
    contain a letter."""
    words = WORD.findall(text)
    reasons = []
    if not 50 <= len(words) <= 100_000:
        reasons.append("word_count")
    if words and not 3 <= sum(map(len, words)) / len(words) <= 10:
        reasons.append("mean_word_length")
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and sum(line.rstrip().endswith(("...", "…")) for line in lines) / len(lines) > 0.3:
        reasons.append("ellipsis_lines")
    if words and sum(any(c.isalpha() for c in w) for w in words) / len(words) < 0.8:
        reasons.append("alphabetic_words")
    return reasons


def redact_contacts(text):
    """Replace emails, phone numbers and IPv4 addresses with <EMAIL>, <PHONE>, <IP>; return counts."""
    counts = {}
    for name, pattern, token in [("email", EMAIL, "<EMAIL>"), ("phone", PHONE, "<PHONE>"), ("ip", IPV4, "<IP>")]:
        text, counts[name] = pattern.subn(token, text)
    return text, counts


def line_hash(line):
    return hashlib.sha256(line.encode("utf-8")).digest()[:16]


def drop_repeated_lines(documents):
    """Two passes over in-memory documents: count hashed lines, then drop every line seen more than once.

    Blank lines are paragraph structure, not boilerplate, so they are never counted or dropped.
    """
    counts = Counter(line_hash(line) for document in documents for line in document.splitlines() if line.strip())
    return ["\n".join(line for line in document.splitlines() if not line.strip() or counts[line_hash(line)] == 1)
            for document in documents]


def normalize(text):
    """Lowercase, NFD, strip accents and punctuation, collapse whitespace (dedup-only view)."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_dedup(documents):
    seen, kept = set(), []
    for document in documents:
        digest = hashlib.sha256(normalize(document).encode()).digest()
        if digest not in seen:
            seen.add(digest)
            kept.append(document)
    return kept


def shingles(text, n=3, unit="char"):
    """Character n-grams by default; unit='word' uses whitespace word n-grams."""
    if type(n) is not int or n < 1 or unit not in ('char', 'word'):
        raise ValueError("n must be positive")
    items = normalize(text)
    if unit == "word":
        items = items.split()
    if not items:
        return set()
    if len(items) < n:
        return {" ".join(items) if unit == "word" else items}
    joiner = " " if unit == "word" else ""
    return {joiner.join(items[i:i + n]) for i in range(len(items) - n + 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


def minhash(items, count=128):
    if not items or type(count) is not int or count < 1:
        raise ValueError("filter empty documents before MinHash")
    return tuple(min(int.from_bytes(hashlib.blake2b(item.encode(), digest_size=8,
                     key=seed.to_bytes(8, "little")).digest(), "little") for item in items)
                 for seed in range(count))


def lsh_candidate_probability(similarity, bands, rows):
    if not 0 <= similarity <= 1 or any(type(x) is not int or x < 1 for x in (bands, rows)):
        raise ValueError('similarity in [0,1] and positive integer bands/rows required')
    return 1 - (1 - similarity ** rows) ** bands


def near_duplicate_clusters(documents, bands=16, rows=8, threshold=0.8, n=5, unit="word"):
    """MinHash signatures -> LSH band buckets -> verify true Jaccard -> clusters of indices.

    Clusters are connected components of verified pairs (union-find), so near-duplicate
    chains can merge. Returns the index of the first document of each cluster.
    """
    lsh_candidate_probability(threshold, bands, rows)
    sets = [shingles(d, n, unit) for d in documents]
    signatures = [minhash(s, bands * rows) if s else None for s in sets]
    parent = list(range(len(documents)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    candidates = set()
    for band in range(bands):
        buckets = defaultdict(list)
        for index, signature in enumerate(signatures):
            if signature is not None:
                buckets[signature[band * rows:(band + 1) * rows]].append(index)
        for members in buckets.values():
            candidates.update((a, b) for i, a in enumerate(members) for b in members[i + 1:])
    verified = [(a, b) for a, b in candidates if jaccard(sets[a], sets[b]) >= threshold]
    for a, b in verified:
        ra, rb = find(a), find(b)
        # The smaller index becomes the root, so each cluster keeps its earliest document.
        parent[max(ra, rb)] = min(ra, rb)
    keep = [i for i in range(len(documents)) if find(i) == i]
    return dict(keep=keep, candidate_pairs=len(candidates), verified_pairs=len(verified))


def mmap_next_token_batch(path, starts, context):
    """Read a batch from a 1-D integer .npy file, preserving its header/dtype."""
    if type(context) is not int or context < 1:
        raise ValueError("context must be positive")
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 1 or not np.issubdtype(data.dtype, np.integer):
        raise ValueError("expected a one-dimensional integer token array")
    starts = list(starts)
    if not starts or any(not isinstance(s, (int, np.integer)) or s < 0 or s + context >= len(data) for s in starts):
        raise ValueError("each window must contain context+1 tokens")
    # Copy only chosen windows: do not expose read-only mapped storage to mutation.
    windows = np.stack([data[s:s + context + 1] for s in starts]).astype(np.int64, copy=True)
    if np.any(windows < 0):
        raise ValueError("token IDs must be nonnegative")
    full = torch.from_numpy(windows)
    return full[:, :-1], full[:, 1:]


if __name__ == "__main__":
    page = ("Contact admin@example.com or call (650) 555-0199. Server 192.168.0.12 is down... " * 3).strip()
    print("contacts:", redact_contacts(page))
    story = "Once upon a time a small robot learned to read books with its friends in the library. " * 6
    print("failed rules, story:", rule_quality_check(story), " menu:", rule_quality_check("Home | About | Login ..."))
    article = ("The city council met on Tuesday to discuss the new library budget. Members debated whether "
               "to extend opening hours, hire two more librarians and replace the old heating system before "
               "winter. A final vote is expected next month after a public hearing at the community center.")
    docs = [article, article + " Share this story on social media.",
            "A completely different article about databases, indexes and query planning in practice."]
    result = near_duplicate_clusters(docs, threshold=0.7)
    print("true Jaccard of the near-duplicate pair:", round(jaccard(shingles(docs[0], 5, "word"),
                                                                   shingles(docs[1], 5, "word")), 3))
    print("MinHash kept", len(result["keep"]), "of", len(docs), "verified pairs", result["verified_pairs"])
    for s in (0.5, 0.7, 0.8, 0.9):
        print(f"P(candidate | Jaccard={s}) with 16 bands x 8 rows = {lsh_candidate_probability(s, 16, 8):.3f}")
