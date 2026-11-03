"""Byte-level BPE with the course conventions, plus a Unigram segmentation reference.

BPE: GPT-2 style regex pre-tokenization, special tokens as hard boundaries,
ties broken toward the lexicographically greater byte pair, incremental pair
counts during training, merge-rank encoding and streaming encoding.

Teaching reference for chapter 2; coursework restrictions are explained in chapter 28, so write your own A1 tokenizer rather than adapting this one.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import total_ordering
import codecs
import heapq
import math
import os
import regex

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _special_tokens(tokens):
    tokens = list(dict.fromkeys(tokens))
    if any(not isinstance(token, str) or not token for token in tokens):
        raise ValueError('special tokens must be nonempty strings')
    return tokens


def split_special(text, special_tokens):
    """Split text into (segment, is_special) pairs. Longer specials win overlaps."""
    special_tokens = _special_tokens(special_tokens)
    if not special_tokens:
        return [(text, False)] if text else []
    ordered = sorted(special_tokens, key=len, reverse=True)
    pattern = "(" + "|".join(regex.escape(token) for token in ordered) + ")"
    parts = regex.split(pattern, text)
    # regex.split with one capturing group alternates text, special, text, ...
    return [(part, index % 2 == 1) for index, part in enumerate(parts) if part]


def special_safe_cut(buffer, special_tokens):
    """Longest prefix of a streaming buffer whose special-token split more input cannot change.

    Hold back from the earliest suffix that is a proper prefix of some special token, and never
    cut inside a special-token match: with overlapping specials (e.g. "<|eot|>" and "|>x"),
    a complete earlier match must not be split just because its tail could start another one.
    """
    if not special_tokens:
        return len(buffer)
    hold = max((k for token in special_tokens for k in range(1, len(token))
                if buffer.endswith(token[:k])), default=0)
    cut = len(buffer) - hold
    ordered = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(regex.escape(token) for token in ordered)
    for match in regex.finditer(pattern, buffer):
        if match.start() >= cut:
            break
        if match.end() > cut:
            return match.start()
    return cut


def count_pretokens(text, special_tokens=(), pattern=GPT2_PATTERN):
    """Frequency of each pre-token as a tuple of single-byte IDs.

    Special tokens split the text but never contribute to the counts.
    """
    compiled = regex.compile(pattern)
    counts = Counter()
    for segment, is_special in split_special(text, special_tokens):
        if is_special:
            continue
        for match in compiled.finditer(segment):
            counts[tuple(match.group().encode("utf-8"))] += 1
    return counts


def chunk_boundaries(path, chunks, split_token):
    """Find byte offsets at a separator that the caller has checked cannot overlap specials."""
    if not isinstance(chunks, int) or chunks < 1 or not split_token:
        raise ValueError("positive chunk count and nonempty separator required")
    size = os.path.getsize(path)
    guesses = [size * i // chunks for i in range(chunks + 1)]
    with open(path, "rb") as f:
        for i in range(1, chunks):
            f.seek(guesses[i])
            while True:
                start = f.tell()
                block_size = max(4096, len(split_token))
                block = f.read(block_size)
                found = block.find(split_token)
                if found >= 0:
                    guesses[i] = start + found
                    break
                if len(block) < block_size:
                    guesses[i] = size
                    break
                # Retain the suffix: a separator may straddle two read blocks.
                f.seek(start + len(block) - len(split_token) + 1)
    return sorted(set(guesses))


def _count_chunk(args):
    path, start, end, special_tokens, pattern = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return count_pretokens(text, special_tokens, pattern)


def count_pretokens_file(path, special_tokens=("<|endoftext|>",), workers=4, pattern=GPT2_PATTERN):
    """Parallel pre-token counting; chunks are aligned to the first special token.

    Without an independent separator the file is one chunk. A separator contained in a longer
    special, or overlapped by an earlier match, is not a safe split despite matching its bytes.
    """
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("positive worker count required")
    special_tokens = _special_tokens(special_tokens)
    separator = special_tokens[0] if special_tokens else None
    overlap = separator is not None and any(
        (token != separator and separator in token)
        or any(token.endswith(separator[:k]) for k in range(1, min(len(token), len(separator))))
        for token in special_tokens
    )
    if separator and not overlap:
        bounds = chunk_boundaries(path, workers, special_tokens[0].encode("utf-8"))
    else:
        bounds = [0, os.path.getsize(path)]
    jobs = [(path, a, b, tuple(special_tokens), pattern) for a, b in zip(bounds, bounds[1:])]
    total = Counter()
    if workers == 1 or len(jobs) == 1:
        for partial in map(_count_chunk, jobs):
            total.update(partial)
        return total
    with ProcessPoolExecutor(workers) as executor:
        for partial in executor.map(_count_chunk, jobs):
            total.update(partial)
    return total


@total_ordering
class _Descending:
    """Reverses byte-string order so a min-heap pops the lexicographically greatest pair."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return self.value == other.value


def train_bpe_from_counts(pretoken_counts, vocab_size, special_tokens=()):
    """Train merges from {byte-id tuple: count}. Returns (vocab, merges).

    IDs 0-255 are raw bytes, then special tokens, then merges in creation order.
    After a merge only words containing that pair are rescanned; a lazy max-heap
    finds the next pair (stale heap entries are skipped when popped).
    """
    specials = _special_tokens(special_tokens)
    if type(vocab_size) is not int or vocab_size < 256 + len(specials):
        raise ValueError("vocab_size must cover 256 bytes plus all special tokens")
    if any(not isinstance(count, int) or count < 1 or not word
           or any(type(byte) is not int or not 0 <= byte <= 255 for byte in word)
           for word, count in pretoken_counts.items()):
        raise ValueError('nonempty byte-ID tuples and positive integer pretoken counts required')
    vocab = {i: bytes([i]) for i in range(256)}
    for token in specials:
        vocab[len(vocab)] = token.encode("utf-8")
    words = [list(word) for word in pretoken_counts]
    frequency = list(pretoken_counts.values())
    pair_counts = Counter()
    where = defaultdict(set)
    for index, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += frequency[index]
            where[pair].add(index)
    heap = []

    def push(pair):
        # Highest count first; ties go to the lexicographically greater (bytes, bytes) pair.
        a, b = vocab[pair[0]], vocab[pair[1]]
        heapq.heappush(heap, (-pair_counts[pair], _Descending(a), _Descending(b), pair))

    for pair in pair_counts:
        push(pair)
    merges = []
    while len(vocab) < vocab_size and heap:
        negative, _, _, best = heapq.heappop(heap)
        if pair_counts.get(best, 0) != -negative or negative == 0:
            continue
        new_id = len(vocab)
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        merges.append((vocab[best[0]], vocab[best[1]]))
        changed = set()
        for index in list(where[best]):
            word, count = words[index], frequency[index]
            for pair in zip(word, word[1:]):
                pair_counts[pair] -= count
                where[pair].discard(index)
                changed.add(pair)
            word = merge_pair(word, best, new_id)
            words[index] = word
            for pair in zip(word, word[1:]):
                pair_counts[pair] += count
                where[pair].add(index)
                changed.add(pair)
        for pair in changed:
            if pair_counts[pair] > 0:
                push(pair)
            else:
                del pair_counts[pair]
    return vocab, merges


def train_bpe(text, vocab_size, special_tokens=(), pattern=GPT2_PATTERN):
    return train_bpe_from_counts(count_pretokens(text, special_tokens, pattern), vocab_size, special_tokens)


def merge_pair(ids, pair, replacement):
    """Left-to-right non-overlapping replacement of one adjacent pair."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(replacement)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    """Encode with merge ranks inside each pre-token; decode bytes with U+FFFD for invalid UTF-8.

    Ordinary text and special tokens use separate ID maps: a special token whose bytes are also an
    ordinary token (a one-character special, say) gets its own ID, so allow_special=False text can
    never produce it. The r"\\S+" pattern (the handout's stylized example) drops whitespace, so
    decode(encode(x)) == x holds only for the default GPT-2 pattern.
    """

    def __init__(self, vocab, merges, special_tokens=None, pattern=GPT2_PATTERN):
        self.vocab = dict(vocab)
        by_bytes = defaultdict(list)
        for i in sorted(self.vocab):
            by_bytes[self.vocab[i]].append(i)
        self.specials = _special_tokens(special_tokens or [])
        merge_outputs = {a + b for a, b in merges}
        self.special_ids = {}
        for token in self.specials:
            raw = token.encode("utf-8")
            candidates = by_bytes.get(raw, [])
            if len(raw) == 1:
                # IDs 0-255 are raw bytes and train_bpe registers specials after them: a later duplicate is the special.
                spare = candidates[1:]
            elif raw in merge_outputs:
                # train_bpe registers specials before any merge output, so only an earlier duplicate is spare.
                spare = candidates[:-1]
            else:
                spare = candidates
            if spare:
                self.special_ids[token] = spare[0]
            else:
                new_id = max(self.vocab) + 1
                self.vocab[new_id] = raw
                self.special_ids[token] = new_id
        reserved = set(self.special_ids.values())
        # Ordinary text never maps to a special ID; among duplicates the last ID wins, as before.
        self.ids = {}
        for raw, ids in by_bytes.items():
            ordinary = [i for i in ids if i not in reserved]
            if ordinary:
                self.ids[raw] = ordinary[-1]
        for token, i in self.special_ids.items():
            self.ids.setdefault(token.encode("utf-8"), i)
        self.ranks = {(self.ids[a], self.ids[b]): rank for rank, (a, b) in enumerate(merges)}
        self.merged = {(self.ids[a], self.ids[b]): self.ids[a + b] for a, b in merges}
        self.pattern = regex.compile(pattern)
        self.streaming_supported = pattern in (GPT2_PATTERN, r"\S+")
        self.cache = {}

    def _encode_pretoken(self, piece):
        if piece in self.cache:
            return self.cache[piece]
        ids = [self.ids[bytes([b])] for b in piece.encode("utf-8")]
        while len(ids) > 1:
            # Apply the earliest-learned merge that is present, as training did.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, math.inf))
            if pair not in self.ranks:
                break
            ids = merge_pair(ids, pair, self.merged[pair])
        self.cache[piece] = ids
        return ids

    def encode(self, text, *, allow_special=True):
        out = []
        for segment, is_special in split_special(text, self.specials if allow_special else []):
            if is_special:
                out.append(self.special_ids[segment])
            else:
                for match in self.pattern.finditer(segment):
                    out.extend(self._encode_pretoken(match.group()))
        return out

    def _encode_settled(self, text, *, allow_special=True):
        """Encode pre-tokens whose extent more input cannot change; return the unsettled tail.

        Near the end of the buffer the regex has not seen every character it would inspect:
        a greedy run or the whitespace lookahead needs the next character, and the
        contraction alternative ('re, 'll, 've) needs two characters after its start
        ("we'r" splits as we, ', r until the "e" arrives). A match is final only if it
        ends at least two characters before the buffer end. A trailing special token
        also waits: it may be the start of a longer special token.
        """
        ids, pieces = [], split_special(text, self.specials if allow_special else [])
        for index, (segment, is_special) in enumerate(pieces):
            final = index == len(pieces) - 1
            if is_special:
                if final:
                    return ids, segment
                ids.append(self.special_ids[segment])
                continue
            for match in self.pattern.finditer(segment):
                if final and match.end() > len(segment) - 2:
                    return ids, segment[match.start():]
                ids.extend(self._encode_pretoken(match.group()))
        return ids, ""

    def encode_iterable(self, chunks, *, allow_special=True):
        """Stream GPT-2/S+ pre-tokens; custom regexes buffer input for correctness.

        The unresolved pre-token and encoding cache can grow without a fixed bound.
        Set allow_special=False for untrusted content containing literal control markers.
        """
        if not self.streaming_supported:
            yield from self.encode("".join(chunks), allow_special=allow_special)
            return
        buffer = ""
        specials = self.specials if allow_special else []
        for chunk in chunks:
            buffer += chunk
            cut = special_safe_cut(buffer, specials)
            ids, tail = self._encode_settled(buffer[:cut], allow_special=allow_special)
            yield from ids
            buffer = tail + buffer[cut:]
        yield from self.encode(buffer, allow_special=allow_special)

    def decode_bytes(self, ids):
        return b"".join(self.vocab[i] for i in ids)

    def decode(self, ids):
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def decode_iterable(self, ids):
        """Incrementally decode token IDs without breaking UTF-8 across token boundaries.

        Concatenating the yielded strings equals decode(ids), including replacement
        of malformed bytes. Flush only when the ID stream ends: an incomplete code
        point may be completed by the next token. Output chunks are not graphemes.
        """
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        for token in ids:
            text = decoder.decode(self.vocab[token], final=False)
            if text:
                yield text
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail


def unigram_segment(text, piece_probabilities):
    """Character-piece Viterbi and log-marginal DP; probabilities are supplied.

    This does not train a Unigram vocabulary or perform SentencePiece normalization.
    """
    if not piece_probabilities or any(not s or not math.isfinite(p) or p <= 0
                                      for s, p in piece_probabilities.items()):
        raise ValueError("nonempty pieces and finite positive probabilities required")
    if not math.isclose(sum(piece_probabilities.values()), 1.0, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("piece probabilities must sum to one")
    best = [-math.inf] * (len(text) + 1)
    total = best.copy()
    back = [None] * (len(text) + 1)
    best[0] = total[0] = 0.0
    for end in range(1, len(text) + 1):
        paths = []
        for piece, p in piece_probabilities.items():
            start = end - len(piece)
            if start < 0 or text[start:end] != piece or total[start] == -math.inf:
                continue
            score = math.log(p)
            paths.append(total[start] + score)
            if best[start] + score > best[end]:
                best[end] = best[start] + score
                back[end] = (start, piece)
        if paths:
            m = max(paths)
            total[end] = m + math.log(sum(math.exp(x - m) for x in paths))
    if total[-1] == -math.inf:
        raise ValueError("text is not encodable by the supplied pieces")
    pieces, end = [], len(text)
    while end:
        end, piece = back[end]
        pieces.append(piece)
    return dict(pieces=pieces[::-1], best_log_probability=best[-1], log_probability=total[-1])


if __name__ == "__main__":
    stylized = "low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest"
    _, merges = train_bpe(stylized, 256 + 1 + 6, ["<|endoftext|>"], pattern=r"\S+")
    print("stylized whitespace example, first 6 merges:", [(a + b).decode() for a, b in merges])
    corpus = ("语言模型预测下一个词。<|endoftext|>语言模型也能写代码。<|endoftext|>"
              "The model writes code: x += 1\n<|endoftext|>banana banana bandana")
    vocab, merges = train_bpe(corpus, 300, ["<|endoftext|>"])
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    for text in ["语言模型写代码<|endoftext|>banana", "未见过的🙂字符  and   spaces\n\n"]:
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text
        shown = [tokenizer.vocab[i].decode("utf-8", errors="backslashreplace") for i in ids]
        print(f"{len(text.encode())} bytes -> {len(ids)} tokens:", shown)
