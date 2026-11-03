# 36｜完整教学代码
以下源码从仓库实际可运行文件自动嵌入，便于在单文件 HTML、PDF 中离线阅读。执行仍建议使用仓库模块命令；带相对 import 的代码需要保留 `examples` 包结构。

| 模块 | 内容 | 主要章节 |
|---|---|---|
| tokenizer | byte BPE（预切分、特殊 token、增量合并、流式编码/UTF-8 解码）、Unigram 切分 | 02 |
| model | RMSNorm、RoPE、GQA、SwiGLU、KV cache、消融开关、top-p 解码 | 03、04 |
| optim | AdamW、余弦与 WSD 日程、全局范数裁剪 | 05 |
| train | token 文件训练、固定验证窗口、发散检测、精确恢复 | 05、20 |
| accounting | 参数、FLOPs、显存、MFU、roofline、decode 延迟 | 08、12 |
| attention | 在线 softmax、softmax 反向恒等式、Triton 行 softmax kernel | 10 |
| systems | 自定义 RMSNorm 反向、张量并行反向、低精度累加 | 09、11 |
| distributed | 双进程 DDP、collective 等价、优化器状态分片 | 11 |
| inference | speculative decoding、分页 KV 与前缀共享 | 12 |
| data | Gopher 式规则、联系方式替换、行级与 MinHash 去重、mmap 读取 | 05、14 |
| scaling | 预算受限的 IsoFLOP 设计与多种子误差 | 13 |
| objectives | SFT 损失、蒸馏、packing 注意力与目标边界 mask | 04、16 |
| posttrain | DPO、SimPO、GRPO 族、GSPO、GAE 终止/截断、熵、小模型 GRPO 循环 | 17、18 |
| evaluation | pass@k、Wilson、逐题/逐簇配对 bootstrap、Elo、算术诊断 | 15、20 |
| extensions | MoE 路由、线性注意力、CLIP 损失 | 06、07、19 |
| research / retrieval / reasoning / compression / interpretability | 第 22–27 章的小实验 | 22–27 |
| architectures | 14 个开源模型的参数公式（GQA、MLA、MoE、MTP、Gated DeltaNet、Lightning Attention）、激活参数、KV 每 token | 29 |
| long_context | RoPE 波长、PI、NTK、YaRN、可学习 sink 注意力、流式缓存 | 30 |
| rl_objectives | DAPO 解耦裁剪与 token 级聚合、CISPO、动态采样、超长惩罚 | 31 |
| agents | 工具调用回合、只在模型段计损失的掩码、语法树计算器、可验证奖励 | 32 |
| synthetic_data | 教师采样、拒绝采样过滤、软标签蒸馏的对照实验 | 33 |
| low_precision | MXFP4（MX Algorithm 1）、FP8 1×128 与 128×128 分块量化 | 34 |
| numerical_tools | DPO/KL、预测区间、QAT/2:4、SAE/Muon、softmax/Delta rule、packing，以及校准/采样/接口反例 | 01–07、10、14–19、23、26–28、30–31、34 |
| mechanisms | 词表分块/并行交叉熵、仿射扫描、MLA 吸收、ESS、验证器基率、潜势塑形与随机舍入 | 07–12、18、25、32、34 |

实验脚本（不嵌入全文，见仓库 `scripts/`）：`run_core_experiments.py`、`run_tinystories.py`、`run_grpo_demo.py`、`run_advanced_experiments.py`、`run_frontier_experiments.py`、`run_review_update_experiments.py`、`run_chapter_review.py`、`run_mechanism_experiments.py`、`run_checks.py`。
## tokenizer.py
[对应文件](../../examples/tokenizer.py)
```python
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

```
## model.py
[对应文件](../../examples/model.py)
```python
"""A deliberately small decoder: RMSNorm, RoPE, SwiGLU, GQA and KV cache.

Config switches cover the course ablations: pre/post/no normalization, RoPE or
no position information (NoPE), SwiGLU or ungated SiLU feed-forward, and tied
input/output embeddings.

Teaching reference: it uses nn.Linear, nn.Embedding and F.silu for brevity, which A1
forbids, and coursework restrictions and this demo's scope are explained in chapter 28.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab: int = 256
    width: int = 64
    layers: int = 2
    heads: int = 4
    kv_heads: int = 2
    hidden: int = 176
    rope_base: float = 10000.0
    norm: str = "pre"          # "pre", "post" or "none"
    position: str = "rope"     # "rope" or "none" (NoPE)
    ffn: str = "swiglu"        # "swiglu" or "silu"
    tie_embeddings: bool = False

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in
               (self.vocab, self.width, self.layers, self.heads, self.kv_heads, self.hidden)):
            raise ValueError("all model dimensions must be positive integers")
        if self.width % self.heads or self.heads % self.kv_heads:
            raise ValueError("width must divide into heads; query heads must divide into KV groups")
        if (self.width // self.heads) % 2 or not math.isfinite(self.rope_base) or self.rope_base <= 0:
            raise ValueError("RoPE needs an even head dimension and a positive base")
        if self.norm not in ("pre", "post", "none") or self.position not in ("rope", "none") \
                or self.ffn not in ("swiglu", "silu"):
            raise ValueError("unknown norm, position or ffn option")


def init_linear(weight):
    """N(0, 2/(d_in+d_out)) truncated at three standard deviations."""
    std = math.sqrt(2 / sum(weight.shape))
    nn.init.trunc_normal_(weight, std=std, a=-3 * std, b=3 * std)


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        # Preserve float64 for numerical gradient tests; accumulate low precision in float32.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        scaled = work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps)
        return (scaled * self.weight.to(work.dtype)).to(x.dtype)


def rope(x, positions, base=10000.0):
    # x: [batch, heads, time, head_dim]; adjacent coordinate pairs.
    d = x.shape[-1]
    if d % 2 or d < 2 or not x.is_floating_point() or positions.ndim != 1 or len(positions) != x.shape[-2] \
            or not math.isfinite(base) or base <= 0:
        raise ValueError("RoPE needs an even head dimension")
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    freq = base ** (-torch.arange(0, d, 2, device=x.device, dtype=work_dtype) / d)
    angles = positions.to(device=x.device, dtype=work_dtype)[:, None] * freq[None, :]
    c, s = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.width % cfg.heads or cfg.heads % cfg.kv_heads:
            raise ValueError("width must be divisible by heads; heads must be divisible by kv_heads")
        self.h, self.hk, self.d = cfg.heads, cfg.kv_heads, cfg.width // cfg.heads
        self.base = cfg.rope_base
        self.use_rope = cfg.position == "rope"
        self.q = nn.Linear(cfg.width, self.h * self.d, bias=False)
        self.k = nn.Linear(cfg.width, self.hk * self.d, bias=False)
        self.v = nn.Linear(cfg.width, self.hk * self.d, bias=False)
        self.out = nn.Linear(self.h * self.d, cfg.width, bias=False)

    def forward(self, x, cache=None, use_cache=False):
        b, t, _ = x.shape
        offset = 0 if cache is None else cache[0].shape[-2]
        pos = torch.arange(offset, offset + t, device=x.device)
        q = self.q(x).view(b, t, self.h, self.d).transpose(1, 2)
        k = self.k(x).view(b, t, self.hk, self.d).transpose(1, 2)
        if self.use_rope:
            q, k = rope(q, pos, self.base), rope(k, pos, self.base)
        v = self.v(x).view(b, t, self.hk, self.d).transpose(1, 2)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=-2)
            v = torch.cat((cache[1], v), dim=-2)
        new_cache = (k, v) if use_cache else None
        # Materialized repeats simplify the reference; production kernels share KV directly.
        kq = k.repeat_interleave(self.h // self.hk, dim=1)
        vq = v.repeat_interleave(self.h // self.hk, dim=1)
        scores = q @ kq.transpose(-1, -2) / math.sqrt(self.d)
        allowed = torch.arange(k.shape[-2], device=x.device)[None, :] <= pos[:, None]
        scores = scores.masked_fill(~allowed, -torch.inf)
        softmax_dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
        p = scores.softmax(dim=-1, dtype=softmax_dtype).to(q.dtype)
        y = (p @ vq).transpose(1, 2).contiguous().view(b, t, self.h * self.d)
        return self.out(y), new_cache


class FeedForward(nn.Module):
    """SwiGLU: down(SiLU(gate x) * up x). Ungated SiLU: down(SiLU(up x))."""
    def __init__(self, cfg):
        super().__init__()
        self.gated = cfg.ffn == "swiglu"
        self.gate = nn.Linear(cfg.width, cfg.hidden, bias=False) if self.gated else None
        self.up = nn.Linear(cfg.width, cfg.hidden, bias=False)
        self.down = nn.Linear(cfg.hidden, cfg.width, bias=False)

    def forward(self, x):
        hidden = F.silu(self.gate(x)) * self.up(x) if self.gated else F.silu(self.up(x))
        return self.down(hidden)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mode = cfg.norm
        make_norm = (lambda: RMSNorm(cfg.width)) if cfg.norm != "none" else nn.Identity
        self.norm1, self.norm2 = make_norm(), make_norm()
        self.attn = Attention(cfg)
        self.ffn = FeedForward(cfg)

    def forward(self, x, cache=None, use_cache=False):
        if self.mode == "post":
            # Original Transformer: normalize after each residual addition.
            a, cache = self.attn(x, cache, use_cache)
            x = self.norm1(x + a)
            return self.norm2(x + self.ffn(x)), cache
        a, cache = self.attn(self.norm1(x), cache, use_cache)
        x = x + a
        return x + self.ffn(self.norm2(x)), cache


class TinyLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab, cfg.width)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        # Post-norm blocks already end in a norm; "none" removes every RMSNorm.
        self.norm = RMSNorm(cfg.width) if cfg.norm == "pre" else nn.Identity()
        self.head = nn.Linear(cfg.width, cfg.vocab, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_linear(module.weight)
        if cfg.tie_embeddings:
            # A shared matrix serves as both lookup and readout, so use the readout scale.
            self.embedding.weight = self.head.weight
        else:
            nn.init.trunc_normal_(self.embedding.weight, std=1.0, a=-3.0, b=3.0)

    def forward(self, ids, caches=None, use_cache=False):
        if ids.ndim != 2 or not ids.shape[0] or not ids.shape[1] or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("nonempty integer token IDs with shape [batch, time] required")
        if caches is not None:
            if len(caches) != len(self.blocks):
                raise ValueError("one KV cache pair per model layer required")
            lengths = []
            for pair in caches:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("each KV cache must contain a key and value tensor")
                key, value = pair
                expected = (len(ids), self.cfg.kv_heads)
                if key.ndim != 4 or value.shape != key.shape or key.shape[:2] != expected \
                        or key.shape[-1] != self.cfg.width // self.cfg.heads or key.device != ids.device \
                        or value.device != ids.device or key.dtype != value.dtype:
                    raise ValueError("KV cache shape, dtype or device does not match the model input")
                lengths.append(key.shape[-2])
            if len(set(lengths)) != 1:
                raise ValueError("all layers must cache the same prefix length")
        x = self.embedding(ids)
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, None if caches is None else caches[i], use_cache)
            new_caches.append(cache)
        logits = self.head(self.norm(x))
        return (logits, new_caches) if use_cache else logits


def sample_next(logits, temperature=1.0, top_p=1.0, generator=None):
    """Temperature then nucleus filtering; temperature 0 means greedy argmax."""
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("nonnegative finite temperature and 0 < top_p <= 1 required")
    if logits.ndim != 2 or min(logits.shape) < 1 or not logits.is_floating_point() or torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(logits).any(-1).all():
        raise ValueError("each logit row needs a finite candidate, without NaN or positive infinity")
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    use_double = logits.dtype == torch.float64 or temperature < torch.finfo(torch.float32).tiny
    work = logits.double() if use_double else logits.float()
    # Center before dividing by a tiny temperature; negative infinity masks are valid.
    probs = ((work - work.max(-1, keepdim=True).values) / temperature).softmax(-1)
    if top_p < 1:
        sorted_probs, order = probs.sort(-1, descending=True)
        # Keep the smallest prefix whose mass reaches top_p (always keep the top token).
        keep = sorted_probs.cumsum(-1) - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=generator)


@torch.no_grad()
def generate(model, prompt, new_tokens=40, temperature=1.0, top_p=1.0, eos_id=None, generator=None):
    """Sample with a KV cache in eval mode; the model's previous train/eval mode is restored."""
    if prompt.ndim != 2 or not prompt.shape[0] or not prompt.shape[1] or type(new_tokens) is not int or new_tokens < 0:
        raise ValueError("nonempty prompt and nonnegative count required")
    if eos_id is not None and (type(eos_id) is not int or not 0 <= eos_id < model.cfg.vocab):
        raise ValueError("EOS ID must be in the model vocabulary")
    was_training = model.training
    model.eval()
    try:
        return _generate(model, prompt, new_tokens, temperature, top_p, eos_id, generator)
    finally:
        model.train(was_training)


def _generate(model, prompt, new_tokens, temperature, top_p, eos_id, generator):
    ids, caches = prompt.clone(), None
    chunk = ids
    # Rows that already emitted EOS keep receiving EOS, so each row can be cut at its first EOS.
    done = torch.zeros(len(ids), dtype=torch.bool, device=ids.device)
    for _ in range(new_tokens):
        logits, caches = model(chunk, caches=caches, use_cache=True)
        token = sample_next(logits[:, -1], temperature, top_p, generator)
        if eos_id is not None:
            token = torch.where(done[:, None], torch.full_like(token, eos_id), token)
            done |= token[:, 0] == eos_id
        ids = torch.cat((ids, token), dim=1)
        if eos_id is not None and bool(done.all()):
            break
        chunk = token
    return ids

```
## optim.py
[对应文件](../../examples/optim.py)
```python
"""Dense-gradient AdamW, cosine schedule and global-norm clipping, written out step by step.

Teaching reference: A1 asks you to implement these yourself (coursework restrictions are explained in chapter 28).
"""
import math
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self._validate(lr, betas, eps, weight_decay)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        for group in self.param_groups:
            self._validate(group['lr'], group['betas'], group['eps'], group['weight_decay'])

    @staticmethod
    def _validate(lr, betas, eps, weight_decay):
        if not all(math.isfinite(x) for x in (lr, eps, weight_decay)) or lr < 0 or eps <= 0 or weight_decay < 0 \
                or len(betas) != 2 or not all(math.isfinite(b) and 0 <= b < 1 for b in betas):
            raise ValueError("finite learning rate/weight decay >= 0, epsilon > 0 and two betas in [0,1) required")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("closures are outside this teaching optimizer")
        # Check every group before mutating any weights or optimizer state.
        for group in self.param_groups:
            self._validate(group['lr'], group['betas'], group['eps'], group['weight_decay'])
            if any(p.grad is not None and p.grad.is_sparse for p in group['params']):
                raise ValueError('dense gradients only')
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state.update(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state["step"] += 1
                t, m, v = state["step"], state["m"], state["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                mhat = m / (1 - b1 ** t)
                vhat = v / (1 - b2 ** t)
                p.addcdiv_(mhat, vhat.sqrt().add_(group["eps"]), value=-group["lr"])


def cosine_lr(t, max_lr, min_lr, warmup, cosine_end):
    """Linear warmup over [0, warmup), cosine decay to min_lr at cosine_end, then constant."""
    if not all(math.isfinite(x) for x in (t, max_lr, min_lr, warmup, cosine_end)) or t < 0 \
            or not 0 <= warmup <= cosine_end or not 0 <= min_lr <= max_lr:
        raise ValueError("require 0 <= warmup <= cosine_end and min_lr <= max_lr")
    if t < warmup:
        return max_lr * t / warmup
    if t >= cosine_end:
        return min_lr
    progress = (t - warmup) / (cosine_end - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def wsd_lr(t, max_lr, min_lr, warmup, decay_start, end):
    """Warmup-stable-decay: hold the peak, then decay linearly to min_lr at `end`."""
    if not all(math.isfinite(x) for x in (t, max_lr, min_lr, warmup, decay_start, end)) or t < 0 \
            or not 0 <= min_lr <= max_lr or not 0 <= warmup <= decay_start <= end:
        raise ValueError("require 0 <= warmup <= decay_start <= end")
    if t < warmup:
        return max_lr * t / warmup
    if t >= end:
        return min_lr
    if t < decay_start:
        return max_lr
    progress = (t - decay_start) / (end - decay_start)
    return max_lr + progress * (min_lr - max_lr)


@torch.no_grad()
def clip_grad_norm(parameters, max_norm, eps=1e-6):
    """Scale all gradients together when their joint L2 norm exceeds max_norm.

    Accepts an iterable of parameters or a single tensor, like torch.nn.utils.clip_grad_norm_.
    Returns the norm before clipping, which is the number worth logging.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if not math.isfinite(max_norm) or max_norm < 0 or not math.isfinite(eps) or eps < 0:
        raise ValueError("finite nonnegative max_norm and eps required")
    grads = [g for g in grads if g.numel()]
    if not grads:
        return torch.tensor(0.0)
    if any(g.is_sparse for g in grads) or len({g.device for g in grads}) > 1:
        raise ValueError("dense gradients on one device required; reduce sharded norms separately")
    if any(not torch.isfinite(g).all() for g in grads):
        raise ValueError("nonfinite gradients: inspect the failed step before clipping")
    dtype = torch.float64 if any(g.dtype == torch.float64 for g in grads) else torch.float32
    # Scale before squaring: a finite gradient may overflow a naive sum of squares.
    scale = torch.stack([g.abs().max().to(dtype) for g in grads]).max()
    total = scale if scale == 0 else scale * torch.sqrt(sum((g.to(dtype) / scale).square().sum() for g in grads))
    if not torch.isfinite(total):
        raise ValueError("gradient norm exceeds the accumulation dtype range")
    coefficient = (max_norm / (total + eps)).clamp(max=1) if total > 0 else total.new_tensor(1.)
    for g in grads:
        g.mul_(coefficient)
    return total

```
## train.py
[对应文件](../../examples/train.py)
```python
"""Train the guide's decoder on a token stream, with exact interruption and resume.

Two data sources share one loop:
* default: an original synthetic modular-addition byte corpus (seconds on CPU);
* --train/--valid: 1-D integer token arrays saved as .npy (e.g. TinyStories IDs).

Teaching reference; coursework restrictions and this demo's scope are explained in chapter 28.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
import time
import tempfile
import numpy as np
import torch
from torch.nn import functional as F
from .model import Config, TinyLM, generate
from .optim import AdamW, clip_grad_norm, cosine_lr


def corpus(seed, count):
    rng = random.Random(seed)
    rows = [f"{a}+{b}={(a+b)%10}\n" for a, b in ((rng.randrange(10), rng.randrange(10)) for _ in range(count))]
    return torch.tensor(list("".join(rows).encode()), dtype=torch.long)


def load_tokens(path):
    """Memory-map a 1-D integer .npy file; only sampled windows are read."""
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 1 or not np.issubdtype(data.dtype, np.integer):
        raise ValueError("expected a one-dimensional integer token array")
    return data


def windows(data, starts, context):
    """Inputs and next-token targets for windows of context+1 tokens."""
    starts = list(starts)
    if type(context) is not int or context < 1 or not starts or any(
        not isinstance(s, (int, np.integer)) or s < 0 or s + context >= len(data) for s in starts
    ):
        raise ValueError("positive context and starts whose windows contain context+1 tokens required")
    if isinstance(data, torch.Tensor):
        full = torch.stack([data[s:s + context + 1] for s in starts])
    else:
        full = torch.from_numpy(np.stack([data[s:s + context + 1] for s in starts]).astype(np.int64))
    return full[:, :-1], full[:, 1:]


def batch(data, batch_size, context, generator):
    starts = torch.randint(len(data) - context, (batch_size,), generator=generator).tolist()
    return windows(data, starts, context)


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    embedding = model.embedding.weight.numel()
    head = 0 if model.cfg.tie_embeddings else model.head.weight.numel()
    return dict(total=total, non_embedding=total - embedding - head)


def data_fingerprint(data):
    """Hash logical token content in bounded chunks, including dtype and length."""
    array = data.numpy() if isinstance(data, torch.Tensor) else data
    digest = hashlib.sha256(f"{array.dtype}:{array.shape}".encode())
    for start in range(0, len(array), 1 << 20):
        digest.update(np.asarray(array[start:start + (1 << 20)]).tobytes())
    return digest.hexdigest()


def save_checkpoint(checkpoint, path):
    """A failed serialization must leave the previous resumable checkpoint intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    # A regular file inside a private staging directory gets the process's umask
    # naturally; reading it via os.umask(0) would alter concurrent file creation.
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='.checkpoint-') as directory:
        temporary = Path(directory)/'checkpoint.pt'
        with temporary.open('wb') as file:
            torch.save(checkpoint, file)
            file.flush()
            os.fsync(file.fileno())
        if mode is not None:
            temporary.chmod(mode)
        temporary.replace(path)


def run(steps=120, out="outputs/tiny", seed=7, stop_after=None, resume=None, train=None, valid=None,
        config=None, lr=2e-3, min_lr_ratio=0.1, warmup=None, batch_size=16, context=32,
        weight_decay=0.1, betas=(0.9, 0.95), max_grad_norm=1.0, eval_every=20, eval_windows=384,
        threads=2, divergence_loss=None, verbose=True):
    """steps is the planned schedule horizon, even when stopping early.

    Resume is exact on CPU: same data, configuration, optimizer state, global
    and batch RNG, and the same planned schedule horizon.
    """
    if type(steps) is not int or steps < 1 or (stop_after is not None and (type(stop_after) is not int or not 0 < stop_after <= steps)):
        raise ValueError("positive steps required; stop_after must be within the planned run")
    if any(type(x) is not int or x < 1 for x in (batch_size, context, eval_every, eval_windows, threads)) or (train is None) != (valid is None):
        raise ValueError("positive batch/context/evaluation/threads and paired train/valid files required")
    if not math.isfinite(min_lr_ratio) or not 0 <= min_lr_ratio <= 1 or not math.isfinite(max_grad_norm) or max_grad_norm < 0:
        raise ValueError("finite min_lr_ratio in [0,1] and nonnegative max_grad_norm required")
    if divergence_loss is not None and (not math.isfinite(divergence_loss) or divergence_loss <= 0):
        raise ValueError("divergence_loss must be None or finite and positive")
    warmup = min(10, steps - 1) if warmup is None else warmup
    cosine_lr(0, lr, lr * min_lr_ratio, warmup, steps)
    AdamW._validate(lr, betas, 1e-8, weight_decay)
    saved = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if saved:
        if saved.get("total_steps") != steps:
            raise ValueError("resuming must keep the original learning-rate schedule horizon")
        seed = saved["seed"]
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 10)
    if train is None:
        # IID draws with overlapping arithmetic facts: a distribution-learning smoke test.
        train_data, valid_data = corpus(1, 1500), corpus(2, 200)
    else:
        train_data, valid_data = load_tokens(train), load_tokens(valid)
    if min(len(train_data), len(valid_data)) <= context:
        raise ValueError("both datasets need at least context+1 tokens")
    cfg = Config(**(config or {}))
    for name, data in (("train", train_data), ("valid", valid_data)):
        low, high = int(data.min()), int(data.max())
        if low < 0 or high >= cfg.vocab:
            raise ValueError(f"{name} token IDs span [{low}, {high}] but the model vocabulary is {cfg.vocab}")
    if train is None:
        xv, yv = batch(valid_data, 24, context, torch.Generator().manual_seed(19))
    else:
        # Fixed, evenly spaced validation windows: identical for every run and checkpoint.
        count = min(eval_windows, len(valid_data) - context)
        stride = max(1, (len(valid_data) - context - 1) // count)
        xv, yv = windows(valid_data, range(0, stride * count, stride), context)
    model = TinyLM(cfg)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = AdamW([{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
                lr=lr, betas=betas, weight_decay=weight_decay)
    contract = dict(
        lr=lr, min_lr_ratio=min_lr_ratio, warmup=warmup, batch_size=batch_size,
        context=context, weight_decay=weight_decay, betas=list(betas), max_grad_norm=max_grad_norm,
        eval_every=eval_every, eval_windows=eval_windows, threads=threads, divergence_loss=divergence_loss,
        train_sha256=data_fingerprint(train_data), valid_sha256=data_fingerprint(valid_data),
        torch=str(torch.__version__),
        implementation={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ('train.py', 'model.py', 'optim.py')},
    )
    history, start_step, diverged, running = [], 0, False, []
    if saved:
        if saved.get('resume_contract') != contract:
            previous = saved.get('resume_contract', {})
            changed = [key for key in contract if previous.get(key) != contract[key]]
            raise ValueError("exact resume requires identical settings/data/implementation: " + ', '.join(changed))
        if saved.get('diverged', False):
            raise ValueError("cannot exactly resume a diverged checkpoint")
        if saved["config"] != asdict(cfg):
            raise ValueError("checkpoint configuration differs from this experiment")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        g.set_state(saved["batch_rng"])
        history, start_step = saved["history"], saved["step"]
        running.extend(saved["running_losses"])
    end_step = steps if stop_after is None else stop_after
    if end_step <= start_step:
        raise ValueError("resume endpoint must be after the saved step")
    tokens_per_step = batch_size * context
    started = time.perf_counter()

    @torch.no_grad()
    def evaluate(step, grad_norm=None, current_lr=None):
        nonlocal diverged
        model.eval()
        total = 0.0
        for i in range(0, len(xv), 64):
            logits = model(xv[i:i + 64])
            total += F.cross_entropy(logits.reshape(-1, cfg.vocab), yv[i:i + 64].reshape(-1), reduction="sum").item()
        validation_loss = total / yv.numel()
        record = dict(step=step, tokens=step * tokens_per_step,
                      val_loss=validation_loss if math.isfinite(validation_loss) else None)
        if not math.isfinite(validation_loss):
            record['nonfinite_validation_loss'] = True
            record['diverged'] = True
            diverged = True
        if running:
            # Wall time stays out of the history so resumed and continuous runs compare exactly.
            record.update(train_loss=sum(running) / len(running), lr=current_lr, grad_norm=grad_norm)
            running.clear()
        history.append(record)
        if verbose:
            print(json.dumps(record), flush=True)
        model.train()

    if not saved:
        evaluate(0)
    grad_norm = None
    completed = start_step  # Count only optimizer updates that were actually applied.
    for step in range(start_step, end_step):
        if diverged:
            break
        x, y = batch(train_data, batch_size, context, g)
        current_lr = cosine_lr(step, lr, lr * min_lr_ratio, warmup, steps)
        for group in opt.param_groups:
            group["lr"] = current_lr
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).reshape(-1, cfg.vocab), y.reshape(-1))
        # Non-finite loss is divergence at any step; a loss ceiling applies once warmup is over.
        if not math.isfinite(loss.item()) or (divergence_loss and step >= warmup and loss.item() > divergence_loss):
            diverged, completed = True, step
            history.append(dict(step=step, tokens=step * tokens_per_step, diverged=True,
                                train_loss=loss.item() if math.isfinite(loss.item()) else None,
                                nonfinite_loss=not math.isfinite(loss.item())))
            break
        loss.backward()
        # Finite loss with overflowing gradients is divergence too: stop and checkpoint, don't crash.
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            diverged, completed = True, step
            history.append(dict(step=step, tokens=step * tokens_per_step, diverged=True,
                                train_loss=loss.item(), nonfinite_gradients=True))
            break
        try:
            grad_norm = clip_grad_norm(model.parameters(), max_grad_norm).item()
        except ValueError as error:
            if 'norm exceeds' not in str(error):
                raise
            diverged, completed = True, step
            history.append(dict(step=step, tokens=step * tokens_per_step, diverged=True,
                                train_loss=loss.item(), nonfinite_grad_norm=True))
            break
        opt.step()
        completed = step + 1
        running.append(loss.item())
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            evaluate(step + 1, grad_norm, current_lr)
    elapsed = time.perf_counter() - started
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    save_checkpoint(dict(model=model.state_dict(), optimizer=opt.state_dict(), config=asdict(cfg),
                    step=completed, total_steps=steps, seed=seed, history=history, running_losses=running,
                    torch_rng=torch.get_rng_state(), batch_rng=g.get_state(),
                    resume_contract=contract, diverged=diverged), path / "checkpoint.pt")
    result = dict(seed=seed, steps=completed, total_steps=steps, torch=str(torch.__version__), device="cpu",
                  config=asdict(cfg), parameters=parameter_counts(model), history=history,
                  hyperparameters=dict(lr=lr, min_lr_ratio=min_lr_ratio, warmup=warmup, batch_size=batch_size,
                                       context=context, weight_decay=weight_decay, betas=list(betas),
                                       max_grad_norm=max_grad_norm),
                  diverged=diverged, wall_seconds=round(elapsed, 1),
                  tokens_per_second=round((completed - start_step) * tokens_per_step / elapsed, 1),
                  data="synthetic arithmetic bytes" if train is None else str(train))
    (path / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if train is None and verbose and not diverged:
        sample = generate(model, torch.tensor([list(b"2+3=")]), new_tokens=30, temperature=0)
        print(bytes(sample[0].tolist()).decode("utf-8", errors="replace"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--out", default="outputs/tiny")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--train", help="1-D integer .npy token file")
    parser.add_argument("--valid", help="1-D integer .npy token file")
    parser.add_argument("--config", default="{}", help='JSON Config overrides, e.g. {"norm":"post"}')
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if (args.train is None) != (args.valid is None):
        parser.error("--train and --valid must be given together")
    run(args.steps, args.out, stop_after=args.stop_after, resume=args.resume, train=args.train,
        valid=args.valid, config=json.loads(args.config), lr=args.lr, batch_size=args.batch_size,
        context=args.context, threads=args.threads)

```
## accounting.py
[对应文件](../../examples/accounting.py)
```python
"""Closed-form resource accounting for decoder-only Transformers.

Worked on Llama 2 shapes (the 2026 A1 accounting problems use GPT-2 shapes and
are left to the student): parameter and FLOP breakdowns, mixed-precision model
state, activation memory after Korthikanti et al. (2022), training time against
the GPU-hours Llama 2 reports, the course's roofline numbers, and a
bandwidth-bound decode model in the style of the 2026 inference lecture.
Every function returns plain numbers so chapter tables can be regenerated.
"""
from dataclasses import dataclass
import json
import math

# Dense peak throughput (FLOP/s) and HBM bandwidth (bytes/s) quoted in the course materials.
HARDWARE = {
    "A100": dict(bf16=312e12, memory=80e9),
    "H100": dict(bf16=989e12, tf32=495e12, hbm=3.35e12, memory=80e9),
    "B200": dict(bf16=2.5e15, fp32=80e12, memory=192e9),
}


@dataclass(frozen=True)
class Shape:
    vocab: int
    context: int
    layers: int
    d_model: int
    heads: int
    d_ff: int
    kv_heads: int = None
    tied: bool = False

    def __post_init__(self):
        for value in (self.vocab, self.context, self.layers, self.d_model, self.heads, self.d_ff):
            if type(value) is not int or value < 1:
                raise ValueError("positive integer model dimensions required")
        kv = self.heads if self.kv_heads is None else self.kv_heads
        if type(kv) is not int or kv < 1 or self.d_model % self.heads or self.heads % kv:
            raise ValueError("width must divide into heads and query heads into positive KV groups")

    @property
    def d_kv(self):
        return (self.heads if self.kv_heads is None else self.kv_heads) * (self.d_model // self.heads)


def swiglu_width(d_model, multiple=64):
    """8/3 * d_model rounded to the nearest multiple (A1 convention), never below one multiple."""
    if type(d_model) is not int or type(multiple) is not int or min(d_model, multiple) < 1:
        raise ValueError("positive integer width and alignment required")
    return multiple * max(1, round(8 * d_model / 3 / multiple))


def llama2(size, context=4096):
    """Llama 2 shapes (Touvron et al., 2023): SwiGLU, RMSNorm, untied embeddings; 70B uses 8 KV heads."""
    layers, d, heads, d_ff, kv = {"7b": (32, 4096, 32, 11008, None), "13b": (40, 5120, 40, 13824, None),
                                  "70b": (80, 8192, 64, 28672, 8)}[size]
    return Shape(vocab=32000, context=context, layers=layers, d_model=d, heads=heads, d_ff=d_ff, kv_heads=kv)


def parameters(s):
    attention = 2 * s.d_model * s.d_model + 2 * s.d_model * s.d_kv
    per_layer = attention + 3 * s.d_model * s.d_ff + 2 * s.d_model
    embeddings = s.vocab * s.d_model * (1 if s.tied else 2)
    return dict(embeddings=embeddings, per_layer=per_layer, total=embeddings + s.layers * per_layer + s.d_model)


def forward_flops(s, tokens=None):
    """Matmul FLOPs for one forward pass over `tokens` positions (default: full context)."""
    t = s.context if tokens is None else tokens
    if type(t) is not int or t < 0:
        raise ValueError("nonnegative integer forward token count required")
    parts = dict(
        qkvo_projections=s.layers * 2 * t * (2 * s.d_model * s.d_model + 2 * s.d_model * s.d_kv),
        attention_scores=s.layers * 2 * t * t * s.d_model,
        attention_values=s.layers * 2 * t * t * s.d_model,
        feed_forward=s.layers * 2 * t * 3 * s.d_model * s.d_ff,
        lm_head=2 * t * s.d_model * s.vocab,
    )
    parts["total"] = sum(parts.values())
    return parts


def model_state_bytes(s, bytes_per_param=16):
    """AdamW in mixed precision: BF16 weights and gradients, FP32 master weights and two moments."""
    if not math.isfinite(bytes_per_param) or bytes_per_param <= 0:
        raise ValueError("positive finite bytes per parameter required")
    return bytes_per_param * parameters(s)["total"]


def activation_bytes(s, batch, recompute="none"):
    """Stored activations for one step in 16-bit, from Korthikanti et al. (2022), eq. 1:
    s*b*h*(34 + 5*a*s/h) bytes per layer without model parallelism.

    "selective" drops the 5as/h attention term (recomputed, or never stored with FlashAttention);
    "full" counts only saved layer boundaries (2sbh), not backward's live layer or peak memory.
    "selective" is a historical estimate, not an exact FlashAttention allocation model.
    The constants were derived for a GPT block with a
    4h GeLU MLP and dropout, so for SwiGLU blocks treat the result as an order-of-magnitude estimate.
    """
    t, d, a = s.context, s.d_model, s.heads
    if type(batch) is not int or batch < 1 or recompute not in ('none', 'selective', 'full'):
        raise ValueError("positive integer batch and a supported recomputation mode required")
    per_layer = {"none": t * batch * d * (34 + 5 * a * t / d), "selective": 34 * t * batch * d,
                 "full": 2 * t * batch * d}[recompute]
    return s.layers * per_layer


def training_flops(s, tokens):
    """Forward matmul FLOPs x 3 (backward = 2 x forward) over `tokens`, in full-context sequences."""
    if not math.isfinite(tokens) or tokens < 0:
        raise ValueError("finite nonnegative training token count required")
    return 3 * forward_flops(s)["total"] * tokens / s.context


def training_days(s, tokens, peak, mfu):
    if not math.isfinite(peak) or peak <= 0 or not math.isfinite(mfu) or not 0 < mfu <= 1:
        raise ValueError("positive finite peak and MFU in (0,1] required")
    return training_flops(s, tokens) / (peak * mfu) / 86400


def mfu(model_flops_per_step, step_seconds, devices, peak):
    if any(not math.isfinite(x) or x <= 0 for x in (step_seconds, devices, peak)) \
            or not math.isfinite(model_flops_per_step) or model_flops_per_step < 0:
        raise ValueError("finite nonnegative FLOPs and positive time/device/peak values required")
    return model_flops_per_step / (step_seconds * devices * peak)


def ridge_point(device="H100", dtype="bf16"):
    hw = HARDWARE[device]
    if dtype not in hw or "hbm" not in hw:
        raise ValueError(f"no {dtype} peak or HBM bandwidth recorded for {device}; only course-quoted numbers are kept")
    return hw[dtype] / hw["hbm"]


def decode_step(params, kv_bytes_per_token, context, batch, bandwidth, bytes_per_param=2):
    """Lower-bound decode latency: each step reads all weights plus every request's KV cache."""
    if any(not math.isfinite(x) or x < 0 for x in (params, kv_bytes_per_token, context)) \
            or any(not math.isfinite(x) or x <= 0 for x in (batch, bandwidth, bytes_per_param)):
        raise ValueError("nonnegative finite sizes and positive batch/bandwidth/parameter precision required")
    memory = bytes_per_param * params + batch * context * kv_bytes_per_token
    if memory == 0:
        raise ValueError("decode model requires some weight or KV traffic")
    latency = memory / bandwidth
    return dict(memory_gb=memory / 1e9, latency_ms=1e3 * latency, tokens_per_second=batch / latency)


def llama2_13b(kv_heads=40):
    s = Shape(vocab=32000, context=1024, layers=40, d_model=5120, heads=40, d_ff=13824, kv_heads=kv_heads)
    return s, parameters(s)["total"], 2 * 2 * s.layers * s.d_kv  # K and V, BF16


def report():
    seven = llama2("7b")
    out = dict(llama2_7b_parameters=parameters(seven), llama2_7b_bf16_load_gb=2 * parameters(seven)["total"] / 1e9)
    shares = {}
    for name, shape in [("7b", seven), ("13b", llama2("13b")), ("70b", llama2("70b")), ("7b_32k", llama2("7b", 32768))]:
        f = forward_flops(shape)
        shares[name] = {k: round(v / f["total"], 4) for k, v in f.items() if k != "total"} | {"tflops": f["total"] / 1e12}
    out["forward_flop_shares"] = shares
    out["llama2_7b_model_state_gb"] = model_state_bytes(seven) / 1e9
    out["llama2_7b_activations_gb_per_sequence"] = {mode: activation_bytes(seven, 1, mode) / 1e9
                                                   for mode in ("none", "selective", "full")}
    # Llama 2 7B: 2.0T tokens, 184,320 A100-80GB GPU-hours (Touvron et al., 2023, Tables 1-2).
    flops = training_flops(seven, 2.0e12)
    out["llama2_7b_training"] = dict(flops=flops, six_n_d=6 * parameters(seven)["total"] * 2.0e12,
                                     implied_a100_mfu=flops / (184_320 * 3600 * HARDWARE["A100"]["bf16"]))
    out["h100_bf16_ridge_flops_per_byte"] = ridge_point()
    decode = {}
    for kv_heads, batch in [(40, 1), (40, 64), (8, 64), (8, 256), (40, 256)]:
        s, n, kv = llama2_13b(kv_heads)
        decode[f"kv{kv_heads}_b{batch}"] = decode_step(n, kv, s.context, batch, HARDWARE["H100"]["hbm"])
    out["llama2_13b_h100_decode"] = decode
    out["llama2_13b_parameters"] = llama2_13b()[1]
    return out


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))

```
## attention.py
[对应文件](../../examples/attention.py)
```python
"""Attention references for chapter 10: the online-softmax invariant and a first Triton kernel.

* dense_attention: the naive O(T^2)-memory reference.
* online_attention: keys streamed in blocks with the running (m, l, u) state, the
  invariant FlashAttention is built on.
* softmax_backward: the softmax vector-Jacobian product dS = P * (dP - rowsum(P * dP)),
  the identity behind FlashAttention's backward.
* triton_softmax: a row softmax written as one Triton kernel that streams each row in
  column blocks with the same (m, l) recurrence. It teaches the Triton programming
  model and runs on CPU when TRITON_INTERPRET=1 is set before triton is imported.

These are teaching references, not the 2026 A2 FlashAttention-2 deliverables: the tiled
forward, the logsumexp-based backward and the Triton attention kernel are left to the
assignment (coursework restrictions and this demo's scope are explained in chapter 28).
"""
import math
import os
import torch


def dense_attention(q, k, v, causal=True):
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        t_q, t_k = q.shape[-2], k.shape[-2]
        if t_q != t_k:
            raise ValueError("this reference uses square causal attention")
        mask = torch.arange(t_k, device=q.device)[None, :] <= torch.arange(t_q, device=q.device)[:, None]
        s = s.masked_fill(~mask, -torch.inf)
    return s.softmax(-1) @ v


def online_attention(q, k, v, block_size=16, causal=True):
    """All queries at once, keys streamed in blocks: shows the running (m, l, u) state."""
    if type(block_size) is not int or block_size < 1 or q.ndim != 2 or k.ndim != 2 or v.ndim != 2 \
            or len(k) == 0 or len(q) == 0 or len(k) != len(v) or q.shape[1] != k.shape[1]:
        raise ValueError("positive block size and nonempty input required")
    if causal and len(q) != len(k):
        raise ValueError("this reference uses square causal attention")
    m = q.new_full((len(q), 1), -torch.inf)
    denom = q.new_zeros((len(q), 1))
    numer = q.new_zeros((len(q), v.shape[1]))
    for start in range(0, len(k), block_size):
        kb, vb = k[start:start + block_size], v[start:start + block_size]
        scores = q @ kb.T / math.sqrt(q.shape[1])
        if causal:
            mask = torch.arange(start, start + len(kb), device=q.device)[None, :] <= torch.arange(len(q), device=q.device)[:, None]
            scores = scores.masked_fill(~mask, -torch.inf)
        new_m = torch.maximum(m, scores.max(-1, keepdim=True).values)
        alpha = (m - new_m).exp()
        p = (scores - new_m).exp()
        numer = alpha * numer + p @ vb
        denom = alpha * denom + p.sum(-1, keepdim=True)
        m = new_m
    return numer / denom


def softmax_backward(p, grad_p):
    """Gradient w.r.t. the logits of a row softmax, given P and dL/dP (last dimension)."""
    return p * (grad_p - (p * grad_p).sum(-1, keepdim=True))


def _build_softmax_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def softmax_kernel(X, Y, stride_x, stride_y, n_cols, BLOCK: tl.constexpr):
        # One program per row; the row is read in BLOCK-wide pieces, so it need not fit on chip.
        row = tl.program_id(0)
        m = tl.full((1,), float("-inf"), tl.float32)
        l = tl.zeros((1,), tl.float32)
        for start in range(0, n_cols, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            x = tl.load(X + row * stride_x + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
            m_new = tl.maximum(m, tl.max(x, axis=0))
            # A fully masked prefix keeps m = -inf; exp(-inf - -inf) would be NaN, so shift by 0 instead.
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            l = l * tl.exp(m - m_safe) + tl.sum(tl.exp(x - m_safe), axis=0)
            m = m_new
        for start in range(0, n_cols, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            x = tl.load(X + row * stride_x + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
            y = tl.exp(x - m) / l
            tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=cols < n_cols)

    return softmax_kernel


def triton_softmax(x, block=16):
    """Softmax over the last dimension of a 2-D tensor. Set TRITON_INTERPRET=1 to run on CPU."""
    if x.dim() != 2 or min(x.shape) < 1 or x.dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise ValueError("nonempty 2-D FP16/BF16/FP32 tensor required")
    if type(block) is not int or block < 1 or block & (block - 1):
        raise ValueError('Triton tile width must be a positive power of two')
    if torch.isnan(x).any() or torch.isposinf(x).any() or not torch.isfinite(x).any(-1).all():
        raise ValueError('each softmax row needs a finite candidate, without NaN or positive infinity')
    x = x.contiguous()
    y = torch.empty_like(x)
    _build_softmax_kernel()[(x.shape[0],)](x, y, x.stride(0), y.stride(0), x.shape[1], BLOCK=block)
    return y


if __name__ == "__main__":
    torch.manual_seed(1)
    q, k, v = [torch.randn(37, 16, dtype=torch.float64) for _ in range(3)]
    for causal in (False, True):
        error = (online_attention(q, k, v, 16, causal) - dense_attention(q, k, v, causal)).abs().max().item()
        print(f"online attention, causal={causal}: error {error:.2e}")
    s = torch.randn(5, 37, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(5, 37, dtype=torch.float64)
    p = s.softmax(-1)
    (reference,) = torch.autograd.grad((p * upstream).sum(), s)
    print(f"softmax backward identity: error {(softmax_backward(p.detach(), upstream) - reference).abs().max().item():.2e}")
    if os.environ.get("TRITON_INTERPRET") == "1":
        x = torch.randn(7, 37) * 30
        print("triton softmax (interpreter) error:", (triton_softmax(x) - x.softmax(-1)).abs().max().item())

```
## systems.py
[对应文件](../../examples/systems.py)
```python
"""CPU references for a first-order custom backward, virtual tensor parallelism and low-precision accumulation."""
import json
import math
import torch
from torch.autograd.function import once_differentiable


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gain, eps):
        if x.shape[-1] != gain.numel() or gain.ndim != 1 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("matching one-dimensional gain and positive epsilon required")
        dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        work, weight = x.to(dtype), gain.to(dtype)
        inv = torch.rsqrt(work.square().mean(-1, keepdim=True) + eps)
        ctx.save_for_backward(work, weight, inv)
        ctx.x_dtype, ctx.gain_dtype = x.dtype, gain.dtype
        return (work * inv * weight).to(x.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        x, gain, inv = ctx.saved_tensors
        dy = upstream.to(x.dtype)
        weighted = dy * gain
        dx = weighted * inv - x * inv ** 3 * (weighted * x).mean(-1, keepdim=True)
        # gain is shared by every row, so its gradient sums over all leading dimensions.
        dg = (dy * x * inv).reshape(-1, gain.numel()).sum(0)
        return dx.to(ctx.x_dtype), dg.to(ctx.gain_dtype), None


def rmsnorm_custom(x, gain, eps=1e-6):
    return RMSNormFunction.apply(x, gain, eps)


def tensor_parallel_demo(parts=2):
    """Virtual shards: sum output contributions and input-gradient contributions."""
    if parts < 1 or 12 % parts:
        raise ValueError("parts must divide hidden width 12")
    g = torch.Generator().manual_seed(31)  # local generator: the caller's global RNG is untouched
    x = torch.randn(5, 8, dtype=torch.float64, generator=g).requires_grad_()
    w1 = torch.randn(8, 12, dtype=torch.float64, generator=g).requires_grad_()
    w2 = torch.randn(12, 6, dtype=torch.float64, generator=g).requires_grad_()
    reference = (x @ w1).tanh() @ w2
    contributions = [(x @ a).tanh() @ b for a, b in zip(w1.chunk(parts, dim=1), w2.chunk(parts, dim=0))]
    output = sum(contributions)
    upstream = torch.randn(reference.shape, dtype=reference.dtype, generator=g)
    dx_reference = torch.autograd.grad(reference, x, upstream, retain_graph=True)[0]
    dx_shards = [torch.autograd.grad(y, x, upstream, retain_graph=True)[0] for y in contributions]
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(sum(dx_shards), dx_reference)
    return dict(parts=parts, output_max_error=(output - reference).abs().max().item(),
                input_gradient_max_error=(sum(dx_shards) - dx_reference).abs().max().item(),
                scope="virtual CPU shards; no communication-performance claim")


def accumulation_demo(steps=10000, value=1e-3):
    """Summing many small numbers: where the running sum lives decides the error.

    A different setting from the 2026 A2 mixed_precision_accumulation problem, which is
    left to the student. Kahan summation keeps the running sum in BF16 but carries the
    rounding error of every addition in a second BF16 variable.
    """
    results = {}
    for name, acc_dtype, item_dtype in [("fp32 += fp32", torch.float32, torch.float32),
                                        ("bf16 += bf16", torch.bfloat16, torch.bfloat16),
                                        ("fp32 += bf16", torch.float32, torch.bfloat16)]:
        total = torch.tensor(0, dtype=acc_dtype)
        for _ in range(steps):
            total += torch.tensor(value, dtype=item_dtype)
        results[name] = total.item()
    total = torch.tensor(0, dtype=torch.bfloat16)
    compensation = torch.tensor(0, dtype=torch.bfloat16)
    item = torch.tensor(value, dtype=torch.bfloat16)
    for _ in range(steps):
        y = item - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    results["bf16 += bf16, Kahan"] = total.item()
    return dict(exact=steps * value, bf16_item=item.item(), results=results)


if __name__ == "__main__":
    print(json.dumps(dict(tensor_parallel=tensor_parallel_demo(), accumulation=accumulation_demo()), indent=2))

```
## distributed.py
[对应文件](../../examples/distributed.py)
```python
"""Two-process CPU (Gloo) checks of data-parallel semantics. Run: python -m examples.distributed

1. DDP: equal and unequal local batches reproduce the global-batch gradient when
   each rank scales its loss sum by the GLOBAL token count.
2. all-reduce = reduce-scatter + all-gather on a flat gradient buffer.
3. Optimizer-state sharding (simplified ZeRO-1): every rank sees the same batch here,
   so gradients already agree and the all-reduce real ZeRO-1 needs is omitted; each
   rank keeps AdamW state for only the parameters it owns, updates them, then
   broadcasts them; the result equals unsharded AdamW.

Teaching reference; coursework restrictions and this demo's scope are explained in chapter 28.
"""
import os
from pathlib import Path
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from .optim import AdamW


def ddp_check(rank, world_size):
    torch.manual_seed(9)
    model = torch.nn.Linear(3, 2, bias=False)
    reference = torch.nn.Linear(3, 2, bias=False)
    reference.load_state_dict(model.state_dict())
    x, y = torch.randn(8, 3), torch.randn(8, 2)
    for sizes in [(4, 4), (3, 5)]:
        model.zero_grad()
        reference.zero_grad()
        local_x, local_y = x.split(sizes)[rank], y.split(sizes)[rank]
        local_sum = ((model(local_x) - local_y) ** 2).sum()
        count = torch.tensor(float(local_y.numel()))
        dist.all_reduce(count)
        (world_size * local_sum / count).backward()
        for p in model.parameters():
            dist.all_reduce(p.grad)
            p.grad.div_(world_size)
        ((reference(x) - y) ** 2).mean().backward()
        torch.testing.assert_close(model.weight.grad, reference.weight.grad)


def collective_check(rank, world_size):
    torch.manual_seed(rank)
    flat = torch.randn(4 * world_size)
    expected = flat.clone()
    dist.all_reduce(expected)
    shard = torch.empty(4)
    dist.reduce_scatter_tensor(shard, flat.clone())
    gathered = torch.empty_like(flat)
    dist.all_gather_into_tensor(gathered, shard)
    torch.testing.assert_close(gathered, expected)


def balanced_owners(sizes, world_size):
    """Greedy largest-first assignment so each rank holds a similar number of elements."""
    load, owner = [0] * world_size, [0] * len(sizes)
    for i in sorted(range(len(sizes)), key=lambda i: -sizes[i]):
        owner[i] = min(range(world_size), key=load.__getitem__)
        load[owner[i]] += sizes[i]
    return owner


def zero1_check(rank, world_size, steps=3):
    torch.manual_seed(5)
    full = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 4))
    reference = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 4))
    reference.load_state_dict(full.state_dict())
    params = list(full.parameters())
    owner = balanced_owners([p.numel() for p in params], world_size)
    # Optimizer state exists only for tensors this rank owns.
    owned = [p for i, p in enumerate(params) if owner[i] == rank]
    sharded_opt = AdamW(owned, lr=0.05, weight_decay=0.1)
    reference_opt = AdamW(reference.parameters(), lr=0.05, weight_decay=0.1)
    for step in range(steps):
        x, y = torch.randn(10, 6), torch.randn(10, 4)
        for model in (full, reference):
            model.zero_grad()
            ((model(x) - y) ** 2).mean().backward()
        # Every rank saw the same batch here, so gradients already agree; real DDP averages them.
        sharded_opt.step()
        reference_opt.step()
        for i, p in enumerate(params):
            dist.broadcast(p.data, src=owner[i])
    for p, q in zip(full.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q)
    state = sum(t.numel() for s in sharded_opt.state.values() for t in s.values() if torch.is_tensor(t))
    total = sum(t.numel() for s in reference_opt.state.values() for t in s.values() if torch.is_tensor(t))
    return state, total


def worker(rank, world_size, init_file):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(init_file).as_uri(), rank=rank, world_size=world_size)
    try:
        ddp_check(rank, world_size)
        collective_check(rank, world_size)
        local_state, full_state = zero1_check(rank, world_size)
        sizes = [torch.zeros(1) for _ in range(world_size)]
        dist.all_gather(sizes, torch.tensor([float(local_state)]))
        if rank == 0:
            print("DDP: equal and unequal two-process batches match the global-batch gradient reference.")
            print("Collectives: reduce-scatter followed by all-gather equals all-reduce.")
            print(f"ZeRO-1: sharded AdamW matches unsharded; optimizer-state elements per rank "
                  f"{[int(s.item()) for s in sizes]} vs {full_state} unsharded.", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Gloo binds to the address the hostname resolves to; on hosts where that fails it cannot
    # even use 127.0.0.1. Default to the loopback interface unless one was chosen explicitly.
    if sys.platform.startswith("linux"):
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    with tempfile.TemporaryDirectory() as folder:
        mp.spawn(worker, args=(2, str(Path(folder) / "init")), nprocs=2, join=True)

```
## inference.py
[对应文件](../../examples/inference.py)
```python
"""Inference-serving references: speculative sampling, a paged KV allocator, prefix sharing.

* speculative_step: draft k tokens from q, verify with the target p, accept with
  min(1, p/q), resample a rejection from (p - q)+, and add a bonus token when all
  drafts pass. The output distribution equals sampling from p directly.
* PagedKVCache: fixed-size physical blocks + per-request block tables; requests
  that share a prompt prefix share blocks by reference count (copy-on-write
  is unnecessary here because cached blocks are never modified after filling).
"""
import json
import torch


def speculative_step(target_probs, draft_probs, draft_tokens, generator=None):
    """One verify round.

    target_probs: [k+1, V], target distributions along the SAME proposed prefix.
    draft_probs:  [k, V],   actual draft sampling distributions along that prefix.
    Row i is conditioned on the input prefix and draft_tokens[:i]. After the
    first rejection, later target rows are stale and must not be reused. Returns
    accepted drafts plus a correction/bonus token (1..k+1 tokens in total).
    The caller handles EOS, output budgets and rollback of rejected KV entries.
    """
    if target_probs.ndim != 2 or draft_probs.ndim != 2 or draft_tokens.ndim != 1 \
            or target_probs.shape != (len(draft_tokens) + 1, draft_probs.shape[-1]) \
            or len(draft_probs) != len(draft_tokens) or not target_probs.shape[1]:
        raise ValueError("target [k+1,V], draft [k,V] and draft token IDs [k] required")
    if draft_tokens.dtype != torch.long or draft_tokens.device != target_probs.device or draft_probs.device != target_probs.device:
        raise ValueError("int64 draft tokens and all inputs on one device required")
    dtype = torch.float64 if target_probs.dtype == torch.float64 or draft_probs.dtype == torch.float64 else torch.float32
    distributions = []
    for probabilities in (target_probs, draft_probs):
        if not probabilities.is_floating_point() or not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
            raise ValueError("finite nonnegative probability matrices required")
        work = probabilities.to(dtype)
        # For nonnegative normalized probabilities, normal-value rounding errors scale
        # with their total mass (about 1), not vocabulary size. Only subnormal absolute
        # errors add a V-dependent term. Sum in FP64 to keep reduction error separate.
        sums = probabilities.sum(-1, keepdim=True, dtype=torch.float64)
        precision = torch.finfo(probabilities.dtype)
        tolerance = max(1e-6, 0.5 * precision.eps + 0.5 * probabilities.shape[-1] * precision.tiny * precision.eps)
        if not torch.isfinite(sums).all() or torch.any(sums <= 0) or torch.any((sums - 1).abs() > tolerance):
            raise ValueError("each probability row must sum to one")
        distributions.append(work / sums.to(dtype))
    target_probs, draft_probs = distributions
    if torch.any((draft_tokens < 0) | (draft_tokens >= target_probs.shape[1])):
        raise ValueError("draft token outside vocabulary")
    if len(draft_tokens) and torch.any(draft_probs.gather(1, draft_tokens[:, None]) <= 0):
        raise ValueError("draft tokens must have positive probability under the actual draft sampler")
    out = []
    for i, token in enumerate(draft_tokens.tolist()):
        p, q = target_probs[i, token], draft_probs[i, token]
        if torch.rand((), device=target_probs.device, generator=generator) < torch.clamp(p / q, max=1.0):
            out.append(token)
            continue
        residual = (target_probs[i] - draft_probs[i]).clamp_min(0)
        # If p == q up to rounding, a rejection can leave an all-zero residual; sampling the
        # target itself is then the correct (and only well-defined) choice.
        residual = residual if residual.sum() > 0 else target_probs[i]
        out.append(int(torch.multinomial(residual / residual.sum(), 1, generator=generator)))
        return out
    # Every draft accepted: the target's next distribution is already computed.
    out.append(int(torch.multinomial(target_probs[len(draft_tokens)], 1, generator=generator)))
    return out


def speculative_distribution_check(vocab=5, k=3, trials=20_000, seed=0):
    """Empirical first-token distribution of speculative sampling vs the target."""
    g = torch.Generator().manual_seed(seed)
    target = torch.softmax(torch.randn(k + 1, vocab, generator=g) * 1.5, -1)
    draft = torch.softmax(torch.randn(k, vocab, generator=g) * 1.5, -1)
    counts = torch.zeros(vocab)
    accepted = 0
    for _ in range(trials):
        tokens = torch.stack([torch.multinomial(draft[i], 1, generator=g)[0] for i in range(k)])
        result = speculative_step(target, draft, tokens, g)
        counts[result[0]] += 1
        accepted += len(result)
    # Acceptance rate of the first position; with a shared rate alpha over k drafts the expected
    # number of tokens per round is (1 - alpha^(k+1)) / (1 - alpha) (chapter 12.6).
    alpha = torch.minimum(target[0], draft[0]).sum().item()
    return dict(max_abs_error_first_token=float((counts / trials - target[0]).abs().max()),
                mean_tokens_per_round=accepted / trials, first_position_acceptance=alpha)


class PagedKVCache:
    """Logical token positions -> (physical block, offset) via per-request block tables."""

    def __init__(self, num_blocks, block_size):
        if not isinstance(num_blocks, int) or not isinstance(block_size, int) or min(num_blocks, block_size) < 1:
            raise ValueError("positive integer block count and block size required")
        self.block_size = block_size
        self.free = list(range(num_blocks - 1, -1, -1))
        self.refcount = [0] * num_blocks
        self.tables, self.lengths = {}, {}
        self.prefix_index = {}

    def _allocate(self):
        if not self.free:
            raise MemoryError("out of KV blocks: preempt or reject a request")
        block = self.free.pop()
        self.refcount[block] = 1
        return block

    def add_request(self, request, prompt_tokens):
        """Reuse cached full blocks for an identical prompt prefix; allocate the rest."""
        if request in self.tables:
            raise ValueError("request already exists")
        # Plan allocations before changing refcounts: OOM must leave live requests intact.
        full = len(prompt_tokens) // self.block_size
        keys = [tuple(prompt_tokens[:(i + 1) * self.block_size]) for i in range(full)]
        needed = sum(key not in self.prefix_index for key in keys) + bool(len(prompt_tokens) % self.block_size)
        if needed > len(self.free):
            raise MemoryError("out of KV blocks: request admission did not modify the cache")
        table, length = [], 0
        full_blocks = len(prompt_tokens) // self.block_size
        for i in range(full_blocks):
            key = tuple(prompt_tokens[:(i + 1) * self.block_size])
            if key in self.prefix_index:
                block = self.prefix_index[key]
                self.refcount[block] += 1
            else:
                block = self._allocate()
                self.prefix_index[key] = block
            table.append(block)
            length += self.block_size
        self.tables[request], self.lengths[request] = table, length
        for _ in range(len(prompt_tokens) - length):
            self.append(request)

    def append(self, request):
        """Reserve the slot for one new token; returns (block, offset)."""
        table, length = self.tables[request], self.lengths[request]
        if length % self.block_size == 0:
            table.append(self._allocate())
        self.lengths[request] = length + 1
        return table[-1], length % self.block_size

    def release(self, request):
        for block in self.tables.pop(request):
            self.refcount[block] -= 1
            if self.refcount[block] == 0:
                self.free.append(block)
                self.prefix_index = {k: b for k, b in self.prefix_index.items() if b != block}
        del self.lengths[request]

    def used_blocks(self):
        return sum(count > 0 for count in self.refcount)


def paged_cache_demo():
    cache = PagedKVCache(num_blocks=64, block_size=16)
    system_prompt = list(range(48))  # three full blocks shared by every request
    for r in range(4):
        cache.add_request(r, system_prompt + [1000 + r] * 5)
        for _ in range(20):
            cache.append(r)
    shared = cache.used_blocks()
    unshared = sum(-(-cache.lengths[r] // 16) for r in range(4))
    cache.release(0)
    return dict(blocks_with_prefix_sharing=shared, blocks_without_sharing=unshared,
                blocks_after_releasing_one=cache.used_blocks(),
                contiguous_max_length_reservation=4 * (-(-2048 // 16)))


if __name__ == "__main__":
    print(json.dumps(dict(speculative=speculative_distribution_check(), paged_cache=paged_cache_demo()), indent=2))

```
## data.py
[对应文件](../../examples/data.py)
```python
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

```
## scaling.py
[对应文件](../../examples/scaling.py)
```python
"""IsoFLOP scaling-law fitting on a synthetic loss surface, under a query budget.

The loss surface is a known Chinchilla-style function plus noise, so the
fitted allocation can be checked. Three designs are compared:
* dense: many model sizes per budget (what a textbook plot shows);
* budgeted: a few sizes per budget, chosen around the previous budget's
  optimum (what an A3-style fixed query budget allows);
* budgeted_extended: the same, but a budget whose lowest run lands on the grid
  edge gets extra sizes beyond that edge until the minimum is interior.
Each takes the lowest-loss run per budget (the A3 recommendation) and also
fit a parabola in log N (the Chinchilla variant), then fit N_opt = a C^b.
"""
import numpy as np

TRUE = dict(E=1.0, A=2.0, B=1.0, alpha=0.35, beta=0.3)


def _positive_vector(values, name, minimum=1):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < minimum or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{name} must be a finite positive vector with at least {minimum} entries")
    return values


def loss(n, d, rng=None, noise=0.0):
    value = TRUE["E"] + TRUE["A"] * n ** -TRUE["alpha"] + TRUE["B"] * d ** -TRUE["beta"]
    return value + (rng.normal(0, noise, np.shape(n)) if rng is not None and noise else 0)


def optimal_allocation(compute, a=TRUE["A"], b=TRUE["B"], alpha=TRUE["alpha"], beta=TRUE["beta"]):
    # Loss uses normalized N,D units; C=6ND in the SAME units.
    n = ((alpha * a) / (beta * b)) ** (1 / (alpha + beta)) * (compute / 6) ** (beta / (alpha + beta))
    return n, compute / (6 * n)


def isoflop_minimum(sizes, losses, method="lowest"):
    """Lowest observed run, or the vertex of a parabola fitted in log N."""
    sizes, losses = _positive_vector(sizes, 'sizes'), np.asarray(losses, float)
    if losses.shape != sizes.shape or not np.isfinite(losses).all() or np.any(np.diff(sizes) <= 0) \
            or method not in ('lowest', 'parabola'):
        raise ValueError("strictly increasing sizes, matching finite losses and a supported fit method required")
    best = int(np.argmin(losses))
    if method == "lowest" or len(sizes) < 3:
        return float(sizes[best]), best in (0, len(sizes) - 1)
    c2, c1, _ = np.polyfit(np.log(sizes), losses, 2)
    if c2 <= 0:
        return float(sizes[best]), True
    vertex = -c1 / (2 * c2)
    edge = not np.log(sizes).min() <= vertex <= np.log(sizes).max()
    predicted = float(np.exp(vertex)) if vertex < np.log(np.finfo(float).max) else np.inf
    if not np.isfinite(predicted) or predicted <= 0:
        raise ValueError("parabola vertex is outside the representable model-size range")
    return predicted, edge


def fit_power_law(budgets, values):
    budgets, values = _positive_vector(budgets, 'budgets', 2), _positive_vector(values, 'values', 2)
    if budgets.shape != values.shape or np.unique(budgets).size < 2:
        raise ValueError("aligned samples with at least two distinct compute budgets required")
    slope, intercept = np.polyfit(np.log(budgets), np.log(values), 1)
    return float(np.exp(intercept)), float(slope)


def run_design(budgets, sizes_for, rng, noise, method, extend_edges=False, max_extensions=4):
    """Fit N_opt = a C^b over budgets. With extend_edges, a budget whose lowest run sits on the
    grid edge gets one more size beyond that edge (same log spacing) until the minimum is
    interior, or max_extensions is reached; every added run counts against the query budget.
    Record observed grid-edge minima separately from an extrapolated parabola vertex."""
    budgets = _positive_vector(budgets, 'budgets', 2)
    if np.any(np.diff(budgets) <= 0) or not np.isfinite(noise) or noise < 0 or type(max_extensions) is not int or max_extensions < 0:
        raise ValueError("increasing budgets, finite nonnegative noise and extension count required")
    optima, initial_edges, final_edges, fit_edges, queries, spent = [], [], [], [], 0, 0.0
    previous = None
    for c in budgets:
        sizes = _positive_vector(sizes_for(c, previous), 'sizes', 2 if extend_edges else 1)
        if np.any(np.diff(sizes) <= 0):
            raise ValueError("model sizes must be strictly increasing")
        losses = loss(sizes, c / (6 * sizes), rng, noise)
        first_edge = int(np.argmin(losses)) in (0, len(sizes) - 1)
        for _ in range(max_extensions if extend_edges else 0):
            best = int(np.argmin(losses))
            if best not in (0, len(sizes) - 1):
                break
            ratio = sizes[1] / sizes[0]
            new = sizes[0] / ratio if best == 0 else sizes[-1] * ratio
            sizes = np.append(sizes, new)
            losses = np.append(losses, loss(new, c / (6 * new), rng, noise))
            order = np.argsort(sizes)
            sizes, losses = sizes[order], losses[order]
        queries += len(sizes)
        spent += len(sizes) * c
        n_opt, edge = isoflop_minimum(sizes, losses, method)
        optima.append(n_opt)
        initial_edges.append(first_edge)
        final_edges.append(int(np.argmin(losses)) in (0, len(sizes) - 1))
        fit_edges.append(not sizes.min() <= n_opt <= sizes.max())
        previous = n_opt
    a, b = fit_power_law(budgets, optima)
    return dict(optima=optima, edge_minima=initial_edges, final_edge_minima=final_edges,
                fit_outside_grid=fit_edges, queries=queries, compute_spent=spent, coefficient=a, exponent=b)


def experiment(seed=11, noise=0.003, target=1e9):
    if not np.isfinite(target) or target <= 0:
        raise ValueError("finite positive target compute required")
    rng = np.random.default_rng(seed)
    budgets = np.geomspace(1e3, 1e6, 7)
    dense_grid = np.geomspace(2, 20000, 41)
    # Budgeted design: 5 sizes spanning 4x either side of a guess that follows N ~ C^0.5.
    def budgeted(c, previous):
        guess = previous * np.sqrt(c / budgeted.last) if previous else np.sqrt(c / 6)
        budgeted.last = c
        return guess * np.geomspace(0.25, 4, 5)
    budgeted.last = budgets[0]
    true_n, true_d = optimal_allocation(target)
    designs = {}
    variants = [("dense", lambda c, _: dense_grid, False), ("budgeted", budgeted, False),
                ("budgeted_extended", budgeted, True)]
    for name, sizes_for, extend in variants:
        for method in ("lowest", "parabola"):
            budgeted.last = budgets[0]
            result = run_design(budgets, sizes_for, rng, noise, method, extend_edges=extend)
            predicted = result["coefficient"] * target ** result["exponent"]
            result.update(predicted_N=predicted, relative_error_N=predicted / true_n - 1,
                          loss_at_prediction=float(loss(predicted, target / (6 * predicted))),
                          loss_at_optimum=float(loss(true_n, true_d)))
            designs[f"{name}_{method}"] = result
    return dict(synthetic=True, seed=seed, noise=noise, budgets=budgets.tolist(), target_compute=target,
                true_exponent=TRUE["beta"] / (TRUE["alpha"] + TRUE["beta"]), true_N_at_target=true_n,
                designs=designs)


def seed_study(seeds=range(20), **kwargs):
    """Repeat the whole design under fresh noise: one seed's error says little about a method."""
    runs = [experiment(seed=s, **kwargs) for s in seeds]
    if not runs:
        raise ValueError("at least one seed required")
    summary = {}
    for name in runs[0]["designs"]:
        errors = np.array([r["designs"][name]["relative_error_N"] for r in runs])
        exponents = np.array([r["designs"][name]["exponent"] for r in runs])
        queries = [r["designs"][name]["queries"] for r in runs]
        costs = [r["designs"][name]["compute_spent"] for r in runs]
        summary[name] = dict(queries=queries[0] if min(queries) == max(queries) else None,
                             queries_min=min(queries), queries_max=max(queries), queries_mean=float(np.mean(queries)),
                             compute_spent_min=min(costs), compute_spent_max=max(costs), compute_spent_mean=float(np.mean(costs)),
                             median_abs_error_N=float(np.median(np.abs(errors))),
                             worst_abs_error_N=float(np.abs(errors).max()),
                             exponent_mean=float(exponents.mean()), exponent_std=float(exponents.std()),
                             edge_minima_rate=float(np.mean([np.mean(r["designs"][name]["edge_minima"]) for r in runs])),
                             final_edge_minima_rate=float(np.mean([np.mean(r["designs"][name]["final_edge_minima"]) for r in runs])),
                             fit_outside_grid_rate=float(np.mean([np.mean(r["designs"][name]["fit_outside_grid"]) for r in runs])))
    return dict(synthetic=True, seeds=len(runs), true_exponent=runs[0]["true_exponent"], summary=summary)


if __name__ == "__main__":
    study = seed_study()
    print("true exponent", round(study["true_exponent"], 4), "over", study["seeds"], "noise seeds")
    for name, d in study["summary"].items():
        count = str(d['queries']) if d['queries'] is not None else f"{d['queries_min']}–{d['queries_max']}"
        print(f"{name:18s} queries={count:>5s} exponent={d['exponent_mean']:.3f}±{d['exponent_std']:.3f} "
              f"median |N error|={d['median_abs_error_N']:.1%} worst={d['worst_abs_error_N']:.1%}")

```
## objectives.py
[对应文件](../../examples/objectives.py)
```python
"""Explicit loss and packed-attention contracts shared by the teaching examples.

Teaching reference; coursework restrictions and this demo's scope are explained in chapter 28.
"""
import math
import torch
from torch.nn import functional as F


def accumulation_dtype(x):
    return torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype


def sequence_values(*values):
    """Aligned finite floating vectors; do not broadcast one preference pair into many."""
    first = values[0]
    if first.ndim != 1 or not first.numel() or any(
        value.shape != first.shape or value.device != first.device
        or not value.is_floating_point() or not torch.isfinite(value).all() for value in values
    ):
        raise ValueError("matching finite floating [batch] vectors on one device required")
    dtype = torch.float64 if any(value.dtype == torch.float64 for value in values) else torch.float32
    return tuple(value.to(dtype) for value in values)


def response_mask(mask, shape):
    if tuple(mask.shape) != tuple(shape) or mask.ndim != 2:
        raise ValueError("mask must have the same [batch,time] shape as token values")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must be boolean or binary")
    valid = mask.bool()
    if valid.shape[0] == 0 or torch.any(valid.sum(-1) == 0):
        raise ValueError("each included sequence needs at least one valid response token")
    return valid


def token_logps(logits, labels, mask):
    """Already-shifted logits [B,T,V], labels [B,T]; padding labels can be -100.

    Mask logits BEFORE log_softmax: invalid padding is not allowed to create NaNs
    which would later survive multiplication by zero. Valid rows must be finite.
    """
    if logits.ndim != 3 or logits.shape[:2] != labels.shape or not logits.shape[-1] or not logits.is_floating_point():
        raise ValueError("expected aligned [B,T,V] logits and [B,T] labels")
    if labels.dtype != torch.long or logits.device != labels.device or mask.device != logits.device:
        raise ValueError("int64 labels and logits/mask on one device required")
    valid = response_mask(mask, labels.shape)
    work = torch.where(valid[..., None], logits, 0).to(accumulation_dtype(logits))
    if not torch.isfinite(work).all():
        raise ValueError("nonfinite logits at a valid response position")
    safe_labels = labels.masked_fill(~valid, 0)
    if torch.any((safe_labels < 0) | (safe_labels >= logits.shape[-1])):
        raise ValueError("valid labels must be inside the vocabulary")
    logps = F.log_softmax(work, -1).gather(-1, safe_labels[..., None]).squeeze(-1)
    return logps.masked_fill(~valid, 0)


def sft_loss(logits, labels, mask, normalization="token"):
    logps = token_logps(logits, labels, mask)
    valid = mask.bool()
    if normalization == "token":
        return -logps.sum() / valid.sum()
    if normalization == "sequence":
        return -(logps.sum(-1) / valid.sum(-1)).mean()
    raise ValueError("normalization must be token or sequence")


def distillation_loss(student_logits, teacher_logits, mask, temperature=2.0):
    """Token-averaged forward KL on a SHARED vocabulary, scaled by temperature^2."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("matching [B,T,V] logits and positive temperature required")
    if not student_logits.shape[-1] or not student_logits.is_floating_point() or not teacher_logits.is_floating_point() \
            or teacher_logits.device != student_logits.device or mask.device != student_logits.device:
        raise ValueError("nonempty floating logits and mask on one device required")
    valid = response_mask(mask, student_logits.shape[:2])
    student = student_logits.masked_fill(~valid[..., None], 0).to(accumulation_dtype(student_logits))
    teacher = teacher_logits.detach().masked_fill(~valid[..., None], 0).to(student.dtype)
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("valid distillation logits must be finite")
    logp, logq = F.log_softmax(student / temperature, -1), F.log_softmax(teacher / temperature, -1)
    kl = (logq.exp() * (logq-logp)).sum(-1).masked_fill(~valid, 0)
    return temperature**2 * kl.sum() / valid.sum()


def _check_document_ids(document_ids):
    if document_ids.ndim != 2 or document_ids.dtype not in (torch.int32, torch.int64) \
            or torch.any(document_ids < -1):
        raise ValueError("integer document IDs [B,T], with -1 reserved for padding, required")


def packed_causal_mask(document_ids):
    """Return [B,T,T] allowed edges. IDs >=0 are documents; -1 is padding.

    Document IDs must identify unique segments within each sequence. Padding query
    rows are all False and need separate handling before an attention softmax.
    """
    _check_document_ids(document_ids)
    t = document_ids.shape[-1]
    causal = torch.arange(t, device=document_ids.device)[None, :] <= torch.arange(t, device=document_ids.device)[:, None]
    valid = document_ids >= 0
    same_document = document_ids[:, :, None] == document_ids[:, None, :]
    return same_document & causal & valid[:, :, None] & valid[:, None, :]


def packed_next_token_mask(document_ids, supervised=None):
    """Mask for ids[:,:-1] -> ids[:,1:] within independent packed documents.

    `document_ids` describes the unshifted token stream; each segment gets a
    unique ID per row, including its BOS/EOS. `supervised`, if supplied, is an
    unshifted boolean/binary mask on target roles. This function neither builds
    the attention mask nor inserts a BOS, so it cannot restore an absent first
    token prediction. All-zero rows are returned for callers to handle explicitly.
    """
    _check_document_ids(document_ids)
    valid = document_ids >= 0
    allowed = valid[:, :-1] & valid[:, 1:] & (document_ids[:, :-1] == document_ids[:, 1:])
    if supervised is not None:
        if supervised.shape != document_ids.shape or supervised.device != document_ids.device \
                or not torch.all((supervised == 0) | (supervised == 1)):
            raise ValueError("aligned binary supervision mask on the same device required")
        allowed = allowed & supervised[:, 1:].bool()
    return allowed

```
## posttrain.py
[对应文件](../../examples/posttrain.py)
```python
"""Post-training objectives (DPO, SimPO, GRPO family, GSPO) and a complete GRPO loop on a tiny LM.

Teaching reference with its own conventions, chosen for explanation; they do not match the
2026 A5 handout. Do not submit or adapt it as assignment code; write your own from the handout
(coursework restrictions are explained in chapter 28).
"""
import json
import math
import torch
from torch.nn import functional as F
from .objectives import accumulation_dtype, response_mask, sequence_values, token_logps


def masked_sequence_logps(logits, labels, response_mask):
    # logits and labels are already shifted/aligned; [B,T,V], [B,T], [B,T].
    return token_logps(logits, labels, response_mask).sum(-1)


def token_entropy(logits, mask):
    """Full-vocabulary conditional entropy; invalid padding contributes exactly zero."""
    if logits.ndim!=3 or not logits.shape[-1] or not logits.is_floating_point() or logits.device != mask.device:
        raise ValueError("expected [batch,time,vocabulary] logits")
    valid=response_mask(mask,logits.shape[:2])
    work=logits.masked_fill(~valid[...,None],0).to(accumulation_dtype(logits))
    if not torch.isfinite(work).all():
        raise ValueError("valid logits must be finite")
    logp=F.log_softmax(work,-1)
    return -(logp.exp()*logp).sum(-1).masked_fill(~valid,0)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    policy_chosen, policy_rejected, ref_chosen, ref_rejected = sequence_values(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    margin = beta * ((policy_chosen - ref_chosen.detach()) - (policy_rejected - ref_rejected.detach()))
    return -F.logsigmoid(margin).mean()


def simpo_loss(chosen_logps, rejected_logps, chosen_lengths, rejected_lengths, beta=2.0, margin=1.0):
    """Reference-free, length-normalized preference objective for supplied sequence logps."""
    if not math.isfinite(beta) or not math.isfinite(margin) or beta<=0 or margin<0:
        raise ValueError("positive beta and nonnegative margin required")
    if any(x.shape!=chosen_logps.shape for x in [rejected_logps,chosen_lengths,rejected_lengths]):
        raise ValueError("all per-sequence inputs must have matching shapes")
    if torch.any(chosen_lengths<=0) or torch.any(rejected_lengths<=0):
        raise ValueError("responses must have positive effective lengths")
    chosen_logps, rejected_logps = sequence_values(chosen_logps, rejected_logps)
    if any(x.device != chosen_logps.device or not torch.isfinite(x).all() for x in (chosen_lengths, rejected_lengths)):
        raise ValueError("finite effective lengths on the log-probability device required")
    gap=beta*(chosen_logps/chosen_lengths-rejected_logps/rejected_lengths)-margin
    return -F.logsigmoid(gap).mean()


def advantages(rewards, method="grpo", eps=1e-6):
    # [prompts, group_size], population standard deviation (correction=0).
    if not rewards.is_floating_point():
        rewards = rewards.to(torch.get_default_dtype())  # integer/bool 0-1 rewards are common
    if rewards.ndim != 2 or min(rewards.shape) < 1 or not math.isfinite(eps) or eps <= 0 or not torch.isfinite(rewards).all():
        raise ValueError("finite, nonempty [prompts,group] rewards and positive eps required")
    if method in ("maxrl", "rft") and not torch.all((rewards == 0) | (rewards == 1)):
        raise ValueError("this MaxRL/RFT teaching variant assumes binary rewards")
    rewards = rewards.detach().to(accumulation_dtype(rewards))
    mean = rewards.mean(-1, keepdim=True)
    centered = rewards - mean
    if method == "grpo":
        return centered / (rewards.std(-1, keepdim=True, correction=0) + eps)
    if method == "dr_grpo":
        return centered
    if method == "maxrl":
        return centered / mean.clamp_min(eps)
    if method == "rft":
        return rewards
    if method == "rloo":
        n = rewards.shape[-1]
        if n < 2:
            raise ValueError("leave-one-out needs at least 2 responses")
        return centered * n / (n - 1)
    raise ValueError(method)


@torch.no_grad()
def generalized_advantages(rewards, values, next_values, terminated, truncated, gamma=1., lam=.95):
    """GAE for a flat vector of real transitions; no padding or advantage normalization.

    values[t] = V(s_t), next_values[t] = V(s_{t+1}) before any environment reset.
    A true terminal disables bootstrap and the trace. A sampling timeout keeps
    the final-state bootstrap but stops the trace before the next episode. Both
    outputs (advantages, lambda-return value targets) are detached. The final
    recorded transition stops the trace even if neither boundary flag is set.
    """
    if rewards.ndim != 1 or not rewards.numel() or any(
        x.shape != rewards.shape or x.device != rewards.device or not x.is_floating_point()
        for x in (rewards, values, next_values)
    ):
        raise ValueError("matching nonempty floating [T] transition vectors on one device required")
    if any(x.shape != rewards.shape or x.device != rewards.device or x.dtype != torch.bool
           for x in (terminated, truncated)):
        raise ValueError("aligned boolean terminated/truncated flags required")
    if not math.isfinite(gamma) or not math.isfinite(lam) or not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("finite gamma and lambda in [0,1] required")
    dtype = torch.float64 if any(x.dtype == torch.float64 for x in (rewards, values, next_values)) else torch.float32
    reward, value = rewards.to(dtype), values.to(dtype)
    following = next_values.to(dtype).masked_fill(terminated, 0)
    if any(not torch.isfinite(x).all() for x in (reward, value, following)):
        raise ValueError("finite rewards, values and nonterminal bootstrap values required")
    residual = reward + gamma * following - value
    result = torch.empty_like(residual)
    carry = residual.new_zeros(())
    continuing = ~(terminated | truncated)
    for t in range(len(residual) - 1, -1, -1):
        carry = residual[t] + gamma * lam * torch.where(continuing[t], carry, torch.zeros_like(carry))
        result[t] = carry
    return result, result + value


def pg_estimator_variance(probs, rewards, scores, baseline):
    """Exact per-sample variance of (r - b) * score for a small discrete policy, by enumeration.

    probs, rewards and scores give one entry per action; score is d log pi(a) / d theta along
    one parameter direction. The mean, sum_a pi(a) r(a) score(a), does not depend on b.
    """
    z = [(r - baseline) * g for r, g in zip(rewards, scores)]
    mean = sum(p * x for p, x in zip(probs, z))
    return sum(p * x * x for p, x in zip(probs, z)) - mean ** 2


def optimal_baseline(probs, rewards, scores):
    """The variance-minimizing constant baseline b* = E[r s^2] / E[s^2] (chapter 21.3)."""
    return (sum(p * r * g * g for p, r, g in zip(probs, rewards, scores))
            / sum(p * g * g for p, g in zip(probs, scores)))


def masked_log_ratio(logp, old_logp, mask):
    if logp.shape != old_logp.shape or not logp.is_floating_point() or not old_logp.is_floating_point():
        raise ValueError("current and old log probabilities must have matching shapes")
    if logp.device != old_logp.device or mask.device != logp.device:
        raise ValueError("log probabilities and response mask must share a device")
    valid = response_mask(mask, logp.shape)
    current = logp.masked_fill(~valid, 0).to(accumulation_dtype(logp))
    old = old_logp.detach().masked_fill(~valid, 0).to(current.dtype)
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise ValueError("nonfinite log probability at a valid response token")
    return current - old, valid


def finite_ratio(log_ratio):
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("importance ratio overflow on valid tokens; inspect policy lag")
    return ratio


def check_advantage(advantage, batch, eps):
    if advantage.shape != (batch,) or not torch.isfinite(advantage).all() or not 0 <= eps < 1:
        raise ValueError("finite [batch] advantage and 0 <= clipping epsilon < 1 required")


def on_policy_loss(logp, advantage, mask, normalization="sequence", max_length=None):
    valid = response_mask(mask, logp.shape)
    clean = logp.masked_fill(~valid, 0).to(accumulation_dtype(logp))
    check_advantage(advantage, len(logp), 0)
    if not torch.isfinite(clean).all():
        raise ValueError("nonfinite log probability at a valid response token")
    objective = clean * advantage.detach()[:, None]
    return aggregate_policy_objective(objective, valid, normalization, max_length)


def aggregate_policy_objective(objective, mask, normalization, max_length):
    if normalization == "sequence":
        return -(objective.sum(-1) / mask.sum(-1)).mean()
    if normalization == "constant":
        if max_length is None or max_length < 1 or torch.any(mask.sum(-1) > max_length):
            raise ValueError("fixed max_length must bound all valid response lengths")
        return -objective.sum() / (len(objective) * max_length)
    raise ValueError(normalization)


def clipped_policy_loss(logp, old_logp, advantage, mask, eps=0.2, normalization="sequence", max_length=None):
    # [B,T] token log probabilities; advantage [B].
    log_ratio, valid = masked_log_ratio(logp, old_logp, mask)
    check_advantage(advantage, len(logp), eps)
    ratio = finite_ratio(log_ratio)
    a = advantage.detach()[:, None]
    objective = torch.minimum(ratio * a, ratio.clamp(1-eps, 1+eps) * a).masked_fill(~valid, 0)
    return aggregate_policy_objective(objective, valid, normalization, max_length)


def gspo_loss(logp, old_logp, advantage, mask, eps=0.01):
    delta, valid = masked_log_ratio(logp, old_logp, mask)
    check_advantage(advantage, len(logp), eps)
    log_ratio = delta.sum(-1) / valid.sum(-1)
    ratio = finite_ratio(log_ratio)
    a = advantage.detach()
    return -torch.minimum(ratio * a, ratio.clamp(1-eps, 1+eps) * a).mean()


@torch.no_grad()
def sampled_kl_metric(logp, ref_logp, mask):
    """Nonnegative k3 diagnostic under samples from CURRENT policy, not an RL loss.

    On a fixed action sample, differentiating this scalar is not generally the
    gradient of the expected KL. Old-policy samples also change its expectation.
    """
    delta, valid = masked_log_ratio(ref_logp, logp, mask)
    values = torch.expm1(delta) - delta
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite KL diagnostic on a valid token")
    return values.masked_fill(~valid, 0).sum() / valid.sum()


ARITHMETIC_FACTS = [(a, b) for a in range(10) for b in range(10)]


def arithmetic_reward(a, b, response):
    """(total, format): format needs an ASCII digit then newline; total also needs the right digit.

    str.isdigit() would also accept "²" (U+00B2) or Arabic-Indic digits, which int() rejects or misreads.
    """
    text = bytes(response).decode("utf-8", errors="replace")
    formatted = len(text) == 2 and text[0] in "0123456789" and text[1] == "\n"
    return float(formatted and int(text[0]) == (a + b) % 10), float(formatted)


@torch.no_grad()
def greedy_accuracy(model, facts):
    from .model import generate
    prompts = torch.tensor([list(f"{a}+{b}=".encode()) for a, b in facts])
    responses = generate(model, prompts, 2, temperature=0)[:, 4:]
    return sum(arithmetic_reward(a, b, r.tolist())[0] for (a, b), r in zip(facts, responses)) / len(facts)


def grpo_arithmetic_demo(steps=200, prompts_per_step=32, group=8, lr=1e-3, epochs_per_rollout=2,
                         microbatches=4, method="grpo", clip=0.2, seed=0, pretrain_steps=120, verbose=True):
    """A complete on-policy RL loop on a real (tiny) language model.

    Policy: TinyLM pretrained for `pretrain_steps` on synthetic arithmetic text (it knows
    the format but mostly guesses the answer). Each step: sample prompts, roll out a
    group of 2-byte answers per prompt, score them, compute group advantages, then run
    `epochs_per_rollout` passes of the clipped surrogate with microbatch accumulation.
    The first pass is on-policy (ratio = 1); later passes reuse stale rollouts.
    20 facts never appear as RL prompts (they do occur in the pretraining text), so their
    accuracy shows whether RL sharpened shared skill or only rewarded prompts.
    """
    import contextlib
    import io
    import random
    import tempfile
    from .model import Config, TinyLM, generate
    from .optim import AdamW, clip_grad_norm
    from .train import run
    if any(type(value) is not int or value < 1 for value in
           (steps, prompts_per_step, group, epochs_per_rollout, microbatches, pretrain_steps)):
        raise ValueError("positive integer step, group, epoch and microbatch counts required")
    if not math.isfinite(lr) or lr <= 0 or not 0 <= clip < 1 or method not in ('grpo', 'dr_grpo', 'maxrl', 'rloo', 'rft'):
        raise ValueError("positive finite learning rate, valid clip and supported advantage required")
    torch.manual_seed(seed)
    with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
        run(steps=pretrain_steps, out=folder)
        saved = torch.load(f"{folder}/checkpoint.pt", weights_only=True)
    policy = TinyLM(Config(**saved["config"]))
    policy.load_state_dict(saved["model"])
    facts = ARITHMETIC_FACTS[:]
    random.Random(seed).shuffle(facts)
    train_facts, heldout = facts[:80], facts[80:]
    opt = AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    sampler, rng = torch.Generator().manual_seed(seed), random.Random(seed + 1)
    log = [dict(step=0, train_accuracy=greedy_accuracy(policy, train_facts),
                heldout_accuracy=greedy_accuracy(policy, heldout))]
    interval_norms = []  # pre-clip norm of EVERY optimizer update since the last log entry
    for step in range(1, steps + 1):
        chosen = [rng.choice(train_facts) for _ in range(prompts_per_step)]
        prompts = torch.tensor([list(f"{a}+{b}=".encode()) for a, b in chosen]).repeat_interleave(group, 0)
        sequences = generate(policy, prompts, 2, temperature=1.0, generator=sampler)
        responses = sequences[:, 4:]
        scored = [arithmetic_reward(a, b, r.tolist()) for (a, b), r in
                  zip([f for f in chosen for _ in range(group)], responses)]
        rewards = torch.tensor([r for r, _ in scored]).view(prompts_per_step, group)
        adv = advantages(rewards, method).reshape(-1)
        mask = torch.ones_like(responses, dtype=torch.float32)
        policy.train()

        def response_logits():
            # Logits at positions 3 and 4 predict the two response bytes.
            return policy(sequences[:, :-1])[:, 3:]

        with torch.no_grad():
            old_logp = token_logps(response_logits(), responses, mask)
        for _ in range(epochs_per_rollout):
            opt.zero_grad(set_to_none=True)
            for rows in torch.arange(len(sequences)).chunk(microbatches):
                logits = policy(sequences[rows, :-1])[:, 3:]
                logp = token_logps(logits, responses[rows], mask[rows])
                loss = clipped_policy_loss(logp, old_logp[rows], adv[rows], mask[rows], eps=clip)
                # Sequence-normalized loss: weight each microbatch by its share of sequences.
                (loss * len(rows) / len(sequences)).backward()
            grad_norm = clip_grad_norm(policy.parameters(), 1.0).item()
            interval_norms.append(grad_norm)
            opt.step()
        if step % 10 == 0 or step == steps:
            with torch.no_grad():
                entropy = token_entropy(response_logits(), mask).mean().item()
            log.append(dict(step=step, train_reward=rewards.mean().item(),
                            format_reward=sum(f for _, f in scored) / len(scored),
                            mixed_groups=((rewards.min(1).values == 0) & (rewards.max(1).values == 1)).float().mean().item(),
                            entropy=entropy, grad_norm=grad_norm,
                            max_grad_norm=max(interval_norms), clipped_updates=sum(n > 1.0 for n in interval_norms),
                            updates=len(interval_norms),
                            train_accuracy=greedy_accuracy(policy, train_facts),
                            heldout_accuracy=greedy_accuracy(policy, heldout)))
            interval_norms.clear()
            if verbose:
                print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in log[-1].items()}), flush=True)
    return dict(method=method, steps=steps, rollouts_per_step=prompts_per_step * group, log=log,
                scope="2-byte answers from a 125K-parameter model; shows the RL loop, not reasoning")


if __name__ == "__main__":
    print(json.dumps(grpo_arithmetic_demo(), indent=2))

```
## evaluation.py
[对应文件](../../examples/evaluation.py)
```python
"""Stable pass@k/pass^k, Wilson intervals, paired bootstrap and toy-LM diagnostics."""
import argparse
import json
import math
import operator
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import Config, TinyLM
from .train import corpus


def _log_fraction(numerator, denominator):
    # Near 1 use log1p; a tiny fraction can instead round its complement to exactly 1.
    if 2 * numerator > denominator:
        return math.log1p(-(denominator - numerator) / denominator)
    return math.log(numerator) - math.log(denominator)


def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples with c correct: 1 - C(n-c, k) / C(n, k), in a stable product form."""
    try:
        n, c, k = (operator.index(x) for x in (n, c, k))  # accepts numpy integers, rejects floats
    except TypeError as error:
        raise ValueError("n, c and k must be integers") from error
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("integer counts must satisfy 0 <= c <= n and 1 <= k <= n")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return -math.expm1(math.fsum(_log_fraction(n - c - i, n - i) for i in range(k)))


def pass_all_k(n, c, k):
    """Unbiased pass^k: all k independent attempts on the SAME task must succeed.

    Average C(c,k)/C(n,k) over tasks; raising the pooled accuracy to k loses task heterogeneity.
    """
    try:
        n, c, k = (operator.index(x) for x in (n, c, k))
    except TypeError as error:
        raise ValueError("n, c and k must be integers") from error
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("integer counts must satisfy 0 <= c <= n and 1 <= k <= n")
    if c < k:
        return 0.0
    if c == n:
        return 1.0
    return math.exp(math.fsum(_log_fraction(c - i, n - i) for i in range(k)))


def wilson_interval(successes, total, z=1.959963984540054):
    try:
        successes, total = operator.index(successes), operator.index(total)
    except TypeError as error:
        raise ValueError("integer trial and success counts required") from error
    if total < 1 or not 0 <= successes <= total or not math.isfinite(z) or z <= 0:
        raise ValueError("valid counts and positive z required")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def paired_bootstrap(a, b, repetitions=2000, seed=17):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0 or repetitions < 1:
        raise ValueError("matching nonempty per-item score vectors and positive repetitions required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("scores must be finite")
    rng = np.random.default_rng(seed)
    delta = b - a
    values = [delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(repetitions)]
    return dict(difference=float(delta.mean()), interval_95=np.quantile(values, [.025, .975]).tolist())


def paired_cluster_bootstrap(a, b, clusters, repetitions=2000, seed=17, weighting="item"):
    """Resample independent clusters, keeping all paired items within each draw.

    `item` estimates total paired difference / total items; `cluster` gives
    each cluster mean equal weight. Every draw samples G clusters with replacement,
    including multiplicity. This covers cluster sampling, not training-seed noise
    or model selection, and needs at least two independent clusters.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    labels = list(clusters)
    if a.ndim != 1 or a.shape != b.shape or a.size != len(labels) or not a.size \
            or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite matching per-item scores and one cluster ID per pair required")
    if type(repetitions) is not int or repetitions < 2 or weighting not in ("item", "cluster"):
        raise ValueError("at least two repetitions and item/cluster weighting required")
    groups = {}
    for index, label in enumerate(labels):
        if label is None or (isinstance(label, (float, np.floating)) and not math.isfinite(label)):
            raise ValueError("cluster IDs cannot be missing or nonfinite")
        try:
            groups.setdefault(label, []).append(index)
        except TypeError as error:
            raise ValueError("hashable cluster IDs required") from error
    if len(groups) < 2:
        raise ValueError("at least two independent clusters required")
    delta = b - a
    totals = np.array([delta[indices].sum() for indices in groups.values()])
    sizes = np.array([len(indices) for indices in groups.values()])
    means = totals / sizes
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        draw = rng.integers(len(groups), size=len(groups))
        samples.append(totals[draw].sum() / sizes[draw].sum() if weighting == "item" else means[draw].mean())
    return dict(difference=float(delta.mean() if weighting == "item" else means.mean()),
                interval_95=np.quantile(samples, [.025, .975]).tolist(),
                clusters=len(groups), weighting=weighting)


def elo_win_probability(rating_a, rating_b):
    """Arena-style Elo: Bradley-Terry with base-10 logits scaled by 400."""
    if not math.isfinite(rating_a) or not math.isfinite(rating_b):
        raise ValueError("finite ratings required")
    delta = (rating_a - rating_b) * (math.log(10) / 400)
    small = math.exp(-abs(delta))
    return 1 / (1 + small) if delta >= 0 else small / (1 + small)


@torch.no_grad()
def arithmetic_diagnostics(checkpoint, records_per_window=5):
    """Per-role NLL and exact-answer accuracy for the synthetic arithmetic model.

    Windows hold whole 6-byte records and stay inside the training context, so
    every position the model is scored on was seen during training.
    """
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = TinyLM(Config(**saved["config"]))
    model.load_state_dict(saved["model"])
    model.eval()
    window = 6 * records_per_window
    tokens = corpus(2, 600)[: (3600 // window) * window].reshape(-1, window)
    logits = model(tokens[:, :-1])
    nll = F.cross_entropy(logits.reshape(-1, 256), tokens[:, 1:].reshape(-1), reduction="none").reshape(len(tokens), -1)
    # Target index i predicts byte i+1 of the window; byte offset within a record decides its role.
    offsets = torch.arange(1, window) % 6
    roles = {"operand": (offsets == 0) | (offsets == 2), "format": (offsets == 1) | (offsets == 3) | (offsets == 5),
             "answer": offsets == 4}
    result = {name: dict(tokens=int(mask.sum()) * len(tokens), nll=nll[:, mask].mean().item())
              for name, mask in roles.items()}
    prompts = torch.tensor([list(f"{a}+{b}=".encode()) for a in range(10) for b in range(10)])
    targets = torch.tensor([ord(str((a + b) % 10)) for a in range(10) for b in range(10)])
    last = model(prompts)[:, -1]
    successes = int((last.argmax(-1) == targets).sum())
    return dict(checkpoint_step=saved["step"], position_metrics=result,
                all_100_prompt_accuracy=successes / 100, correct_prompts=successes,
                answer_nll_from_fresh_prompt=F.cross_entropy(last, targets).item(),
                exact_arithmetic_reference_nats_per_byte=2 * math.log(10) / 6,
                scope="all 100 facts also occur in training text; this measures fitting seen facts, not OOD arithmetic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = arithmetic_diagnostics(args.checkpoint) if args.checkpoint else {
        "pass_at_3": pass_at_k(10, 2, 3), "wilson_50_of_100": wilson_interval(50, 100),
        "paired_example": paired_bootstrap([0, 1, 1, 0], [1, 1, 1, 0]),
        "elo_100_point_gap_win_probability": elo_win_probability(1100, 1000)}
    print(json.dumps(result, indent=2))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")

```
## extensions.py
[对应文件](../../examples/extensions.py)
```python
"""Small MoE, linear-attention and image/text-alignment references.

These implementations expose mathematical structure, not production efficiency.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TopKMoE(nn.Module):
    """All experts are evaluated for clarity; sparse *outputs*, not sparse compute."""
    def __init__(self, width=16, hidden=32, experts=4, topk=2):
        super().__init__()
        if not 1 <= topk <= experts:
            raise ValueError("1 <= topk <= experts required")
        self.topk = topk
        self.router = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(width, hidden), nn.SiLU(),
                                                   nn.Linear(hidden, width)) for _ in range(experts))

    def forward(self, x):
        logits = self.router(x)
        probs = logits.to(torch.float64 if logits.dtype == torch.float64 else torch.float32).softmax(-1)
        weights, indices = probs.topk(self.topk, dim=-1)
        # Retain probabilities from the full softmax, including for top-1.
        gates = torch.zeros_like(probs).scatter(-1, indices, weights).to(x.dtype)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        y = (outputs * gates[..., None]).sum(-2)
        assignments = F.one_hot(indices, num_classes=len(self.experts)).float().sum(-2)
        load = assignments.reshape(-1, len(self.experts)).mean(0) / self.topk
        mean_probability = probs.reshape(-1,len(self.experts)).mean(0)
        balance = len(self.experts) * (load.detach() * mean_probability).sum()
        return y, balance, load


def _linear_attention_inputs(q, k, v):
    if q.ndim != 2 or k.shape != q.shape or v.ndim != 2 or len(v) != len(q) \
            or min(*q.shape, v.shape[-1]) < 1:
        raise ValueError("nonempty q,k [T,Dk] and v [T,Dv] with matching time axes required")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64) or any(
        x.dtype != q.dtype or x.device != q.device or not torch.isfinite(x).all() for x in (q, k, v)
    ):
        raise ValueError("finite matching floating dtypes and devices required")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    def feature(x):
        x = x.to(dtype)
        # ELU(x)+1 equals exp(x) on x<0. Computing (exp(x)-1)+1
        # loses small positive values to cancellation, even in FP32.
        return torch.where(x >= 0, x + 1, x.clamp(max=0).exp())
    return feature(q), feature(k), v.to(dtype)


def linear_attention_recurrent(q, k, v):
    """Causal ELU+1 kernel, denominator floor 1e-8; FP32/64 accumulation.

    Returns the input dtype. This is not softmax attention; the denominator
    floor changes the normalized kernel when its total mass is very small.
    """
    output_dtype = q.dtype
    q, k, v = _linear_attention_inputs(q, k, v)
    # Explicit casts alone do not stop an enclosing autocast region from
    # lowering these matrix products again and erasing small feature values.
    with torch.autocast(device_type=q.device.type, enabled=False):
        state = q.new_zeros(q.shape[-1], v.shape[-1])
        normalizer = q.new_zeros(q.shape[-1])
        outputs = []
        for qt, kt, vt in zip(q,k,v):
            state = state + kt[:, None] * vt[None, :]
            normalizer = normalizer + kt
            outputs.append((qt @ state) / (qt @ normalizer).clamp_min(1e-8))
        return torch.stack(outputs).to(output_dtype)


def linear_attention_explicit(q, k, v):
    """Dense causal reference for the same kernel and denominator convention."""
    output_dtype = q.dtype
    q, k, v = _linear_attention_inputs(q, k, v)
    with torch.autocast(device_type=q.device.type, enabled=False):
        weights = (q @ k.T).tril()
        return ((weights @ v) / weights.sum(-1,keepdim=True).clamp_min(1e-8)).to(output_dtype)


def clip_loss(image_features, text_features, temperature=0.1):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    if image_features.ndim != 2 or image_features.shape != text_features.shape or min(image_features.shape) < 1 \
            or image_features.device != text_features.device or not torch.isfinite(image_features).all() \
            or not torch.isfinite(text_features).all():
        raise ValueError('finite aligned [batch, features] image/text pairs on one device required')
    images, texts = F.normalize(image_features,dim=-1), F.normalize(text_features,dim=-1)
    scores = images @ texts.T / temperature
    targets = torch.arange(len(scores),device=scores.device)
    return (F.cross_entropy(scores,targets)+F.cross_entropy(scores.T,targets))/2


if __name__ == '__main__':
    torch.manual_seed(2)
    q, k, v = [torch.randn(13, 8) for _ in range(3)]
    error = (linear_attention_recurrent(q, k, v) - linear_attention_explicit(q, k, v)).abs().max().item()
    print('linear attention recurrent/explicit maximum error:', error)
    model = TopKMoE()
    y, balance, load = model(torch.randn(2, 5, 16))
    print('MoE shape:', list(y.shape), 'expert load:', load.detach().tolist(), 'balance loss:', balance.item())
    features = torch.eye(8)
    print('CLIP aligned loss:', clip_loss(features, features).item(),
          'permuted loss:', clip_loss(features, features.roll(1, 0)).item())

```
## research.py
[对应文件](../../examples/research.py)
```python
"""Small numerical tools for experimental design and domain adaptation."""
import json
import numpy as np


def reweight_domains(weights, excess_losses, eta=1.0, floor_mass=0.05):
    """One illustrative exponentiated-gradient step; not a DoReMi trainer."""
    w = np.asarray(weights, dtype=float)
    e = np.asarray(excess_losses, dtype=float)
    if w.ndim != 1 or w.size == 0 or w.shape != e.shape or np.any(w <= 0):
        raise ValueError("matching nonempty vectors and strictly positive weights required")
    if not np.isfinite(w).all() or not np.isfinite(e).all() or not np.isfinite(eta) or eta < 0 or not 0 <= floor_mass <= 1:
        raise ValueError("finite inputs, nonnegative eta and floor_mass in [0,1] required")
    # Normalization cancels inside softmax; never sum huge weights in ordinary space.
    logw = np.log(w)
    logw -= logw.max()
    if eta == 0:
        shifted = np.zeros_like(e)
    else:
        with np.errstate(over='ignore'):
            shifted = (e - e.max()) * eta
    logits = logw + shifted
    updated = np.exp(logits - logits.max())
    updated /= updated.sum()
    return (1 - floor_mass) * updated + floor_mass / len(w)


def pareto_front(rows, maximize):
    """Return indices not strictly dominated; one direction flag per metric."""
    x = np.asarray(rows, dtype=float)
    directions = np.asarray(maximize, dtype=bool)
    if x.ndim != 2 or len(x) == 0 or directions.shape != (x.shape[1],) or not np.isfinite(x).all():
        raise ValueError("finite nonempty matrix and one direction flag per metric required")
    oriented = x * np.where(directions, 1, -1)
    return [i for i, row in enumerate(oriented) if not any(
        np.all(other >= row) and np.any(other > row) for j, other in enumerate(oriented) if j != i)]


def factorial_interaction(s00, s10, s01, s11):
    return s11 - s10 - s01 + s00


def gradient_transfer_demo(step=0.1):
    """Two opposing quadratic tasks: local conflict, not an LM forgetting experiment."""
    theta = np.zeros(2)
    a, b = np.array([1., 0.]), np.array([-1., 0.])
    ga, gb = theta - a, theta - b
    updated = theta - step * gb
    return dict(gradient_inner_product=float(ga @ gb),
                predicted_old_loss_change=float(-step * (ga @ gb)),
                actual_old_loss_change=float(.5 * np.sum((updated - a) ** 2) - .5 * np.sum((theta - a) ** 2)),
                actual_new_loss_change=float(.5 * np.sum((updated - b) ** 2) - .5 * np.sum((theta - b) ** 2)))


if __name__ == "__main__":
    print(json.dumps(dict(domain_weights=reweight_domains([.5, .3, .2], [.1, .8, .2]).tolist(),
                          pareto_indices=pareto_front([[80, 200], [81, 300], [79, 250]], [True, False]),
                          interaction=factorial_interaction(70, 72, 73, 74),
                          gradient_transfer=gradient_transfer_demo()), indent=2))

```
## retrieval.py
[对应文件](../../examples/retrieval.py)
```python
"""BM25 retrieval, reciprocal-rank fusion and ranking metrics on an original fixed corpus.

No learned embeddings, ANN index, or language-generation call is used here.
"""
from collections import Counter
import json
import math
import operator
import re


def terms(text):
    # ASCII words/numbers and individual CJK characters; an explicit toy convention.
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


class BM25:
    def __init__(self, documents, k1=1.2, b=0.75):
        documents = list(documents)
        if not documents or len({key for key, _ in documents}) != len(documents) or not math.isfinite(k1) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("unique nonempty corpus, positive k1 and b in [0,1] required")
        self.documents = list(documents)
        self.k1, self.b = k1, b
        self.counts = [Counter(terms(text)) for _, text in documents]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / len(self.lengths)
        if self.average == 0:
            raise ValueError("corpus must contain searchable terms")
        self.df = Counter(term for c in self.counts for term in c)

    def search(self, query, k=3):
        if type(k) is not int or k < 1:
            raise ValueError("k must be positive")
        result, n = [], len(self.documents)
        query_terms = set(terms(query))
        for index, counts in enumerate(self.counts):
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = self.df[term]
                idf = math.log1p((n - df + .5) / (df + .5))
                denominator = tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average)
                score += idf * tf * (self.k1 + 1) / denominator
            if score > 0:
                result.append((self.documents[index][0], score))
        return sorted(result, key=lambda x: (-x[1], x[0]))[:k]


def reciprocal_rank_fusion(rankings, k=60):
    """Merge ranked ID lists by sum of 1/(k + rank); uses ranks only, so score scales never mix."""
    if not math.isfinite(k) or k < 0:
        raise ValueError('finite nonnegative RRF rank offset required')
    scores = Counter()
    for ranking in rankings:
        for rank, doc in enumerate(dict.fromkeys(ranking), start=1):
            scores[doc] += 1 / (k + rank)
    return [doc for doc, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def ranking_metrics(ranked_ids, relevant, k):
    relevant = set(relevant)
    if type(k) is not int or k < 1 or not relevant:
        raise ValueError("positive k and nonempty relevant set required")
    ranked = list(dict.fromkeys(ranked_ids))[:k]
    relevant = set(relevant)
    first = next((i + 1 for i, doc in enumerate(ranked) if doc in relevant), None)
    return dict(recall=len(set(ranked) & relevant) / len(relevant), mrr=0.0 if first is None else 1 / first)


def ndcg_at_k(ranked_ids, gains, k):
    """Graded relevance: DCG = sum_i (2^g_i - 1) / log2(i + 1) over the top k, divided by the ideal DCG.

    gains maps document id -> finite nonnegative relevance grade (missing ids count as 0). Unlike recall@k,
    a highly relevant document ranked first is worth more than a marginal one ranked first.
    """
    try:
        k = operator.index(k)
        gains = {doc: float(grade) for doc, grade in gains.items()}
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("integer k and finite numeric relevance grades required") from error
    if k < 1 or not gains or any(not math.isfinite(g) or g < 0 for g in gains.values()) or max(gains.values()) == 0:
        raise ValueError("positive integer k and finite nonnegative grades with a positive maximum required")
    ranked = list(dict.fromkeys(ranked_ids))[:k]
    maximum = max(gains.values())
    def gain(grade):
        # Divide numerator and denominator by 2**maximum; expm1 preserves tiny positive grades.
        return 2.0 ** (grade - maximum) * -math.expm1(-grade * math.log(2))
    dcg = math.fsum(gain(gains.get(doc, 0)) / math.log2(i + 2) for i, doc in enumerate(ranked))
    ideal = sorted(gains.values(), reverse=True)[:k]
    return dcg / math.fsum(gain(g) / math.log2(i + 2) for i, g in enumerate(ideal))


def demo():
    corpus = [("rope", "RoPE 使用旋转位置编码，让 query 和 key 的内积与相对位置有关。"),
              ("kv", "KV cache 缓存历史键值，避免逐 token 生成时重复计算过去前缀。"),
              ("adam", "AdamW 保存一阶和二阶动量，并解耦权重衰减。"),
              ("data", "训练数据要过滤、去重，并保留来源和版本。")]
    index = BM25(corpus)
    queries = [("RoPE 相对位置", {"rope"}), ("KV cache 历史键值", {"kv"}), ("AdamW 动量", {"adam"}),
               # A paraphrase with no shared terms: lexical retrieval returns nothing.
               ("如何表示词序", {"rope"})]
    results = []
    for query, relevant in queries:
        ranked = index.search(query)
        results.append(dict(query=query, ranked=ranked, metrics=ranking_metrics([x[0] for x in ranked] or ["none"], relevant, 3)))
    fused = reciprocal_rank_fusion([["kv", "rope", "data"], ["rope", "adam"]])
    return dict(scope="fixed four-document lexical retrieval demo; not an end-to-end RAG evaluation",
                queries=results, rrf_example=fused)


if __name__ == "__main__":
    print(json.dumps(demo(), ensure_ascii=False, indent=2))

```
## reasoning.py
[对应文件](../../examples/reasoning.py)
```python
"""Synthetic candidate generation and verifier selection: no real LM benchmark."""
import json
import math
import numpy as np


def iid_pass_probability(success_probability, k):
    if not 0 <= success_probability <= 1 or not isinstance(k, int) or k < 1:
        raise ValueError("probability in [0,1] and positive integer k required")
    if success_probability == 1:
        return 1.0
    # The direct form is exact for moderate p (e.g. 1-(1-0.25)**1 == 0.25); expm1/log1p keeps
    # precision only where it is needed, when p is so small that 1-p rounds toward 1.
    if success_probability < 1e-8:
        return -math.expm1(k * math.log1p(-success_probability))
    return 1 - (1 - success_probability) ** k


def select_candidate(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("finite nonempty score vector required")
    return int(np.argmax(scores))


def verifier_experiment(seed=23, trials=20000, success_probability=.25, false_high_probability=.03, ks=(1, 4, 16, 64)):
    if type(trials) is not int or trials < 1 or not ks or not 0 < success_probability < 1 or not 0 <= false_high_probability <= 1 \
            or any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("invalid experiment settings")
    rng = np.random.default_rng(seed)
    maximum = max(ks)
    correct = rng.random((trials, maximum)) < success_probability
    false_high = (~correct) & (rng.random((trials, maximum)) < false_high_probability)
    # Correct answers score 1, verifier-fooling wrong answers 2, other wrong answers 0; tiny jitter breaks ties.
    scores = np.where(false_high, 2., np.where(correct, 1., 0.)) + rng.random((trials, maximum)) * 1e-4
    false_high_rate = (1 - success_probability) * false_high_probability
    records = []
    for k in ks:
        picked = scores[:, :k].argmax(-1)
        records.append(dict(k=k, oracle_coverage=float(correct[:, :k].any(-1).mean()),
                            selected_accuracy=float(correct[np.arange(trials), picked].mean()),
                            theoretical_iid_coverage=iid_pass_probability(success_probability, k),
                            theoretical_selected_accuracy=(1 - false_high_rate) ** k
                            - (1 - success_probability - false_high_rate) ** k))
    return dict(synthetic=True, seed=seed, trials=trials, success_probability=success_probability,
                false_high_probability_given_incorrect=false_high_probability, records=records)


if __name__ == "__main__":
    print(json.dumps(verifier_experiment(), indent=2))

```
## compression.py
[对应文件](../../examples/compression.py)
```python
"""Small dequantized references; these do NOT pack bits or measure GPU speed."""
import json
import math
import torch


def symmetric_quantize(weight, bits=4, per_output_channel=False):
    if weight.ndim != 2 or not weight.numel() or not weight.is_floating_point() or not torch.isfinite(weight).all() or type(bits) is not int or not 2 <= bits <= 8:
        raise ValueError("finite floating matrix and bit width 2..8 required")
    qmax = 2 ** (bits - 1) - 1
    weight = weight.to(torch.float64 if weight.dtype == torch.float64 else torch.float32)
    maximum = weight.abs().amax(dim=1, keepdim=True) if per_output_channel else weight.abs().max()
    smallest = torch.nextafter(weight.new_tensor(0.), weight.new_tensor(1.))
    scale = torch.where(maximum > 0, (maximum / qmax).clamp_min(smallest), torch.ones_like(maximum))
    codes = (weight / scale).round().clamp(-qmax, qmax)
    limit = torch.finfo(weight.dtype).max
    return (codes * scale).clamp(-limit, limit), scale


def reconstruction_error(weight, approximation, inputs):
    if weight.shape != approximation.shape or weight.ndim != 2 or inputs.ndim != 2 or weight.shape[1] != inputs.shape[0]:
        raise ValueError("use W [out,in] and X [in,samples]")
    return ((approximation - weight) @ inputs).square().mean()


def low_rank_approximation(weight, rank):
    if weight.ndim != 2 or type(rank) is not int or not 1 <= rank <= min(weight.shape) or not torch.isfinite(weight).all():
        raise ValueError("rank outside matrix dimensions")
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank]


def input_weighted_low_rank(weight, covariance, rank):
    """Best rank-r layer approximation under a positive-definite input metric.

    Uses H=L L^T, takes truncated SVD of W L, then solves X L=(W L)_r.
    This is a local reconstruction optimum, not an end-to-end model optimum.
    """
    if weight.ndim != 2 or covariance.shape != (weight.shape[1], weight.shape[1]) \
            or not torch.isfinite(covariance).all() or not torch.allclose(covariance, covariance.T):
        raise ValueError("symmetric input covariance of shape [in,in] required")
    factor = torch.linalg.cholesky(covariance)
    truncated = low_rank_approximation(weight @ factor, rank)
    return torch.linalg.solve_triangular(factor.T, truncated.T, upper=True).T


def optimal_compensation(hessian, index, delta):
    """Continuous quadratic correction with one coordinate fixed; not full GPTQ."""
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1] or type(index) is not int or not 0 <= index < len(hessian) or not math.isfinite(delta):
        raise ValueError("square Hessian and valid coordinate required")
    if not torch.isfinite(hessian).all() or not torch.allclose(hessian, hessian.T):
        raise ValueError("finite symmetric Hessian required")
    torch.linalg.cholesky(hessian)  # Require positive definite geometry.
    unit = torch.zeros(len(hessian), dtype=hessian.dtype, device=hessian.device)
    unit[index] = 1
    direction = torch.linalg.solve(hessian, unit)
    return delta * direction / direction[index]


def demo():
    weight = torch.tensor([[.1, 1.], [.02, .08]], dtype=torch.float64)
    quantized, _ = symmetric_quantize(weight, bits=2)
    channel, _ = symmetric_quantize(weight, bits=2, per_output_channel=True)
    inputs = torch.tensor([[10., -10., 8., -8.], [.1, -.1, .2, -.2]], dtype=torch.float64)
    # Deliberate counterexample: smaller weight error can hurt important activations more.
    small_error = weight + torch.tensor([[.1, 0.], [0., 0.]], dtype=weight.dtype)
    large_error = weight + torch.tensor([[0., .2], [0., 0.]], dtype=weight.dtype)
    return dict(scope="synthetic matrices, dequantized arithmetic; no packed-memory or speed claim",
                tensor_quantization_output_mse=reconstruction_error(weight, quantized, inputs).item(),
                channel_quantization_output_mse=reconstruction_error(weight, channel, inputs).item(),
                smaller_weight_mse=(small_error - weight).square().mean().item(),
                smaller_weight_error_output_mse=reconstruction_error(weight, small_error, inputs).item(),
                larger_weight_mse=(large_error - weight).square().mean().item(),
                larger_weight_error_output_mse=reconstruction_error(weight, large_error, inputs).item())


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))

```
## interpretability.py
[对应文件](../../examples/interpretability.py)
```python
"""Temporary activation-patching hooks and an explicitly controlled toy example."""
from contextlib import contextmanager
import json
import torch
from torch import nn


@contextmanager
def patch_output(module, replacement, mask):
    """Replace selected output entries; always remove the hook, including on exceptions."""
    replacement = replacement.detach().clone()
    mask = mask.bool().clone()

    def hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.shape != replacement.shape or output.shape != mask.shape:
            raise ValueError("hook requires Tensor output and exact replacement/mask shapes")
        return torch.where(mask.to(output.device), replacement.to(device=output.device, dtype=output.dtype), output)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def recovery(clean, corrupt, patched, eps=1e-8):
    denominator = clean - corrupt
    if abs(denominator) < eps:
        raise ValueError("clean-corrupt metric gap is too small")
    return (patched - corrupt) / denominator


class ToyCircuit(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Identity()

    def forward(self, x):
        h = self.hidden(x)
        return h[:, 0] + 2 * h[:, 1]


def demo():
    model = ToyCircuit()
    clean, corrupt = torch.tensor([[1., 2., 50.]]), torch.tensor([[0., 2., -50.]])
    c, b = model(clean).item(), model(corrupt).item()
    with patch_output(model.hidden, clean, torch.tensor([[True, False, False]])):
        causal = model(corrupt).item()
    with patch_output(model.hidden, clean, torch.tensor([[False, False, True]])):
        irrelevant = model(corrupt).item()
    return dict(scope="known three-channel toy circuit, not an interpretation of a trained LM",
                clean=c, corrupt=b, causal_patch=causal, irrelevant_patch=irrelevant,
                recovery=recovery(c, b, causal), hook_cleanup_verified=model(corrupt).item() == b)


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))

```
## architectures.py
[对应文件](../../examples/architectures.py)
```python
"""Parameter and KV-cache accounting computed directly from open models' config.json.

The formulas follow the Hugging Face modeling code for each family. Totals are
compared with the safetensors metadata (sources/model-configs/index.json), so a
wrong formula shows up as a mismatch instead of a silently wrong table.

Covered: dense GQA (Qwen3, OLMo 2, SmolLM3, Mistral), sparse MoE without/with
shared experts (Qwen3-MoE, GLM-4.5), MLA + MoE + multi-token prediction
(DeepSeek-V3, Kimi K2), MoE with sinks and banded attention (gpt-oss), a
gated-DeltaNet/attention hybrid (Qwen3-Next) and a lightning-attention/softmax
hybrid MoE (MiniMax-M1).
"""
import json
import math
import operator
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "sources/model-configs"


def load(repo):
    index = json.loads((CONFIGS / "index.json").read_text(encoding="utf-8"))["models"][repo]
    return json.loads((CONFIGS / index["file"]).read_text(encoding="utf-8")), index


def head_dim(c):
    return c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]


def gqa_attention(c, bias=False, qk_norm=False, output_bias=True):
    """Q/K/V/O projections. GLM-4.5 has q/k/v biases but no o_proj bias; gpt-oss has all four."""
    d, h, kv, hd = c["hidden_size"], c["num_attention_heads"], c.get("num_key_value_heads") or c["num_attention_heads"], head_dim(c)
    weights = d * h * hd + 2 * d * kv * hd + h * hd * d
    if bias:
        weights += h * hd + 2 * kv * hd + (d if output_bias else 0)
    if qk_norm:
        weights += 2 * hd
    return weights


def mla_attention(c):
    """DeepSeek MLA: low-rank Q, a shared compressed KV latent plus a decoupled RoPE key."""
    d, h = c["hidden_size"], c["num_attention_heads"]
    rq, rkv = c["q_lora_rank"], c["kv_lora_rank"]
    nope, rope, vd = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
    q = d * rq + rq + rq * h * (nope + rope)
    kv = d * (rkv + rope) + rkv + rkv * h * (nope + vd)
    return q + kv + h * vd * d


def swiglu(d, width):
    return 3 * d * width


def gated_deltanet(c):
    """Qwen3-Next linear-attention layer (in_proj_qkvz, in_proj_ba, conv1d, A/dt, gated norm, out_proj)."""
    d = c["hidden_size"]
    hk, hv = c["linear_num_key_heads"], c["linear_num_value_heads"]
    dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
    key, value = hk * dk, hv * dv
    conv = (2 * key + value) * c["linear_conv_kernel_dim"]
    return d * (2 * key + 2 * value) + d * 2 * hv + conv + 2 * hv + dv + value * d


# Non-trainable tensors some checkpoints save alongside the weights, read from the safetensors
# headers: Kimi K2 stores a BF16 rotary inv_freq buffer of shape [56] in each of its 61 layers.
STORED_BUFFERS = {"moonshotai/Kimi-K2-Instruct": 61 * 56}


def mtp_module(c, block):
    """DeepSeek-style multi-token-prediction layer as stored in the checkpoint.

    eh_proj maps [norm(embedding); norm(hidden)] (2d) to d, plus enorm/hnorm and one
    transformer block. These checkpoints also store their own copy of the embedding
    and output head inside the MTP layer, which the safetensors total counts.
    """
    depth = c.get("num_nextn_predict_layers") or 0
    d, vocab = c["hidden_size"], c["vocab_size"]
    return depth * (2 * d * d + 2 * d + block + 2 * vocab * d + d)


def count(repo):
    c, index = load(repo)
    kind = c["model_type"]
    d, layers, vocab = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
    embed = vocab * d * (1 if c.get("tie_word_embeddings") else 2)
    norms = 2 * d
    parts = dict(embeddings=embed, final_norm=d)
    active_layers = 0
    if kind in ("qwen3", "olmo2", "smollm3", "mistral"):
        qk_norm = {"qwen3": 2 * head_dim(c), "olmo2": d + (c.get("num_key_value_heads") or c["num_attention_heads"]) * head_dim(c)}.get(kind, 0)
        per = gqa_attention(c) + qk_norm + swiglu(d, c["intermediate_size"]) + norms
        parts["layers"] = layers * per
        active_layers = parts["layers"]
    elif kind in ("qwen3_moe", "glm4_moe"):
        e, k = c.get("num_experts") or c["n_routed_experts"], c["num_experts_per_tok"]
        shared = c.get("n_shared_experts") or 0
        dense_first = c.get("first_k_dense_replace") or 0
        attn = gqa_attention(c, bias=c.get("attention_bias", False), output_bias=False) \
            + (2 * head_dim(c) if kind == "qwen3_moe" or c.get("use_qk_norm") else 0)
        expert = swiglu(d, c["moe_intermediate_size"])
        router = d * e + (e if kind == "glm4_moe" else 0)
        moe_layer = attn + norms + router + (e + shared) * expert
        dense_layer = attn + norms + swiglu(d, c["intermediate_size"])
        parts["layers"] = dense_first * dense_layer + (layers - dense_first) * moe_layer
        active_layers = dense_first * dense_layer + (layers - dense_first) * (attn + norms + router + (k + shared) * expert)
        parts["mtp"] = mtp_module(c, moe_layer)
    elif kind in ("deepseek_v3", "kimi_k2"):
        e, k, shared = c["n_routed_experts"], c["num_experts_per_tok"], c["n_shared_experts"]
        dense_first = c["first_k_dense_replace"]
        attn = mla_attention(c)
        expert = swiglu(d, c["moe_intermediate_size"])
        router = d * e + e  # gate weight + score-correction bias used for balancing
        moe_layer = attn + norms + router + (e + shared) * expert
        dense_layer = attn + norms + swiglu(d, c["intermediate_size"])
        parts["layers"] = dense_first * dense_layer + (layers - dense_first) * moe_layer
        active_layers = dense_first * dense_layer + (layers - dense_first) * (attn + norms + router + (k + shared) * expert)
        parts["mtp"] = mtp_module(c, moe_layer)
    elif kind == "gpt_oss":
        e, k, width = c["num_local_experts"], c["num_experts_per_tok"], c["intermediate_size"]
        attn = gqa_attention(c, bias=True) + c["num_attention_heads"]  # + one learned sink logit per head
        expert = 3 * d * width + 2 * width + d  # fused gate/up (with bias) and down (with bias)
        router = d * e + e
        parts["layers"] = layers * (attn + norms + router + e * expert)
        active_layers = layers * (attn + norms + router + k * expert)
    elif kind == "qwen3_next":
        e, k = c["num_experts"], c["num_experts_per_tok"]
        interval = c["full_attention_interval"]
        full = layers // interval
        attn = gqa_attention(c) + 2 * head_dim(c)
        # Qwen3-Next gates the attention output: q_proj produces an extra head_dim per query head.
        attn += d * c["num_attention_heads"] * head_dim(c)
        expert = swiglu(d, c["moe_intermediate_size"])
        moe = d * e + e * expert + swiglu(d, c["shared_expert_intermediate_size"]) + d
        parts["layers"] = full * attn + (layers - full) * gated_deltanet(c) + layers * (moe + norms)
        # The checkpoint carries one MTP block (not listed in config.json): fc on [embed; hidden],
        # three norms, one full-attention layer and one MoE layer.
        parts["mtp"] = 2 * d * d + 3 * d + attn + moe + norms
        active_moe = d * e + k * expert + swiglu(d, c["shared_expert_intermediate_size"]) + d
        active_layers = full * attn + (layers - full) * gated_deltanet(c) + layers * (active_moe + norms)
    elif kind == "minimax_m1":
        e, k = c["num_local_experts"], c["num_experts_per_tok"]
        full = sum(c["attn_type_list"])
        attn = gqa_attention(c)
        h, hd = c["num_attention_heads"], head_dim(c)
        # Lightning (linear) attention: fused qkv, a sigmoid output gate, out_proj and a per-channel norm.
        lightning = d * 3 * h * hd + d * h * hd + h * hd * d + h * hd
        expert = swiglu(d, c["intermediate_size"])
        router = d * e  # stored in FP32: 80 x 6144 x 32 = 15,728,640, the F32 count in the metadata
        parts["layers"] = full * attn + (layers - full) * lightning + layers * (router + e * expert + norms)
        active_layers = full * attn + (layers - full) * lightning + layers * (router + k * expert + norms)
    else:
        raise ValueError(f"no formula for {kind}")
    if repo in STORED_BUFFERS:
        parts["stored_buffers"] = STORED_BUFFERS[repo]
    total = sum(parts.values())
    reported = index["safetensors_total"]
    # Active parameters per token, using the gpt-oss model card's definition: every layer's
    # attention, the experts actually routed to, and the unembedding; the input embedding is
    # a table lookup, not a matmul, so it is not counted. MTP modules are excluded (dropped at
    # inference or used only as a speculative-decoding draft).
    active = active_layers + vocab * d + d
    return dict(repo=repo, model_type=kind, total=total, reported=reported,
                relative_gap=(total - reported) / reported if reported else None,
                total_without_mtp=total - parts.get("mtp", 0) - parts.get("stored_buffers", 0), active=active, parts=parts)


def kv_bytes_per_token(repo, context=None, bytes_per_value=2):
    """Per-token cache across all layers, honoring MLA latents, sliding windows, NoPE and linear layers."""
    if context is not None:
        try:
            # accepts numpy integers; bool is an int subclass, so reject it before converting
            context = None if isinstance(context, bool) else operator.index(context)
        except TypeError:
            context = None
        if context is None or context < 1:
            raise ValueError("context must be None or a positive integer")
    if not math.isfinite(bytes_per_value) or bytes_per_value <= 0:
        raise ValueError("positive finite bytes per KV value required")
    c, _ = load(repo)
    kind, layers = c["model_type"], c["num_hidden_layers"]
    if kind in ("deepseek_v3", "kimi_k2"):
        per_layer = c["kv_lora_rank"] + c["qk_rope_head_dim"]
        return dict(bytes=layers * per_layer * bytes_per_value, note="MLA latent + decoupled RoPE key")
    kv, hd = c.get("num_key_value_heads") or c["num_attention_heads"], head_dim(c)
    per_layer = 2 * kv * hd * bytes_per_value
    if kind == "qwen3_next":
        full = layers // c["full_attention_interval"]
        return dict(bytes=full * per_layer, note=f"{full} of {layers} layers keep KV; DeltaNet layers hold a fixed state")
    if kind == "gpt_oss":
        window = c["sliding_window"]
        sliding = sum(t == "sliding_attention" for t in c["layer_types"])
        full = layers - sliding
        if context is None:
            # Long-context limit: banded layers contribute nothing per additional token.
            return dict(bytes=full * per_layer, note=f"{sliding} banded layers cap at {window} tokens; long-context limit")
        # Average over a context: banded layers keep at most `window` tokens.
        return dict(bytes=(full * per_layer * context + sliding * per_layer * min(context, window)) / context,
                    note=f"{sliding} banded layers cap at {window} tokens")
    if kind == "minimax_m1":
        full = sum(c["attn_type_list"])
        return dict(bytes=full * per_layer, note=f"{full} of {layers} layers are softmax attention")
    return dict(bytes=layers * per_layer, note="all layers full attention")


MODELS = ["allenai/OLMo-2-0425-1B", "HuggingFaceTB/SmolLM3-3B", "allenai/OLMo-2-1124-7B", "Qwen/Qwen3-8B",
          "mistralai/Mistral-Small-24B-Base-2501", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B", "zai-org/GLM-4.5",
          "openai/gpt-oss-20b", "openai/gpt-oss-120b", "Qwen/Qwen3-Next-80B-A3B-Instruct",
          "MiniMaxAI/MiniMax-M1-80k", "deepseek-ai/DeepSeek-V3", "moonshotai/Kimi-K2-Instruct"]


def report(context=32768):
    rows = []
    for repo in MODELS:
        counted = count(repo)
        kv = kv_bytes_per_token(repo, context)
        rows.append(dict(repo=repo, model_type=counted["model_type"], total_b=round(counted["total"] / 1e9, 3),
                         reported_b=round(counted["reported"] / 1e9, 3), gap=round(counted["relative_gap"], 5),
                         without_mtp_b=round(counted["total_without_mtp"] / 1e9, 2),
                         active_b=round(counted["active"] / 1e9, 2),
                         kv_kib_per_token=round(kv["bytes"] / 1024, 1),
                         kv_gib_at_context=round(kv["bytes"] * context / 2**30, 2), kv_note=kv["note"]))
    return dict(context=context, bytes_per_value=2, rows=rows)


if __name__ == "__main__":
    result = report()
    print(f"{'model':42s} {'counted':>9s} {'reported':>9s} {'gap':>8s} {'active':>7s} {'KV KiB/tok':>10s} {'KV@32K':>8s}")
    for r in result["rows"]:
        print(f"{r['repo']:42s} {r['total_b']:9.3f} {r['reported_b']:9.3f} {r['gap']:+8.4f} {r['active_b']:7.2f} "
              f"{r['kv_kib_per_token']:10.1f} {r['kv_gib_at_context']:8.2f}  {r['kv_note']}")

```
## long_context.py
[对应文件](../../examples/long_context.py)
```python
"""Long-context mechanics: RoPE scaling (PI, NTK-aware, YaRN), attention sinks, sliding windows.

* yarn_inverse_frequencies reproduces the frequency blend used by DeepSeek-V3,
  gpt-oss, Kimi K2 and Qwen long-context variants; tests/test_frontier.py compares it
  with transformers on the pinned configs (gpt-oss needs truncate=False, as its config says).
* yarn_attention_factor is the factor applied to both q and k (cos and sin are scaled),
  so attention logits are multiplied by its square.
* rotation_wavelengths shows which RoPE dimensions complete a full turn inside
  the training context: those are left alone, the slow ones are interpolated.
* sink_attention adds one learned logit per head to the softmax denominator, as in
  gpt-oss, so a head can put (almost) no weight on any real token.
* streaming_kv_positions is the StreamingLLM cache policy: keep the first few
  "sink" tokens plus a recent window.
"""
import json
import math
import torch


def rope_inverse_frequencies(dim, base):
    if type(dim) is not int or dim < 2 or dim % 2 or not math.isfinite(base) or base <= 0:
        raise ValueError("positive even rotary dimension and finite positive base required")
    return 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)


def position_interpolation(dim, base, factor):
    """PI: divide every frequency by the extension factor (equivalently, positions by factor)."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("positive finite interpolation factor required")
    return rope_inverse_frequencies(dim, base) / factor


def ntk_aware(dim, base, factor):
    """NTK-aware scaling: enlarge the base so the slowest frequency stretches by `factor`."""
    if dim <= 2 or not math.isfinite(factor) or factor <= 0:
        raise ValueError("NTK-aware scaling requires dim > 2 and positive finite factor")
    return rope_inverse_frequencies(dim, base * factor ** (dim / (dim - 2)))


def yarn_correction_range(dim, base, original_context, beta_fast=32, beta_slow=1, truncate=True):
    """Dimension indices where a frequency completes beta_fast and beta_slow turns in the original context."""
    rope_inverse_frequencies(dim, base)
    if base <= 1 or not math.isfinite(original_context) or original_context <= 0 \
            or not all(math.isfinite(x) and x > 0 for x in (beta_fast, beta_slow)) or beta_fast < beta_slow:
        raise ValueError("YaRN needs base > 1, positive context and beta_fast >= beta_slow > 0")
    def dim_for(turns):
        return dim * math.log(original_context / (turns * 2 * math.pi)) / (2 * math.log(base))
    low, high = dim_for(beta_fast), dim_for(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def yarn_mscale(factor, mscale=1.0):
    if not math.isfinite(factor) or factor <= 0 or not math.isfinite(mscale) or mscale < 0:
        raise ValueError("finite positive factor and nonnegative mscale required")
    return 1.0 if factor <= 1 else 0.1 * mscale * math.log(factor) + 1.0


def yarn_attention_factor(factor, mscale=None, mscale_all_dim=None):
    """Temperature correction applied to q and k (logits get its square). DeepSeek-style configs give mscale and mscale_all_dim, and the
    factor becomes their ratio (1.0 when both are 1, as in DeepSeek-V3 and Kimi K2, which instead
    fold the correction into the softmax scale of MLA)."""
    if mscale is not None and mscale_all_dim is not None:
        return yarn_mscale(factor, mscale) / yarn_mscale(factor, mscale_all_dim)
    return yarn_mscale(factor)


def yarn_inverse_frequencies(dim, base, factor, original_context, beta_fast=32, beta_slow=1, truncate=True,
                             mscale=None, mscale_all_dim=None):
    """Fast-rotating dims keep their frequency, slow dims are interpolated by `factor`, a linear ramp between.

    Returns (inverse frequencies, q/k temperature factor). Pass the config's mscale/mscale_all_dim
    for DeepSeek-style models: their factor is a ratio (1.0 for DeepSeek-V3 and Kimi K2).
    """
    extrapolation = rope_inverse_frequencies(dim, base)
    interpolation = extrapolation / factor
    low, high = yarn_correction_range(dim, base, original_context, beta_fast, beta_slow, truncate)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    keep = 1 - ramp
    return interpolation * (1 - keep) + extrapolation * keep, yarn_attention_factor(factor, mscale, mscale_all_dim)


def rotation_wavelengths(dim, base):
    """Tokens needed for each RoPE pair to complete one full rotation."""
    return 2 * math.pi / rope_inverse_frequencies(dim, base)


def sink_attention(q, k, v, sink_logits, causal=True):
    """Softmax over [scores, sink]; the sink column is dropped after normalization.

    q, k, v: [heads, T, d]; sink_logits: [heads]. Returns output and the attention mass on real tokens.
    """
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        t = q.shape[-2]
        if k.shape[-2] != t:
            raise ValueError("causal sink reference requires equal query/key lengths")
        scores = scores.masked_fill(torch.ones(t, t, dtype=torch.bool, device=q.device).triu(1), -torch.inf)
    sink = sink_logits[:, None, None].expand(-1, scores.shape[1], 1)
    probs = torch.cat([scores, sink], dim=-1).softmax(-1)[..., :-1]
    return probs @ v, probs.sum(-1)


def streaming_kv_positions(length, sinks=4, window=1024):
    """Positions kept by a StreamingLLM-style cache after `length` tokens."""
    if any(type(value) is not int or value < 0 for value in (length, sinks, window)):
        raise ValueError("nonnegative integer length, sink count and window required")
    if length <= sinks + window:
        return list(range(length))
    return list(range(sinks)) + list(range(length - window, length))


def demo():
    dim, base, original, factor = 128, 10000.0, 4096, 32  # e.g. a 4K-pretrained model extended to 128K
    wavelengths = rotation_wavelengths(dim, base)
    low, high = yarn_correction_range(dim, base, original)
    inv_yarn, mscale = yarn_inverse_frequencies(dim, base, factor, original)
    inv_rope = rope_inverse_frequencies(dim, base)
    ratio = (inv_rope / inv_yarn).tolist()
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 6, 8) for _ in range(3))
    _, mass_open = sink_attention(q, k, v, torch.tensor([-30.0, -30.0]))
    _, mass_sink = sink_attention(q, k, v, torch.tensor([8.0, 8.0]))
    return dict(
        setting=dict(head_dim=dim, base=base, original_context=original, factor=factor),
        shortest_and_longest_wavelength=[round(wavelengths[0].item(), 2), round(wavelengths[-1].item(), 1)],
        pairs_completing_a_turn_in_original_context=int((wavelengths < original).sum()),
        yarn_ramp_dims=[low, high], yarn_attention_scale=round(mscale, 4),
        frequency_divisor_first_mid_last=[round(ratio[0], 3), round(ratio[len(ratio) // 2], 3), round(ratio[-1], 3)],
        sink_mass_on_tokens=dict(sink_logit_minus30=mass_open.mean().item(), sink_logit_8=round(mass_sink.mean().item(), 4)),
        streaming_cache_size_at_100k=len(streaming_kv_positions(100_000)))


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))

```
## rl_objectives.py
[对应文件](../../examples/rl_objectives.py)
```python
"""Policy objectives used by recent reasoning-model RL recipes, on the same tensor contract as posttrain.py.

All losses take per-token log-probs [B,T], detached old log-probs [B,T], per-sequence
advantages [B] and a 0/1 response mask [B,T], and return a scalar to minimize.

* dapo_loss: decoupled clip (eps_low, eps_high) and token-level aggregation over the
  whole batch (DAPO, arXiv 2503.14476).
* cispo_loss: clip the importance weight itself and stop its gradient; exceeding
  the upper bound alone no longer zeros the REINFORCE term (MiniMax-M1, arXiv 2506.13585).
* dynamic_sampling_mask: drop prompt groups whose rewards are all equal.
* overlong_penalty: DAPO's soft length penalty for truncated responses.
* clipped_token_fraction: share of tokens whose gradient a PPO-style clip removes.
"""
import math
import torch
from .posttrain import masked_log_ratio, finite_ratio, check_advantage


def token_level_mean(values, valid):
    """Sum over every valid token in the batch, divided by the total valid-token count."""
    return values.masked_fill(~valid, 0).sum() / valid.sum()


def clipping_bounds(eps_low, eps_high):
    if not math.isfinite(eps_low) or not math.isfinite(eps_high) or not 0 <= eps_low < 1 or eps_high < 0:
        raise ValueError("finite 0 <= eps_low < 1 and eps_high >= 0 required")
    return 1 - eps_low, 1 + eps_high


def dapo_loss(logp, old_logp, advantage, mask, eps_low=0.2, eps_high=0.28):
    log_ratio, valid = masked_log_ratio(logp, old_logp, mask)
    check_advantage(advantage, len(logp), max(eps_low, 0))
    low, high = clipping_bounds(eps_low, eps_high)
    ratio = finite_ratio(log_ratio)
    a = advantage.detach()[:, None]
    objective = torch.minimum(ratio * a, ratio.clamp(low, high) * a)
    return -token_level_mean(objective, valid)


def cispo_loss(logp, old_logp, advantage, mask, eps_high=2.0, eps_low=None):
    """sg(clip(r)) * A * log pi; upper clipping alone does not zero the term's gradient."""
    log_ratio, valid = masked_log_ratio(logp, old_logp, mask)
    check_advantage(advantage, len(logp), 0)
    ratio = finite_ratio(log_ratio)
    low, high = clipping_bounds(0.0 if eps_low is None else eps_low, eps_high)
    low = 0.0 if eps_low is None else low
    weight = ratio.clamp(low, high).detach()
    current = logp.masked_fill(~valid, 0)
    return -token_level_mean(weight * advantage.detach()[:, None] * current, valid)


def clipped_token_fraction(logp, old_logp, advantage, mask, eps_low=0.2, eps_high=0.2):
    """Tokens where the clipped branch is active and therefore receives zero gradient."""
    low, high = clipping_bounds(eps_low, eps_high)
    check_advantage(advantage, len(logp), eps_low)
    log_ratio, valid = masked_log_ratio(logp, old_logp, mask)
    ratio = finite_ratio(log_ratio)
    a = advantage[:, None]
    clipped = ((a > 0) & (ratio > high)) | ((a < 0) & (ratio < low))
    return (clipped & valid).sum().item() / valid.sum().item()


def dynamic_sampling_mask(rewards):
    """[prompts, group] rewards -> bool [prompts]: keep groups that contain different rewards."""
    if rewards.ndim != 2 or min(rewards.shape) < 1 or not torch.isfinite(rewards).all():
        raise ValueError("finite nonempty [prompts, group] rewards required")
    return rewards.max(-1).values != rewards.min(-1).values


def overlong_penalty(lengths, max_length, cache):
    """0 up to max_length - cache, then a linear ramp to -1 at max_length, and -1 beyond."""
    if not math.isfinite(cache) or not math.isfinite(max_length) or not 0 < cache <= max_length:
        raise ValueError("require 0 < cache <= max_length")
    lengths = torch.as_tensor(lengths, dtype=torch.float32)
    if not torch.isfinite(lengths).all() or torch.any(lengths < 0):
        raise ValueError("finite nonnegative lengths required")
    start = max_length - cache
    ramp = (start - lengths) / cache
    return torch.where(lengths <= start, torch.zeros_like(lengths), torch.where(lengths <= max_length, ramp, -torch.ones_like(lengths)))


def demo():
    torch.manual_seed(0)
    old = torch.log_softmax(torch.randn(4, 6), -1)
    # A drifted policy; the third token's ratio is e^0.22 = 1.246, between 1.2 and 1.28.
    logp = old + torch.tensor([[1.2, -0.1, 0.22, 0.0, -1.0, 0.3]] * 4)
    mask = torch.ones(4, 6)
    advantage = torch.tensor([1.0, -1.0, 1.0, -1.0])
    rewards = torch.tensor([[1., 1, 1, 1], [0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]])
    return dict(
        clipped_fraction_symmetric_0_2=clipped_token_fraction(logp, old, advantage, mask, 0.2, 0.2),
        clipped_fraction_dapo_0_2_0_28=clipped_token_fraction(logp, old, advantage, mask, 0.2, 0.28),
        groups_kept_by_dynamic_sampling=dynamic_sampling_mask(rewards).tolist(),
        overlong_penalty=overlong_penalty([14000, 16384, 18000, 20480, 21000], 20480, 4096).tolist())


if __name__ == "__main__":
    import json
    print(json.dumps(demo(), indent=2))

```
## agents.py
[对应文件](../../examples/agents.py)
```python
"""Agent trajectories as training data: segments, loss masks and verifiable rewards.

An agent episode alternates model-generated text (reasoning, tool calls, final
answer) with environment text (tool results). Only model-generated tokens are
actions, so only they enter the policy-gradient or SFT loss. This module builds
that mask for a toy calculator environment, checks it against a byte-level
tokenization, and scores episodes with an outcome reward plus a format check.
"""
import ast
from fractions import Fraction
import json
import operator
import re

TOOL_CALL = re.compile(r"<tool>(.*?)</tool>", re.S)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculator(expression, max_digits=30, max_length=200):
    """The environment: + - * / on integers via an AST walk (no eval, no powers), result or error string.

    Model output is untrusted: overly long input is rejected before parsing (deep nesting would
    otherwise exhaust the recursion limit), and every constant and result is bounded in size.
    Arithmetic is exact (Fraction), so large integer quotients are not rounded through float.
    """
    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            if abs(node.value) >= 10 ** max_digits:
                raise OverflowError
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -value(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            result = OPERATORS[type(node.op)](value(node.left), value(node.right))
            if abs(result) >= 10 ** max_digits:
                raise OverflowError
            return result
        raise ValueError
    if len(expression) > max_length:
        return "error: expression too long"
    try:
        result = value(ast.parse(expression.strip(), mode="eval").body)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return "error: unsupported expression"
    except (ZeroDivisionError, OverflowError) as exc:
        return f"error: {type(exc).__name__}"
    return str(result.numerator) if result.denominator == 1 else f"{float(result):.6g}"


def run_episode(policy_turns, max_calls=4):
    """policy_turns: the model's successive messages. Returns segments [(role, text)]."""
    segments = []
    for turn in policy_turns:
        segments.append(("model", turn))
        call = TOOL_CALL.search(turn)
        if ANSWER.search(turn) or call is None:
            break
        if sum(role == "tool" for role, _ in segments) >= max_calls:
            break
        segments.append(("tool", f"<result>{calculator(call.group(1))}</result>"))
    return segments


def byte_tokens_and_mask(prompt, segments):
    """Byte-level token IDs for the whole episode and a 0/1 mask that is 1 only on model bytes."""
    ids, mask = list(prompt.encode()), [0] * len(prompt.encode())
    for role, text in segments:
        data = text.encode()
        ids.extend(data)
        mask.extend([1 if role == "model" else 0] * len(data))
    return ids, mask


def reward(segments, target):
    """Outcome reward 1 for a correct final answer; format reward checks the answer tag exists."""
    final = segments[-1][1] if segments and segments[-1][0] == "model" else ""
    match = ANSWER.search(final)
    formatted = match is not None
    correct = formatted and match.group(1).strip() == str(target)
    calls = sum(role == "tool" for role, _ in segments)
    errors = sum(role == "tool" and "error" in text for role, text in segments)
    return dict(reward=float(correct), format=float(formatted), tool_calls=calls, tool_errors=errors)


def demo():
    prompt = "Q: What is (17 * 23) + 389? Use the calculator.\n"
    good = ["<tool>17 * 23</tool>", "391 so far. <tool>391 + 389</tool>", "<answer>780</answer>"]
    sloppy = ["<tool>17 x 23</tool>", "<answer>781</answer>"]
    results = {}
    for name, turns in [("good", good), ("sloppy", sloppy)]:
        segments = run_episode(turns)
        ids, mask = byte_tokens_and_mask(prompt, segments)
        results[name] = dict(segments=segments, tokens=len(ids), loss_tokens=sum(mask),
                             tool_tokens=sum(len(t.encode()) for r, t in segments if r == "tool"), **reward(segments, 780))
    return results


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))

```
## synthetic_data.py
[对应文件](../../examples/synthetic_data.py)
```python
"""Synthetic data and distillation on the arithmetic toy, with a known-correct verifier.

Teacher: the 1000-step model from chapter 20 (100% exact answers when greedy).
Student: a fresh, smaller TinyLM. Four ways to train the student on the same
budget of answer tokens are compared. Every arm uses the same loss structure and source
prompt pool: cross-entropy on the three next-byte targets inside the prompt plus a loss on the answer
byte. Filtering changes prompt frequencies, so its sampled prompt sequence is not identical:

* real: ground-truth "a+b=c" rows (the oracle data source);
* teacher_sampled: rows whose answers the teacher generates at temperature T;
* teacher_filtered: the same samples kept only if a verifier marks them correct
  (rejection sampling, the core of most synthetic-data pipelines);
* soft_kd: token-level KL to the teacher's full answer distribution at the same temperature T,
  so it carries the same information as the samples, without the sampling noise.

The point is measurable: unverified sampling at high temperature injects wrong
answers, and filtering removes them. Every arm keeps the answer-loss coefficient
fixed; otherwise loss weighting would confound the comparison of teacher signals.
"""
import contextlib
import io
import json
import math
import random
import tempfile
import torch
from torch.nn import functional as F
from .model import Config, TinyLM, generate
from .optim import AdamW, cosine_lr
from .train import run

FACTS = [(a, b) for a in range(10) for b in range(10)]
STUDENT = dict(width=48, layers=2, heads=4, kv_heads=2, hidden=128)


def teacher(steps=1000, seed=7):
    with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
        run(steps=steps, out=folder, seed=seed)
        saved = torch.load(f"{folder}/checkpoint.pt", weights_only=True)
    model = TinyLM(Config(**saved["config"]))
    model.load_state_dict(saved["model"])
    return model.eval()


@torch.no_grad()
def exact_accuracy(model, facts=FACTS):
    prompts = torch.tensor([list(f"{a}+{b}=".encode()) for a, b in facts])
    answers = model(prompts)[:, -1].argmax(-1)
    targets = torch.tensor([ord(str((a + b) % 10)) for a, b in facts])
    return (answers == targets).float().mean().item()


@torch.no_grad()
def teacher_rows(model, facts, temperature, generator):
    """Teacher completions as raw 6-byte rows (latin-1 keeps every byte a single character)."""
    prompts = torch.tensor([list(f"{a}+{b}=".encode()) for a, b in facts])
    out = generate(model, prompts, 2, temperature=temperature, generator=generator)
    return [bytes(row.tolist()).decode("latin-1") for row in out]


DIGITS = "0123456789"


def is_correct(row):
    # ASCII digits only: rows are latin-1 decoded, and str.isdigit() also accepts "²", which int() rejects.
    return len(row) == 6 and row[0] in DIGITS and row[1] == "+" and row[2] in DIGITS and row[3] == "=" \
        and row[4] in DIGITS and row[5] == "\n" and int(row[4]) == (int(row[0]) + int(row[2])) % 10


def train_student(rows, teacher_model=None, kd_temperature=1.0, steps=300, seed=0, lr=3e-3, batch=32):
    """Train on 6-byte rows. The loss is CE on '+', the second operand, and '=' plus a loss on the
    answer byte: CE with the row's answer (hard target), or, if teacher_model is given, KL to
    the teacher's answer distribution softened by kd_temperature (the row's answer is unused)."""
    if not rows or any(not isinstance(row,str) or len(row.encode('latin-1')) != 6 for row in rows):
        raise ValueError('nonempty six-byte training rows required')
    if type(steps) is not int or type(batch) is not int or min(steps,batch) < 1 \
            or not math.isfinite(kd_temperature) or kd_temperature <= 0:
        raise ValueError('positive integer steps/batch and finite positive teacher temperature required')
    torch.manual_seed(seed)
    student = TinyLM(Config(**STUDENT))
    opt = AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = cosine_lr(step, lr, lr * 0.1, min(20, steps - 1), steps)
        ids = torch.tensor([list(rng.choice(rows).encode("latin-1")) for _ in range(batch)])
        prompts = ids[:, :4]
        logits = student(prompts)
        prefix_loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), prompts[:, 1:].reshape(-1))
        if teacher_model is None:
            answer_loss = F.cross_entropy(logits[:, -1], ids[:, 4])
        else:
            with torch.no_grad():
                target = (teacher_model(prompts)[:, -1] / kd_temperature).log_softmax(-1)
            answer_loss = F.kl_div(logits[:, -1].log_softmax(-1), target, log_target=True, reduction="batchmean")
        loss = prefix_loss + answer_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
    return exact_accuracy(student.eval())


def experiment(temperature=1.5, samples_per_fact=4, seeds=(0, 1, 2, 3, 4)):
    if not math.isfinite(temperature) or temperature <= 0 or type(samples_per_fact) is not int or samples_per_fact < 1 or not seeds:
        raise ValueError('positive temperature/sample count and at least one seed required')
    model = teacher()
    g = torch.Generator().manual_seed(0)
    facts = [f for f in FACTS for _ in range(samples_per_fact)]
    sampled = teacher_rows(model, facts, temperature, g)
    filtered = [r for r in sampled if is_correct(r)]
    if not filtered:
        raise ValueError('the teacher produced no accepted data; cannot train the filtered-data baseline')
    real = [f"{a}+{b}={(a + b) % 10}\n" for a, b in facts]
    well_formed = sum(len(r) == 6 and r[4] in DIGITS and r[5] == "\n" for r in sampled)
    # Rejection sampling keeps more samples of facts the teacher already finds easy.
    kept = [sum(f == g and is_correct(r) for g, r in zip(facts, sampled)) for f in FACTS]
    results = dict(teacher_accuracy=exact_accuracy(model), temperature=temperature,
                   sampled_rows=len(sampled), sampled_well_formed_fraction=well_formed / len(sampled),
                   sampled_correct_fraction=len(filtered) / len(sampled), filtered_rows=len(filtered),
                   facts_by_kept_samples={k: kept.count(k) for k in range(samples_per_fact + 1)}, students={})
    for name, kwargs in [("real", dict(rows=real)), ("teacher_sampled", dict(rows=sampled)),
                         ("teacher_filtered", dict(rows=filtered)), ("soft_kd", dict(rows=real, teacher_model=model, kd_temperature=temperature))]:
        results["students"][name] = [round(train_student(seed=s, **kwargs), 3) for s in seeds]
    return results


if __name__ == "__main__":
    print(json.dumps(experiment(), indent=2))

```
## low_precision.py
[对应文件](../../examples/low_precision.py)
```python
"""Block-scaled low precision: MXFP4 (OCP microscaling) and DeepSeek-V3-style FP8 tiles.

* mxfp4_quantize follows Algorithm 1 of the MX paper (arXiv 2310.10537): blocks of 32
  values share a power-of-two scale X = 2^(floor(log2 max|v|) - emax_elem); elements
  are rounded to the nearest FP4 E2M1 value {0, 0.5, 1, 1.5, 2, 3, 4, 6} with clamping.
  Storage: 4 bits per element + 8 bits per 32 elements = 4.25 bits, the figure the
  gpt-oss card quotes for its MoE weights.
* fp8_tile_quantize uses torch.float8_e4m3fn with one FP32 scale per 1x128 activation
  tile or per 128x128 weight block (the DeepSeek-V3 recipe).

Everything is dequantized back to float to measure error; no packed kernels.
"""
import json
import operator
import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_EMAX = 2          # largest normal 6 = 1.5 * 2^2
E4M3_MAX = 448.0


def _finite_float(values):
    if not values.is_floating_point() or not values.numel() or not torch.isfinite(values).all():
        raise ValueError("nonempty finite floating-point values required")


def _fp32_blocks(values, block):
    _finite_float(values)
    try:
        block = operator.index(block)
    except TypeError as error:
        raise ValueError("positive integer block size required") from error
    if values.ndim < 1 or block < 1 or values.shape[-1] % block:
        raise ValueError("last dimension must be a multiple of a positive block size")
    result = values.reshape(*values.shape[:-1], -1, block).float()
    if not torch.isfinite(result).all():
        raise ValueError("values overflow the FP32 working dtype")
    return result


def _absmax_scale(amax, limit):
    # Handle true zero blocks separately. An arbitrary floor would erase small nonzero blocks.
    smallest = torch.nextafter(amax.new_tensor(0.), amax.new_tensor(1.))
    return torch.where(amax > 0, (amax / limit).clamp_min(smallest), torch.ones_like(amax))


def round_to_e2m1(x):
    """Nearest E2M1 value for finite inputs; exact ties choose the even code, sign kept.

    Accumulates and returns FP64 for FP64 input, otherwise FP32, on the input device.
    """
    _finite_float(x)
    work = x.to(torch.float64 if x.dtype == torch.float64 else torch.float32)
    grid = E2M1.to(device=x.device, dtype=work.dtype)
    magnitude = work.abs().clamp(max=6.0)
    distance = (magnitude[..., None] - grid).abs()
    best = distance.argmin(-1)
    # Prefer the even code index on exact ties, as round-half-to-even would.
    tie = distance == distance.gather(-1, best[..., None])
    codes = torch.arange(len(grid), device=x.device)
    even_choice = torch.where(tie & (codes % 2 == 0), codes, len(grid)).min(-1).values
    best = torch.where(tie.sum(-1) > 1, even_choice, best)
    return torch.copysign(grid[best], work)


def mxfp4_quantize(values, block=32):
    """Finite-input reference: FP32 output with E8M0 scales in [2^-127, 2^127].

    Zero blocks use scale 1; reject NaN/Inf instead of modeling their packed encodings.
    """
    v = _fp32_blocks(values, block)
    amax = v.abs().amax(-1, keepdim=True)
    # frexp gives exact floor(log2(amax)); log2 can round a value just below a power of two up.
    _, exponent = torch.frexp(amax)
    shared = (exponent - 1 - E2M1_EMAX).clamp(-127, 127)
    scale = torch.where(amax > 0, torch.exp2(shared.to(v.dtype)), torch.ones_like(amax))
    dequantized = round_to_e2m1(v / scale) * scale
    return dequantized.reshape(values.shape), scale.squeeze(-1)


def fp8_tile_quantize(values, tile):
    """Per-tile absmax scaling into float8_e4m3fn; tile is (rows, cols).

    Returns FP32 reconstruction and a [number_of_row_tiles, number_of_column_tiles] scale grid.
    Zero blocks use scale 1; a nonzero scale is bounded only by FP32 representability.
    """
    _finite_float(values)
    if values.ndim != 2 or len(tile) != 2:
        raise ValueError("a matrix and a two-dimensional tile required")
    try:
        r, c = (operator.index(size) for size in tile)
    except TypeError as error:
        raise ValueError("positive integer tile dimensions required") from error
    rows, cols = values.shape
    if min(r, c) < 1 or rows % r or cols % c:
        raise ValueError("matrix shape must be a multiple of the tile shape")
    blocks = values.float().reshape(rows // r, r, cols // c, c)
    if not torch.isfinite(blocks).all():
        raise ValueError("values overflow the FP32 working dtype")
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True)
    scale = _absmax_scale(amax, E4M3_MAX)
    q = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    limit = torch.finfo(torch.float32).max
    output = (q.float() * scale).clamp(-limit, limit)
    return output.reshape(rows, cols), scale.squeeze(3).squeeze(1)


def relative_error(approx, exact):
    """Relative L2 error; exact zero reconstructed as zero scores 0, otherwise infinity."""
    _finite_float(approx)
    _finite_float(exact)
    if approx.shape != exact.shape or approx.device != exact.device:
        raise ValueError("matching shapes and devices required")
    approx, exact = approx.double(), exact.double()
    scale = torch.maximum(approx.abs().amax(), exact.abs().amax())
    if scale == 0:
        return 0.0
    denominator = (exact / scale).norm()
    return float('inf') if denominator == 0 else (((approx / scale) - (exact / scale)).norm() / denominator).item()


def int4_absmax(values, block=32):
    """Symmetric INT4 with a free FP scale per block, for comparison with MXFP4's power-of-two scale."""
    v = _fp32_blocks(values, block)
    scale = _absmax_scale(v.abs().amax(-1, keepdim=True), 7)
    limit = torch.finfo(torch.float32).max
    return ((v / scale).round().clamp(-7, 7) * scale).clamp(-limit, limit).reshape(values.shape)


def demo():
    torch.manual_seed(0)
    weights = torch.randn(256, 512) * 0.02
    mx, scales = mxfp4_quantize(weights)
    tile_w, _ = fp8_tile_quantize(weights, (128, 128))
    bits = 4 + 8 / 32
    # Algorithm 1 floors the shared exponent, so the block maximum can exceed 6 and clamp.
    # Rounding the exponent up instead never clamps but wastes resolution on every block.
    blocks = weights.reshape(256, -1, 32)
    clamped = blocks.abs() / scales[..., None] > 6.0
    up = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1, keepdim=True) / 6.0)))
    no_clamp = (round_to_e2m1(blocks / up) * up).reshape(weights.shape)
    result = dict(
        mxfp4_bits_per_weight=bits,
        mxfp4_weight_relative_error=round(relative_error(mx, weights), 4),
        mxfp4_clamped_element_fraction=round(clamped.float().mean().item(), 4),
        mxfp4_blocks_with_clamping=round(clamped.any(-1).float().mean().item(), 4),
        mxfp4_ceil_exponent_relative_error=round(relative_error(no_clamp, weights), 4),
        int4_absmax_per_32_relative_error=round(relative_error(int4_absmax(weights), weights), 4),
        mxfp4_scales_are_powers_of_two=bool(torch.all(torch.log2(scales) == torch.round(torch.log2(scales)))),
        fp8_128x128_weight_relative_error=round(relative_error(tile_w, weights), 4),
        # Element counts from the gpt-oss-120b safetensors metadata (sources/model-configs/index.json):
        # the U8 entry is the 114.66B MXFP4 expert weights, the rest is BF16. The card reports 60.8 GiB.
        gpt_oss_120b_checkpoint_gib=round((114_661_785_600 * bits / 8 + 2_167_371_072 * 2) / 2**30, 1))
    # One activation channel far larger than the rest: compare a single scale with 1x128 tiles,
    # measuring error on the ordinary channels, which is where a shared scale does damage.
    normal = torch.ones(512, dtype=torch.bool)
    normal[7] = False
    for magnitude in (200.0, 1e5):
        activations = torch.randn(256, 512)
        activations[:, 7] *= magnitude
        one, _ = fp8_tile_quantize(activations, (256, 512))
        tiles, _ = fp8_tile_quantize(activations, (1, 128))
        result[f"outlier_x{magnitude:g}"] = dict(
            normal_channel_error_one_scale=round(relative_error(one[:, normal], activations[:, normal]), 4),
            normal_channel_error_1x128=round(relative_error(tiles[:, normal], activations[:, normal]), 4),
            flushed_to_zero_one_scale=round(((one[:, normal] == 0) & (activations[:, normal] != 0)).float().mean().item(), 4))
    return result


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))

```
## numerical_tools.py
[对应文件](../../examples/numerical_tools.py)
```python
"""Controlled numerical experiments for the 2026-10 review update.

No official assignment implementations or real-model performance claims.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F


def dpo_shared_parameter_demo(step=0.1):
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    def logps(t):
        common = F.logsigmoid(-2*t)
        return common+F.logsigmoid(t), common+F.logsigmoid(-t)
    chosen, rejected = logps(theta)
    loss = -F.logsigmoid((chosen-chosen.detach())-(rejected-rejected.detach()))
    gradient, = torch.autograd.grad(loss, theta)
    updated = theta.detach()-step*gradient
    a, b = logps(updated)
    return dict(theta_after=updated.item(), chosen_before=chosen.exp().item(),
                rejected_before=rejected.exp().item(), chosen_after=a.exp().item(),
                rejected_after=b.exp().item(), remaining_after=torch.sigmoid(2*updated).item(),
                ratio_after=(a-b).exp().item(), scope="one-parameter normalized toy sequence model")


def corrected_kl(logits, reference, sampling):
    """Exact finite-sum IS k3 check at a FIXED prefix; sampling/ref have full support."""
    if logits.ndim != 1 or reference.shape != logits.shape or sampling.shape != logits.shape:
        raise ValueError("matching vectors required")
    for distribution in (reference, sampling):
        if not torch.isfinite(distribution).all() or torch.any(distribution<=0) or not torch.isclose(distribution.sum(),distribution.new_tensor(1.)):
            raise ValueError("finite normalized distributions with positive support required")
    if not torch.isfinite(logits).all():
        raise ValueError("finite logits required")
    logp = logits.log_softmax(-1)
    p = logp.exp()
    q, old = reference.detach(), sampling.detach()
    # Algebraically identical to the fully differentiable IS-k3 sum. Cancel p
    # before division to avoid 0 * inf when a tiny probability underflows.
    return (q - p + p * (logp - q.log())).sum()


def entropy_logit_differential(logits, delta):
    """First-order entropy change at a fixed prefix for a specified logit perturbation."""
    if logits.ndim != 1 or delta.shape != logits.shape or not torch.isfinite(logits).all() or not torch.isfinite(delta).all():
        raise ValueError("matching finite vectors required")
    logp = logits.log_softmax(-1)
    p = logp.exp()
    entropy = -(p * logp).sum()
    return -(p * (logp + entropy) * delta).sum()


def chapter_review_demo():
    """Small counterexamples for chapter-by-chapter review; no new LM training."""
    from .model import Config, TinyLM
    from .posttrain import clipped_policy_loss
    difference = {}
    with torch.random.fork_rng():
        for depth in (1, 2):
            torch.manual_seed(0)
            model = TinyLM(Config(vocab=32, width=16, layers=depth, heads=2,
                                 kv_heads=2, hidden=32, position="none")).eval().double()
            with torch.no_grad():
                a, b = torch.tensor([[3, 5, 9]]), torch.tensor([[5, 3, 9]])
                difference[str(depth)] = (model(a)[0, -1] - model(b)[0, -1]).abs().max().item()
    # Each positive-advantage sample shares this single parameter. One finite step
    # crosses the clip boundary, which is not a projection constraint.
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    logp = F.logsigmoid(theta).reshape(1, 1)
    loss = clipped_policy_loss(logp, logp.detach(), torch.ones(1), torch.ones(1, 1))
    gradient, = torch.autograd.grad(loss, theta)
    updated = theta.detach() - 2 * gradient
    ratio_after = (F.logsigmoid(updated) - F.logsigmoid(theta.detach())).exp()
    z = torch.tensor([.3, -.5, .8], dtype=torch.float64, requires_grad=True)
    delta = torch.tensor([.1, -.4, .2], dtype=torch.float64)
    p, lp = z.softmax(-1), z.log_softmax(-1)
    h = -(p * lp).sum()
    derivative, = torch.autograd.grad(h, z)
    flops = 2 * 70e9 * 15e12
    gpu_hours = flops / (989e12 * .4) / 3600
    return dict(
        nope_last_logit_difference=difference,
        hybrid_state_ratios={str(t): .25 + .25 / (t / 8192) for t in (8192, 131072)},
        ppo=dict(ratio_after=ratio_after.item(), clip_boundary=1.2, parameter_step=updated.item()),
        entropy=dict(analytic=entropy_logit_differential(z, delta).item(), autograd=(derivative * delta).sum().item()),
        annotation_cost=dict(flops=flops, gpu_hours=gpu_hours, ratio_to_6000_hours=gpu_hours / 6000),
        scope="CPU counterexamples, fixed-prefix derivatives and ideal cost accounting; no GPU performance or new LM training",
    )


def chapter_invariants_demo():
    """Small references for the guide's normalization, stability and evaluation examples.

    These compare algebraic identities and counterexamples, not trained-model quality.
    """
    from .extensions import clip_loss
    from .long_context import rope_inverse_frequencies, yarn_inverse_frequencies
    from .optim import AdamW

    logits = torch.tensor([1000., 998., -20., 999.], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([[1., -1.], [2., 3.], [4., 0.], [-2., 5.]], dtype=torch.float64, requires_grad=True)
    normalizers = torch.stack([part.logsumexp(0) for part in logits.split(2)])
    local_outputs = torch.stack([part.softmax(0) @ value for part, value in zip(logits.split(2), values.split(2))])
    merged = normalizers.softmax(0) @ local_outputs
    dense = logits.softmax(0) @ values
    actual = torch.autograd.grad(merged.square().sum(), (logits, values))
    expected = torch.autograd.grad(dense.square().sum(), (logits, values))

    key = torch.tensor([2., 0.], dtype=torch.float64)
    state = torch.tensor([[.2, .3], [.4, -.1]], dtype=torch.float64)
    target = torch.tensor([1., -1.], dtype=torch.float64)
    residual = target - state.T @ key
    contractions = {}
    for rate in (.125, .25, .75):
        updated = state + rate * key[:, None] * residual[None, :]
        ratio = ((target - updated.T @ key).square().sum() / residual.square().sum()).item()
        contractions[str(rate)] = ratio

    images = torch.eye(2, dtype=torch.float64, requires_grad=True)
    texts = torch.tensor([[1., .3], [.4, 1.]], dtype=torch.float64)
    full = clip_loss(images, texts, temperature=1.)
    micro = sum(clip_loss(images[i:i+1], texts[i:i+1], temperature=1.) for i in range(2)) / 2

    probability = torch.tensor([.7, .3, 0., 0.], dtype=torch.float64)
    experts = {"uniform": 4 * (1 - (1 - .25) ** 3),
               "concentrated": (1 - (1 - probability) ** 3).sum().item()}
    gradient_old, gradient_new = torch.tensor([1., 1.]), torch.tensor([1., -.5])
    preconditioner = torch.diag(torch.tensor([.1, 10.]))

    parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    optimizer = AdamW([parameter], lr=.1, weight_decay=0.)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    initial = parameter.detach().item()
    parameter.grad = None
    optimizer.step()
    absent = parameter.detach().item()
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()

    base = rope_inverse_frequencies(64, 50000.)
    scaled, _ = yarn_inverse_frequencies(64, 50000., 32, 4096, beta_fast=1, beta_slow=1)
    return dict(
        scope="CPU algebraic identities and counterexamples; no new large-model or GPU measurements",
        softmax_merge=dict(output_error=(merged-dense).abs().max().item(),
                           gradient_error=max((a-b).abs().max().item() for a,b in zip(actual,expected))),
        delta_rule=dict(key_squared_norm=key.square().sum().item(), squared_error_ratios=contractions),
        clip_microbatches=dict(full_loss=full.item(), separate_loss=micro.item(),
                              full_gradient_norm=torch.autograd.grad(full,images)[0].norm().item()),
        expert_union=experts,
        optimizer_zero_vs_none=dict(no_gradient_change=absent-initial,
                                    zero_gradient_change=parameter.detach().item()-absent),
        preconditioned_transfer=dict(raw_inner_product=(gradient_old@gradient_new).item(),
                                    weighted_inner_product=(gradient_old@preconditioner@gradient_new).item()),
        retained_token_mix=[1/11, 10/11],
        incomplete_k3=dict(expected_k3=math.log(2)-.5, true_kl=math.log(2)),
        normalized_recovery=dict(mean_of_ratios=.55, ratio_of_sums=2/11),
        padded_attention_pair_ratio=2*8192**2/(1024**2+8192**2),
        kimi_yarn=dict(unscaled_pairs=int((scaled==base).sum()), interpolated_pairs=int((scaled==base/32).sum())),
    )


def packing_equivalence_demo():
    """A one-head network: packed independent documents vs separate forwards.

    No positional encoding or padding query is needed in this small construction.
    Both paths share embedding/projection weights and score the same five targets.
    """
    from .objectives import packed_causal_mask, packed_next_token_mask, sft_loss
    rng = torch.Generator().manual_seed(40336)
    embedding = torch.randn(9, 6, dtype=torch.float64, generator=rng).requires_grad_()
    query, key, value = [torch.randn(6, 6, dtype=torch.float64, generator=rng).requires_grad_() for _ in range(3)]
    head = torch.randn(6, 9, dtype=torch.float64, generator=rng).requires_grad_()
    parameters = (embedding, query, key, value, head)
    # Reuse BOS/EOS IDs; distinct document IDs identify the two independent segments.
    ids = torch.tensor([[1, 3, 4, 2, 1, 5, 2]])
    documents = torch.tensor([[0, 0, 0, 0, 1, 1, 1]])
    allowed = packed_causal_mask(documents[:, :-1])[0]
    mask = packed_next_token_mask(documents)

    def logits(tokens, visibility=None):
        x = embedding[tokens]
        scores = (x @ query) @ (x @ key).T / math.sqrt(x.shape[-1])
        if visibility is None:
            visibility = torch.ones(len(tokens), len(tokens), dtype=torch.bool).tril()
        return (scores.masked_fill(~visibility, -torch.inf).softmax(-1) @ (x @ value)) @ head

    packed_logits = logits(ids[0, :-1], allowed)
    packed = sft_loss(packed_logits[None], ids[:, 1:], mask)
    separate = sum(F.cross_entropy(logits(part[:-1]), part[1:], reduction='sum')
                   for part in (ids[0, :4], ids[0, 4:])) / mask.sum()
    packed_gradient = torch.autograd.grad(packed, parameters)
    separate_gradient = torch.autograd.grad(separate, parameters)
    perturbed = ids[0, :-1].clone()
    perturbed[1:3] = torch.tensor([7, 8])
    isolated = logits(perturbed, allowed)[4:]
    return dict(targets=int(mask.sum()), cross_document_targets_suppressed=int((~mask).sum()),
                loss_error=abs(packed.item()-separate.item()),
                gradient_error=max((a-b).abs().max().item() for a,b in zip(packed_gradient,separate_gradient)),
                other_document_logit_error=(isolated-packed_logits[4:]).abs().max().item())


def chapter_interfaces_demo():
    """Small executable checks for data/model/evaluation interfaces, without training."""
    from .evaluation import paired_bootstrap, paired_cluster_bootstrap
    from .extensions import linear_attention_recurrent, linear_attention_explicit
    from .long_context import ntk_aware, position_interpolation, rope_inverse_frequencies, yarn_inverse_frequencies
    from .posttrain import generalized_advantages
    from .tokenizer import Tokenizer

    tokenizer = Tokenizer({i:bytes([i]) for i in range(256)}, [])
    ids = list('中🙂'.encode('utf-8'))
    attention = {}
    for dtype in (torch.float16, torch.float32, torch.float64):
        q = torch.ones(2, 16, dtype=dtype)
        k = torch.full_like(q, -20.)
        v = torch.ones(2, 3, dtype=dtype)
        recurrent, dense = linear_attention_recurrent(q,k,v), linear_attention_explicit(q,k,v)
        attention[str(dtype)] = dict(finite=bool(recurrent.isfinite().all() and dense.isfinite().all()),
                                    reference_error=(recurrent.double()-1).abs().max().item(),
                                    dense_error=(recurrent-dense).abs().max().item())
    a, b, clusters = [0.]*10, [1.]+[0.]*9, ['small']+['large']*9
    rewards = torch.tensor([0., 0., 1.], dtype=torch.float64)
    values = torch.tensor([.2, .4, .7], dtype=torch.float64)
    following = torch.tensor([.4, .7, 0.], dtype=torch.float64)
    last = torch.tensor([False, False, True])
    no_boundary = torch.zeros_like(last)
    terminal_adv, _ = generalized_advantages(rewards,values,following,last,no_boundary,lam=1.)
    following[-1] = .6
    truncated_adv, _ = generalized_advantages(rewards,values,following,no_boundary,last,lam=1.)
    posterior = torch.tensor([.9, .1], dtype=torch.float64)
    scores = torch.tensor([[2., -1.], [-1., 2.]], dtype=torch.float64)
    base = rope_inverse_frequencies(128, 10000.)
    yarn, _ = yarn_inverse_frequencies(128, 10000., 32., 4096)
    return dict(
        scope='Small CPU interface examples, exact counterexamples and cluster resampling; no model training',
        packing=packing_equivalence_demo(),
        utf8=dict(whole=tokenizer.decode(ids), streamed=''.join(tokenizer.decode_iterable(iter(ids))),
                  incorrect_tokenwise=''.join(tokenizer.decode([i]) for i in ids)),
        linear_attention=attention,
        initialization=dict(uniform_nll=math.log(2), symmetric_expected_nll=(-scores.log_softmax(-1)[:,0]).mean().item(),
                            biased_independent_nll=-math.log(.9)),
        calibration=dict(constant_score=.5, true_positive_fraction=posterior.mean().item(),
                         true_density_ratios=(posterior/(1-posterior)).tolist(), score_based_ratio=1.),
        bootstrap=dict(item=paired_cluster_bootstrap(a,b,clusters),
                       cluster=paired_cluster_bootstrap(a,b,clusters,weighting='cluster'),
                       wrongly_independent=paired_bootstrap(a,b)),
        gae=dict(lambda_one_terminal=terminal_adv.tolist(), lambda_one_timeout=truncated_adv.tolist()),
        rope_scaling=dict(setting=dict(dim=128, base=10000., factor=32., original_context=4096),
                          pi=(base/position_interpolation(128,10000.,32.)).tolist(),
                          ntk=(base/ntk_aware(128,10000.,32.)).tolist(), yarn=(base/yarn).tolist()),
    )


def mismatch_stats(train_logps, rollout_logps, mask):
    if train_logps.shape != rollout_logps.shape or mask.shape != train_logps.shape or not torch.any(mask):
        raise ValueError("matching tensors and a nonempty mask required")
    if not train_logps.is_floating_point() or not rollout_logps.is_floating_point() \
            or train_logps.device != rollout_logps.device or mask.device != train_logps.device \
            or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('floating log probabilities and a binary mask on one device required')
    current = train_logps[mask.bool()].detach().double()
    old = rollout_logps[mask.bool()].detach().double()
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise ValueError("valid log probabilities must be finite")
    delta = current - old
    if not torch.isfinite(delta).all():
        raise ValueError('log-probability difference exceeds the FP64 range')
    absolute = delta.abs()
    return dict(mean=delta.mean().item(), absolute_p50=absolute.quantile(.5).item(),
                absolute_p95=absolute.quantile(.95).item(), absolute_p99=absolute.quantile(.99).item(),
                absolute_max=absolute.max().item())


def bootstrap_power_prediction(compute, losses, target, repeats=500, seed=19):
    """Two-parameter power model L=a*C^b; independent-row bootstrap + residual draw.

    Illustrates intervals, not an LM scaling law or a calibrated far-extrapolation bound.
    """
    c, y = np.asarray(compute, dtype=float), np.asarray(losses, dtype=float)
    if c.ndim!=1 or c.shape!=y.shape or len(c)<4 or np.any(c<=0) or np.any(y<=0) \
            or not math.isfinite(target) or target<=0 or type(repeats) is not int or repeats<10:
        raise ValueError("positive matching samples, target and at least 10 repeats required")
    if not np.isfinite(c).all() or not np.isfinite(y).all():
        raise ValueError("finite samples required")
    x, logy = np.log(c), np.log(y)
    design = np.column_stack((np.ones(len(c)), x))
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError("distinct compute budgets are needed to identify a power-law exponent")
    coef = np.linalg.lstsq(design, logy, rcond=None)[0]
    residual = logy-design@coef
    residual = residual-residual.mean()
    rng = np.random.default_rng(seed)
    fitted, future = [], []
    for _ in range(repeats):
        indices = rng.integers(len(c), size=len(c))
        matrix = design[indices]
        if np.linalg.matrix_rank(matrix)<2:
            continue
        fit = np.linalg.lstsq(matrix, logy[indices], rcond=None)[0]
        value = np.array([1., math.log(target)])@fit
        fit_residual = logy[indices]-matrix@fit
        # exp(log-space trend) is a median, not the arithmetic conditional mean.
        # Empirical residual smearing estimates the mean on the original scale.
        fitted.append(math.exp(value)*np.exp(fit_residual).mean())
        future.append(math.exp(value+rng.choice(residual)))
    if len(fitted) < 2:
        raise ValueError("too few identifiable bootstrap resamples; use more distinct budgets or repeats")
    return dict(fitted_mean_interval=np.quantile(fitted,[.025,.975]).tolist(),
                new_run_prediction_interval=np.quantile(future,[.025,.975]).tolist(),
                prediction=float(np.exp(np.array([1.,math.log(target)])@coef)*np.exp(residual).mean()),
                scope="independent synthetic-row bootstrap; model misspecification not covered")


def fake_quantize_ste(x, scale=0.5, qmax=7):
    """Fixed scale, full identity STE (including clipped values) for teaching."""
    if not math.isfinite(scale) or scale<=0 or qmax<1:
        raise ValueError("positive finite scale and qmax required")
    quantized = (x/scale).round().clamp(-qmax,qmax)*scale
    return x+(quantized-x).detach()


def two_of_four_mask(weight):
    if weight.ndim!=2 or weight.shape[-1]%4 or not torch.isfinite(weight).all():
        raise ValueError("finite matrix with input dimension divisible by four required")
    groups = weight.reshape(weight.shape[0],-1,4)
    selected = groups.abs().topk(2,dim=-1).indices
    return torch.zeros_like(groups,dtype=torch.bool).scatter(-1,selected,True).reshape(weight.shape)


def sae_encode_decode(h, encoder, decoder):
    """Bias-free illustrative SAE, h [B,d], encoder [d,k], decoder [k,d]."""
    z = (h@encoder).relu()
    return z, z@decoder


def steer_feature_with_residual(h, z, decoder, feature, shift):
    modified = z.clone()
    modified[:,feature] += shift
    return h+(modified-z)@decoder


def muon_shape_rms(matrix):
    """SVD partial polar factor; zero singular directions stay zero, as in NS."""
    if matrix.ndim!=2 or min(matrix.shape)<1 or not torch.isfinite(matrix).all():
        raise ValueError("finite nonempty matrix required")
    u, s, vh = torch.linalg.svd(matrix,full_matrices=False)
    active = s > s.max()*max(matrix.shape)*torch.finfo(matrix.dtype).eps
    polar = (u*active)@vh
    scaled = .2*math.sqrt(max(matrix.shape))*polar
    return dict(unscaled_rms=polar.square().mean().sqrt().item(),
                scaled_rms=scaled.square().mean().sqrt().item())


def demo():
    rng = np.random.default_rng(11)
    c = np.geomspace(1,100,50)
    loss = 3*c**(-.2)*np.exp(rng.normal(0,.08,len(c)))
    return dict(dpo=dpo_shared_parameter_demo(),
                prediction=bootstrap_power_prediction(c,loss,150),
                fp4_bits=dict(mxfp4=4+8/32,nvfp4_local=4+8/16),
                muon=muon_shape_rms(torch.tensor([[1.,2.,3.],[2.,-1.,4.]],dtype=torch.float64)))


if __name__ == "__main__":
    import json
    print(json.dumps(demo(),indent=2))

```
## mechanisms.py
[对应文件](../../examples/mechanisms.py)
```python
"""Executable derivations: exact loss tiling, affine scan, MLA and statistical diagnostics.

Small CPU references. They expose invariants, not production GPU performance.
"""
import math
import torch
from torch.autograd.function import once_differentiable


class _LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, block):
        # hidden [M,D], weight [V,D]; one tile [M,block] at a time.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            lse = hidden.new_full((len(hidden),), -torch.inf)
            for start in range(0, len(weight), block):
                logits = hidden @ weight[start:start + block].T
                lse = torch.logaddexp(lse, logits.logsumexp(-1))
            target = (hidden * weight[labels]).sum(-1)
            ctx.save_for_backward(hidden, weight, labels, lse)
            ctx.block = block
            return (lse - target).mean()

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        hidden, weight, labels, lse = ctx.saved_tensors
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            dx, dw = torch.zeros_like(hidden), torch.empty_like(weight)
            rows = torch.arange(len(hidden), device=hidden.device)
            for start in range(0, len(weight), ctx.block):
                end = min(start + ctx.block, len(weight))
                gradient = (hidden @ weight[start:end].T - lse[:, None]).exp()
                owned = (labels >= start) & (labels < end)
                gradient[rows[owned], labels[owned] - start] -= 1
                gradient *= upstream / len(hidden)
                dx.add_(gradient @ weight[start:end])
                dw[start:end] = gradient.T @ hidden
            return dx, dw, None, None


def tiled_linear_cross_entropy(hidden, weight, labels, block=16):
    """Exact mean CE, without storing all logits. FP32/64, no bias, first-order only.

    Includes full hidden/weight gradients; no gradient filtering, label smoothing,
    ignore_index or GPU fusion. Filter padded targets before calling. Forward and
    recomputation disable autocast so the stated FP32/64 arithmetic stays consistent.
    """
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden [M,D] and weight [V,D] must have matching D")
    if min(*hidden.shape, len(weight)) < 1 or labels.shape != hidden.shape[:1] or labels.dtype != torch.long:
        raise ValueError("nonempty inputs and one int64 label per row required")
    if hidden.dtype not in (torch.float32, torch.float64) or weight.dtype != hidden.dtype:
        raise ValueError("matching FP32 or FP64 inputs required")
    if hidden.device != weight.device or labels.device != hidden.device:
        raise ValueError("inputs must share a device")
    if not isinstance(block, int) or block < 1 or torch.any((labels < 0) | (labels >= len(weight))):
        raise ValueError("positive tile width and in-vocabulary labels required")
    if not torch.isfinite(hidden).all() or not torch.isfinite(weight).all():
        raise ValueError("finite inputs required")
    return _LinearCrossEntropy.apply(hidden, weight, labels, block)


def sharded_cross_entropy(shards, labels):
    """Virtual vocabulary ranks: global maximum/sum, then owner-only target logit.

    All shards live in one process; autograd here is NOT a distributed collective.
    """
    if not shards or any(s.ndim != 2 or min(s.shape) < 1 for s in shards):
        raise ValueError("nonempty [M,V_rank] shards required")
    if any(s.shape[0] != shards[0].shape[0] for s in shards) or labels.shape != shards[0].shape[:1]:
        raise ValueError("matching row counts and labels required")
    if labels.dtype != torch.long or any(s.device != labels.device or not s.is_floating_point()
                                         or not torch.isfinite(s).all() for s in shards):
        raise ValueError("finite floating shards and int64 labels on one device required")
    dtype = torch.float64 if any(s.dtype == torch.float64 for s in shards) else torch.float32
    shards = [s.to(dtype) for s in shards]
    if torch.any((labels < 0) | (labels >= sum(s.shape[1] for s in shards))):
        raise ValueError("labels outside vocabulary")
    maximum = torch.stack([s.max(-1).values for s in shards]).max(0).values.detach()
    denominator = sum((s - maximum[:, None]).exp().sum(-1) for s in shards)
    target, offset = torch.zeros_like(maximum), 0
    for shard in shards:
        local = labels - offset
        owned = (local >= 0) & (local < shard.shape[1])
        selected = shard.gather(1, local.clamp(0, shard.shape[1] - 1)[:, None]).squeeze(1)
        target = target + selected.masked_fill(~owned, 0)
        offset += shard.shape[1]
    return (maximum + denominator.log() - target).mean()


def affine_scan(a, b, initial):
    """Inclusive parallel scan of h[t]=a[t]*h[t-1]+b[t], diagonal a.

    Hillis-Steele doubling: O(log T) dependency depth, O(T log T) work/storage
    with autograd. Not Mamba's work-efficient GPU kernel.
    """
    if a.ndim != 2 or a.shape != b.shape or initial.shape != a.shape[1:] or len(a) == 0:
        raise ValueError("a,b [T,D] and initial [D] required")
    aa, bb, step = a, b, 1
    while step < len(a):
        new_a = aa[step:] * aa[:-step]
        new_b = aa[step:] * bb[:-step] + bb[step:]
        aa = torch.cat((aa[:step], new_a))
        bb = torch.cat((bb[:step], new_b))
        step *= 2
    return aa * initial + bb


def mla_content_output(query, latent, up_key, up_value, output, absorbed=True):
    """Content-only MLA at one query: head-wise attention, then output projection.

    q [H,Dk], c [T,R], Wk [H,Dk,R], Wv [H,Dv,R], Wo [H,Dout,Dv].
    Positional scores and masks are intentionally omitted to isolate the identity.
    """
    if absorbed:
        q_latent = torch.einsum('hd,hdr->hr', query, up_key)
        scores = q_latent @ latent.T / math.sqrt(query.shape[-1])
        context = scores.softmax(-1) @ latent
        combined = torch.einsum('hov,hvr->hor', output, up_value)
        return torch.einsum('hor,hr->o', combined, context)
    keys = torch.einsum('tr,hdr->htd', latent, up_key)
    values = torch.einsum('tr,hvr->htv', latent, up_value)
    scores = torch.einsum('hd,htd->ht', query, keys) / math.sqrt(query.shape[-1])
    context = torch.einsum('ht,htv->hv', scores.softmax(-1), values)
    return torch.einsum('hov,hv->o', output, context)


def importance_diagnostics(log_weights):
    """Stable weight concentration diagnostics, not an unbiasedness guarantee."""
    if log_weights.ndim != 1 or not len(log_weights) or not torch.isfinite(log_weights).all():
        raise ValueError("a finite nonempty vector of log weights required")
    values = log_weights.detach().double()
    normalized = values.softmax(0)
    return dict(ess=(1 / normalized.square().sum()).item(),
                max_mass=normalized.max().item(),
                log_mean_weight=(values.logsumexp(0) - math.log(len(values))).item())


def verifier_precision(base_success, true_accept, false_accept):
    """P(correct | accepted), given generator success and verifier TPR/FPR."""
    if not all(0 <= x <= 1 for x in (base_success, true_accept, false_accept)):
        raise ValueError("probabilities must be in [0,1]")
    mass = base_success * true_accept + (1 - base_success) * false_accept
    if mass == 0:
        raise ValueError("conditioning event has zero probability")
    return base_success * true_accept / mass


def potential_shaping(rewards, potentials, gamma=1.):
    """Add gamma*Phi(s[t+1])-Phi(s[t]); terminal potential must be handled by caller."""
    if rewards.ndim != 1 or potentials.shape != (len(rewards) + 1,) or not 0 <= gamma <= 1:
        raise ValueError("T rewards, T+1 potentials and gamma in [0,1] required")
    return rewards + gamma * potentials[1:] - potentials[:-1]


def stochastic_round_interval(x, lower, upper, generator=None):
    """In-range stochastic rounding to TWO adjacent values; no clipping or STE.

    The probability and the uniform draw are computed in float64: a bf16/fp16 draw is itself
    quantized (bf16 steps of 2^-8 near 1), which would bias the rounding it is meant to unbias.
    The result keeps x's dtype; both endpoints must be exactly representable in that dtype.
    """
    if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
        raise ValueError("finite ordered endpoints required")
    if not x.is_floating_point():
        raise ValueError("floating-point input required")
    low, high = x.new_tensor(lower), x.new_tensor(upper)
    if low.item() != lower or high.item() != upper:
        raise ValueError("endpoints must be exactly representable in the input dtype")
    values = x.double()
    if not torch.isfinite(values).all() or torch.any((values < lower) | (values > upper)):
        raise ValueError("x must lie in the specified interval")
    span = upper - lower
    # Opposite finite FP64 endpoints can have an overflowing difference.
    probability = ((values - lower) / span if math.isfinite(span)
                   else (values / 2 - lower / 2) / (upper / 2 - lower / 2))
    draws = torch.rand(x.shape, dtype=torch.float64, device=x.device, generator=generator)
    # Scalar-only torch.where branches would first round both endpoints to the default dtype.
    return torch.where(draws < probability, high, low)


def demo():
    from torch.nn import functional as F
    rng = torch.Generator().manual_seed(336)
    h = torch.randn(7, 5, generator=rng, dtype=torch.float64, requires_grad=True)
    w = torch.randn(19, 5, generator=rng, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 3, 6, 9, 12, 15, 18])
    reference = F.cross_entropy(h @ w.T, labels)
    tiled = tiled_linear_cross_entropy(h, w, labels, 6)
    actual = torch.autograd.grad(tiled, (h, w))
    expected = torch.autograd.grad(reference, (h, w))
    logits = h.detach() @ w.detach().T
    split = sharded_cross_entropy(list(logits.split([4, 8, 7], dim=1)), labels)
    a, b = torch.rand(9, 4, generator=rng, dtype=torch.float64), torch.randn(9, 4, generator=rng, dtype=torch.float64)
    initial = torch.randn(4, generator=rng, dtype=torch.float64)
    state, sequential = initial, []
    for at, bt in zip(a, b):
        state = at * state + bt
        sequential.append(state)
    tensors = [torch.randn(*shape, generator=rng, dtype=torch.float64)
               for shape in ((2, 3), (7, 4), (2, 3, 4), (2, 2, 4), (2, 5, 2))]
    mla_error = (mla_content_output(*tensors) - mla_content_output(*tensors, absorbed=False)).abs().max()
    samples = stochastic_round_interval(torch.full((50000,), .3), 0., .5, rng)
    p = [.01, .05, .25, .5, .9]
    return dict(
        scope='Small CPU derivations and controlled sampling; no industrial training or GPU benchmark',
        linear_ce=dict(loss=tiled.item(), loss_error=abs(tiled.item() - reference.item()),
                       gradient_max_error=max((x-y).abs().max().item() for x,y in zip(actual, expected)),
                       virtual_vocab_parallel_error=abs(split.item() - reference.item())),
        affine_scan_max_error=(affine_scan(a,b,initial)-torch.stack(sequential)).abs().max().item(),
        mla_content_max_error=mla_error.item(),
        importance=dict(uniform=importance_diagnostics(torch.zeros(16)),
                        concentrated=importance_diagnostics(torch.tensor([20.]+[0.]*15))),
        verifier=dict(base_success=p, accepted_precision=[verifier_precision(x,.9,.05) for x in p]),
        rounding=dict(expected=.3, empirical_mean=samples.mean().item(), expected_variance=.3*(.5-.3),
                      empirical_variance=samples.var(correction=0).item()),
        moe_expected_experts={str(batch):128*(1-(1-4/128)**batch) for batch in (1,8,32,128)},
    )


if __name__ == '__main__':
    import json
    print(json.dumps(demo(), indent=2))

```
