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
