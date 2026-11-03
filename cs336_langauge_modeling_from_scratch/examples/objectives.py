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
