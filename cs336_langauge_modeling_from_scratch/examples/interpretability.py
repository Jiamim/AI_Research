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
