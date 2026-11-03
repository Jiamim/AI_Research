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
