#!/usr/bin/env python3
"""显式执行Transformer Engine 2.8原生NVFP4 Linear；不回退到数值模拟。

本探针限定B200/B300一类SM100/103；不自动安装依赖，不下载模型。
仅验证一次前后向的可执行性和有限输出；不是LLM质量或速度benchmark。
"""
import argparse
import importlib.metadata
from pathlib import Path
import sys
import traceback

SEED = 20261020

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require_native_device(device):
    import torch
    if torch.version.hip or device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('原生NVFP4探针需要NVIDIA CUDA；不会降级成CPU/QDQ模拟。')
    capability = torch.cuda.get_device_capability(device)
    # 白名单是本探针验证目标，不能用 capability >= (10,0) 推断任意未来架构兼容。
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError(f'本TE 2.8探针限定SM100/103，实际为{capability}。'
                           'T4/A100/H100运行数值demo；SM120需单独匹配其后端。')
    return capability


def tensor_metadata(**values):
    """从实际张量采集dtype/shape/device；不借用教学runtime的FP64默认值。"""
    return {name: dict(dtype=str(value.dtype), shape=list(value.shape), device=str(value.device))
            for name, value in values.items()}


def run():
    if not __debug__:
        raise RuntimeError('请不要使用 -O，原生探针需要保留数值断言。')
    import torch
    from experiments.demo_runtime import DemoRuntime
    runtime = DemoRuntime('cuda:0')
    capability = require_native_device(runtime.device)
    # 固定API版本；模块导入可能加载CUDA扩展，错误直接暴露，不能报告成成功模拟。
    try:
        version = importlib.metadata.version('transformer_engine')
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError('请先按TE 2.8官方安装说明准备CUDA/PyTorch扩展；本脚本不自动安装。') from exc
    if version.split('.')[:2] != ['2', '8']:
        raise RuntimeError(f'本探针对照TE 2.8，当前{version}；先核对版本与API。')
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling
    from transformer_engine.pytorch.fp8 import check_nvfp4_support
    supported, reason = check_nvfp4_support()
    if not supported:
        raise RuntimeError(reason)

    # 第1步：相同权重与输入构造BF16参考，尺寸取16的倍数以匹配量化/内核。
    torch.manual_seed(SEED)
    layer = te.Linear(256, 256, bias=False, params_dtype=torch.bfloat16,
                      device=runtime.device)
    x = torch.randn(256, 256, device=runtime.device, dtype=torch.bfloat16, requires_grad=True)
    with torch.no_grad():
        reference = torch.nn.functional.linear(x, layer.weight)

    # 第2步：显式启用NVFP4 recipe；Tensor Core输出与梯度仍可为BF16/FP32。
    recipe = NVFP4BlockScaling()
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        output = layer(x)
        loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(runtime.device)

    # 第3步：验证前后向执行成功与数值有限，不设置任意MSE阈值冒充质量保证。
    assert output.shape == reference.shape
    for value in (output, x.grad, layer.weight.grad):
        assert value is not None and value.device == runtime.device and torch.isfinite(value).all()
    relative_mse = ((output.float()-reference.float()).square().sum()
                    /reference.float().square().sum()).item()
    # dtype来自运行结果；recipe标志可能受环境变量影响，也要记录实际值。
    environment = dict(device=str(runtime.device), torch_version=torch.__version__,
                       cuda_build=torch.version.cuda, gpu_name=torch.cuda.get_device_name(runtime.device),
                       seed=SEED, input_preparation='torch.randn on CUDA after torch.manual_seed',
                       tensors=tensor_metadata(input=x, weight=layer.weight, output=output,
                                               reference=reference, loss=loss,
                                               input_gradient=x.grad, weight_gradient=layer.weight.grad))
    recipe_settings = {name: getattr(recipe, name) for name in
                       ('disable_rht', 'disable_stochastic_rounding', 'disable_2d_quantization')}
    recipe_settings.update(name=type(recipe).__name__, fp4_format=recipe.fp4_format.name,
                           fp8_format=recipe.fp8_format.name, description=repr(recipe))
    return dict(status='executed', mode='native_te_nvfp4_requested',
                compute_capability=list(capability), transformer_engine=version,
                runtime=environment, recipe=recipe_settings, output_nmse_vs_bf16=relative_mse,
                scope='TE NVFP4 forward/backward API smoke test; no profiler instruction proof, no benchmark or model quality claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='显式保存探针结果的目录')
    args = parser.parse_args()
    from experiments.result_utils import run_metadata, write_json
    record = {**run_metadata({'mode': 'native_te_nvfp4', 'seed': SEED}), 'status': 'running'}
    def save():
        if args.output_dir:
            write_json(args.output_dir/'native_nvfp4_probe.json', record)
    save()
    try:
        record.update(run())
    except BaseException as exc:
        record.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc),
                                                  traceback=traceback.format_exc()))
        save()
        raise
    save()
    import json
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
