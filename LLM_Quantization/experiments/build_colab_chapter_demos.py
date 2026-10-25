#!/usr/bin/env python3
"""生成自带源码的Colab逐章GPU Notebook；无需知道本地项目的远程仓库地址。"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def notebook_text():
    import nbformat
    from experiments.chapter_demo_catalog import DEMOS
    # 只打包运行、核验需要的源码和章节；排除模型、历史输出和密钥等无关文件。
    paths = set((ROOT/'experiments').glob('*.py'))
    paths.update((ROOT/'experiments/demos').glob('*.py'))
    paths.update(ROOT/d['document'] for d in DEMOS)
    paths.update(ROOT/p for p in ['requirements.txt', 'requirements-demos.txt', 'constraints-cpu.txt',
                                  'tests/test_demo_torch_kernels.py'])
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(paths):
            info = zipfile.ZipInfo(str(p.relative_to(ROOT)), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, p.read_bytes())
    package = output.getvalue()
    encoded = base64.b64encode(package).decode()
    digest = hashlib.sha256(package).hexdigest()
    cells = []

    def markdown(text):
        cells.append(nbformat.v4.new_markdown_cell(text))

    def code(text):
        cells.append(nbformat.v4.new_code_cell(text))

    markdown('# 逐章量化 demo：Colab GPU\n\n运行时选择 GPU，再从上到下执行。'
             f'此Notebook自带20章及新增主题的{len(DEMOS)}个demo源码，无需另上传工程。采用PyTorch CUDA数值模拟，'
             '默认FP64用于数学核验；不测试原生INT4/FP4 Tensor Core加速。'
             '当前发布仅经CPU验证，GPU测试由本Notebook实际执行并记录。')
    markdown('## 1. 解包本次源码快照\n\n这是生成单元，包含压缩源码。'
             '先核对包与文件SHA-256，再写入新建的临时项目目录；不会覆盖已有工程。')
    code(f'''import base64, hashlib, io, json, os, sys, tempfile, zipfile
from pathlib import Path

# 包内仅有公开教学源码与文档；校验失败立即停止。
PACKAGE_SHA256 = {digest!r}
FILE_SHA256 = {manifest!r}
PACKAGE_B64 = {encoded!r}
package = base64.b64decode(PACKAGE_B64)
assert hashlib.sha256(package).hexdigest() == PACKAGE_SHA256
parent = '/content' if Path('/content').exists() else None
project = Path(tempfile.mkdtemp(prefix='quant_chapter_demos_', dir=parent))
with zipfile.ZipFile(io.BytesIO(package)) as archive:
    assert set(archive.namelist()) == set(FILE_SHA256)
    for name, expected in FILE_SHA256.items():
        destination = (project/name).resolve()
        if not destination.is_relative_to(project.resolve()):
            raise ValueError('非法归档路径')
        content = archive.read(name)
        assert hashlib.sha256(content).hexdigest() == expected, name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
os.chdir(project)
sys.path.insert(0, str(project))
print('源码目录：', project, '；核验文件数：', len(FILE_SHA256))''')
    markdown('## 2. 检查环境与GPU\n\n保留Colab预装的CUDA版PyTorch。'
             '若没有GPU，请先更改运行时；不要安装CPU专用torch构建。')
    code('''import importlib.util, subprocess
# 仅在缺包时安装，避免覆盖Colab已经可用的CUDA版本。
missing = [name for name in ('numpy', 'torch') if importlib.util.find_spec(name) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
import torch
version = tuple(int(part) for part in torch.__version__.split('+')[0].split('.')[:2])
if version < (2, 2):
    raise RuntimeError('需要PyTorch 2.2+；请使用更新的Colab运行时。')
print('PyTorch:', torch.__version__, 'CUDA构建:', torch.version.cuda)
if not torch.cuda.is_available():
    raise RuntimeError('没有可用CUDA，请在Colab选择GPU运行时后重新运行。')
print('GPU:', torch.cuda.get_device_name(0))
probe = torch.ones((2, 2), device='cuda')
print('CUDA矩阵乘探针:', probe@probe)
torch.cuda.synchronize()''')
    markdown('## 3. 选择章节并运行\n\n`all`表示全部；可选00–16、math、int8、research，'
             '或gpu_formats、mr_gptq、four_over_six、training_transpose、gradient_rounding、oscillation、scale_layout。'
             '随机输入由CPU准备并搬到设备，数值计算用CUDA；打印和文件I/O仍在主机。\n\n'
             '先运行默认示例。练习时在左侧文件浏览器打开解包目录下的 `experiments/demos/`，修改源码并保存，'
             '将下方 `PRACTICE=True` 后重新执行本单元，无需重建文档或重新解包。'
             '数值断言与设备检查仍会执行；如改变了题设，应相应推导新的预期结果。'
             '每次执行创建新结果目录，失败时也可直接跳到下载单元取回已有结果和错误日志。')
    code('''# 显式cuda失败不会改用cpu；练习模式仅放宽正文一致性检查。
CHAPTER = 'all'
DEVICE = 'cuda'
PRACTICE = False
RESULT_DIR = Path(tempfile.mkdtemp(prefix='run_', dir=project))
command = [sys.executable, '-B', 'experiments/chapter_demo_lab.py',
           '--chapter', CHAPTER, '--device', DEVICE,
           '--write-results', '--output-dir', str(RESULT_DIR)]
if PRACTICE:
    command.append('--practice')
# 日志先写磁盘，避免异常令已完成的输出消失；每轮有独立目录。
with (RESULT_DIR/'chapter_execution.log').open('w') as log:
    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
print((RESULT_DIR/'chapter_execution.log').read_text())
print('本轮结果目录：', RESULT_DIR)
completed.check_returncode()
report = json.loads((RESULT_DIR/'chapter_demo_results.json').read_text())
assert report['status'] == 'completed'
for row in report['results']:
    print('\n章节', row['id'], row['title'], row['runtime']['device'])
    print(row['output'])'''.replace("print('\n章节'", "print('\\n章节'"))
    markdown('## 4. GPU独立数值对照\n\n在CUDA上比较PyTorch量化/缓存与原NumPy参考，'
             f'并运行全部{len(DEMOS)}个demo的设备核验。练习模式只跳过正文同步要求。'
             '失败时保留错误和测试日志，不把CPU通过视作GPU通过。')
    code("environment = dict(os.environ, QUANT_DEMO_PRACTICE='1' if PRACTICE else '0')\n"
         "with (RESULT_DIR/'gpu_tests.log').open('w') as log:\n"
         "    checked = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',\n"
         "                              '-p', 'test_demo_torch_kernels.py', '-v'],\n"
         "                             env=environment, stdout=log, stderr=subprocess.STDOUT, text=True)\n"
         "print((RESULT_DIR/'gpu_tests.log').read_text())\nchecked.check_returncode()")
    markdown('## 5. 可选：原生NVFP4前后向探针\n\n默认关闭。只有已配置TE 2.8的SM100/SM103环境才开启；'
             '脚本不会安装软件或回退到QDQ。T4/A100/H100无需运行此单元；SM120需匹配专用后端。'
             '此探针执行真实NVFP4 recipe，但不代替profiler指令确认或模型质量评测。')
    code("RUN_NATIVE_NVFP4 = False\nif RUN_NATIVE_NVFP4:\n"
         "    with (RESULT_DIR/'native_execution.log').open('w') as log:\n"
         "        native = subprocess.run([sys.executable, 'experiments/native_nvfp4_probe.py',\n"
         "                                 '--output-dir', str(RESULT_DIR)],\n"
         "                                stdout=log, stderr=subprocess.STDOUT, text=True)\n"
         "    print((RESULT_DIR/'native_execution.log').read_text())\n    native.check_returncode()\n"
         "else:\n    print('原生NVFP4探针未运行；上面的CUDA demo属于数值模拟。')")
    markdown('## 6. 下载结果\n\nZIP包含本轮逐章JSON、实际执行源码、测试日志和已运行的原生探针结果。'
             'JSON记录实际设备、运行状态及源码哈希；原生探针还记录实际张量dtype、种子和recipe。'
             '失败后也可运行本单元；`failed/partial/running`不能当作完整通过。')
    code("import shutil\nfrom google.colab import files\n"
         "archive_path = shutil.make_archive(str(RESULT_DIR), 'zip', root_dir=RESULT_DIR)\n"
         "files.download(archive_path)")
    for i, cell in enumerate(cells):
        cell.id = f'chapter-demo-{i:02d}'
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'chapter_demo_validation': {'gpu': 'not_run_in_cpu_development_environment',
                                    'package_sha256': digest, 'file_count': len(manifest)}})
    nbformat.validate(notebook)
    return nbformat.writes(notebook)


def validate_notebook():
    """检查代码语法、无伪造执行输出，以及内嵌文件对应当前源码。"""
    import ast
    import nbformat
    path = ROOT/'experiments/notebooks/colab_chapter_demos.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            if cell.outputs or cell.execution_count is not None:
                raise ValueError('发布的Colab demo Notebook不应冒充已完成GPU执行')
    if path.read_text() != notebook_text():
        raise ValueError('Colab源码快照过期，请运行 build_colab_chapter_demos.py')
    print('PASS Colab demo notebook syntax and embedded source snapshot')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = notebook_text()
    path = ROOT/'experiments/notebooks/colab_chapter_demos.ipynb'
    if args.check:
        if not path.exists() or path.read_text() != text:
            raise SystemExit('Colab源码快照过期，请重新生成Notebook')
    else:
        path.write_text(text)
    print('PASS self-contained Colab chapter notebook')


if __name__ == '__main__':
    main()
