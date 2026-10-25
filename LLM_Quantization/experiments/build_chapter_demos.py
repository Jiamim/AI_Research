"""把逐步注释的实操源码嵌入对应章节；由现有文档生成器统一调用。"""
import re

from experiments.chapter_demo_catalog import DEMOS, ROOT


def regions(source):
    """同章可以有多个非嵌套实操；返回每个唯一ID的完整区间。"""
    matches = re.finditer(r'<!-- (?:chapter-demo: ([a-z0-9_]+)|(/chapter-demo)) -->', source)
    opened, found = None, {}
    for match in matches:
        if match[1]:
            if opened or match[1] in found:
                raise ValueError('章节实操标记嵌套或重复')
            opened = (match[1], match.start())
        else:
            if opened is None:
                raise ValueError('章节实操结束标记无对应开始')
            found[opened[0]] = (opened[1], match.end())
            opened = None
    if opened:
        raise ValueError('章节实操标记不成对')
    return found


def strip(source):
    """先移除生成代码再排版；手写插入槽保留，重复构建不改变位置。"""
    for start, end in sorted(regions(source).values(), reverse=True):
        source = source[:start]+source[end:]
    return source


def render(demo):
    """说明和源码一起生成，读者可直接展开复制完整程序。"""
    code = (ROOT/demo['source']).read_text(encoding='utf-8').strip()
    return '\n'.join([
        f'<!-- chapter-demo: {demo["id"]} -->',
        f'{"### 专题实操" if demo.get("slot") else "## 本章实操"}｜{demo["title"]}', '',
        f'<a id="{demo.get("anchor", "chapter-demo")}"></a>', '',
        f'**实践目的：** {demo["purpose"]}', '',
        '**环境：** Python 3.10+、NumPy、PyTorch；支持 CPU/CUDA，Colab 选择 GPU 运行时。无需下载模型。在项目根目录执行：', '',
        '```bash', f'python experiments/chapter_demo_lab.py --chapter {demo["id"]} --device cuda',
        '# 没有GPU时显式使用CPU；auto在CUDA可用时优先使用GPU',
        f'python experiments/chapter_demo_lab.py --chapter {demo["id"]} --device cpu',
        '# 也可直接执行源码模块（同样在项目根目录）',
        f'python -m {demo["source"][:-3].replace("/", ".")}', '```', '',
        '**Colab：** 可直接上传[逐章 GPU Notebook](../experiments/notebooks/colab_chapter_demos.ipynb)，'
        '它自带所需源码；按单元选择本章运行。修改源码后将 `PRACTICE=True`，或命令行加 `--practice`；'
        '此模式跳过正文一致性检查，保留数值断言、设备核验及实际源码哈希。'
        '指定 `cuda` 后不可用会报错，不会静默退回 CPU。'
        '随机输入在 CPU 准备后一次搬到所选设备；矩阵、量化、训练和统计计算使用该设备的 PyTorch 张量。'
        '文件 I/O、Python 流程控制和打印在主机；默认 FP64 核验精度，不代表原生低比特 Tensor Core 性能。', '',
        f'**完整源码：** [带逐步注释的 demo](../{demo["source"]})。'
        '默认只打印结果；保存命令与全部章节入口见[实操导航](../README.md#chapter-demos)。', '',
        '<details><summary>展开完整代码：每一步的行为、计算逻辑与检查条件</summary>', '',
        f'<!-- chapter-demo-code: {demo["id"]} -->', '```python', code, '```', '', '</details>', '',
        f'**运行后观察：** {demo["observe"]}', '',
        f'**动手修改：** {demo["modify"]}', '',
        f'**适用边界：** {demo["boundary"]}',
        *(['真实模型/GPU流程继续使用 [Colab Notebook](../experiments/notebooks/colab_qwen3_quantization.ipynb)，'
           '执行范围见[模型实验验收](10_实验_评估与选型.md#model-validation-milestones)。']
          if demo['id'] in ('07', '09', '10', 'int8', 'research') else []),
        '', '<!-- /chapter-demo -->',
    ])


def merge(source, demo):
    """只替换自己的区域；缺失或重复标记报错，避免吞掉手写正文。"""
    existing = regions(source)
    if demo['id'] in existing:
        start, end = existing[demo['id']]
        return source[:start]+render(demo)+source[end:]
    if demo.get('slot'):
        marker = f'<!-- demo-slot: {demo["id"]} -->'
        if source.count(marker) != 1:
            raise ValueError(f'找不到唯一专题实操插入点: {demo["id"]}')
        return source.replace(marker, marker+'\n\n'+render(demo), 1)
    # 主线/附录/专题放在章测前；科研指南无章测，放在研究方法完成后。
    marker = ('## 7. 可信实验的详细清单' if demo['id'] == 'research'
              else '<!-- chapter-assessment -->')
    if source.count(marker) != 1:
        raise ValueError(f'找不到唯一实操插入点: {demo["document"]}')
    return source.replace(marker, render(demo)+'\n\n'+marker, 1)


def add_outputs(files):
    """与已有技术示例共享一个构建事务，先验证所有文件再写入。"""
    if len({d['id'] for d in DEMOS}) != len(DEMOS):
        raise ValueError('章节实操 ID 必须唯一')
    for demo in DEMOS:
        path = ROOT/demo['document']
        source = files[path] if path in files else path.read_text(encoding='utf-8')
        files[path] = merge(source, demo)
