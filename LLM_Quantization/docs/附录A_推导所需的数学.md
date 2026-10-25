# 附录 A｜看不懂公式时，先补这几件事

<!-- course-objectives -->
**学习目标：** 掌握矩阵方向、二阶统计和二次优化；区分范数与误差指标；识别期望、无偏和凸性结论的前提。

**先修知识：** 高中代数；可结合第 00–02 章的小矩阵例子使用。 [进入本章测试](#chapter-assessment)；建议先独立作答，再展开答案。
<!-- /course-objectives -->

不需要先学完一本线性代数教材。下面这些工具足以读懂本教程的大多数推导。

## A.1 向量、点积与矩阵乘法

一个向量就是按顺序排好的一组数，例如 $x=(2,3)$。两个同长度向量的点积，是对应位置相乘再求和：

$$
x\cdot w=2\times0.6+3\times(-1.2)=-2.4.
$$

神经网络的一次线性变换，就是很多个点积一起计算。

假设两条输入，每条有两个特征；三个输出神经元：

$$
X=\begin{bmatrix}1&2\\3&4\end{bmatrix},\qquad
W=\begin{bmatrix}1&0\\0&1\\1&-1\end{bmatrix}.
$$

本教程规定 $Y=XW^T$，于是：

$$
Y=\begin{bmatrix}1&2&-1\\3&4&-1\end{bmatrix}.
$$

形状是 $(2\times2)(2\times3)=2\times3$。前后两个内部维度必须相同。

转置 $W^T$ 把行变列。记住：

$$
(AB)^T=B^TA^T.
$$

转置后顺序反过来，是很多量化推导容易漏掉的细节。

## A.2 范数：如何把一堆误差变成一个数

向量 $e=(e_1,e_2)$ 的平方长度：

$$
\|e\|_2^2=e_1^2+e_2^2.
$$

矩阵的 Frobenius 范数平方，就是所有元素的平方和：

$$
\|E\|_F^2=\sum_{i,j}E_{ij}^2.
$$

例如误差 $(0.1,-0.2)$：平方和是 0.05，MSE 是 $0.05/2=0.025$。SSE 与 MSE 只差元素数量的归一化，报告数字时不能混用。

平方误差的好处是非负、可求导、大误差惩罚更强；它不是唯一正确的误差定义。

## A.3 为什么 $X^TX$ 能告诉我们输入相关性

对 $X\in\mathbb R^{n\times d}$：

$$
(X^TX)_{ij}=\sum_{t=1}^nX_{ti}X_{tj}.
$$

对角项 $(X^TX)_{ii}$ 是第 $i$ 个输入通道的平方和；非对角项是两个通道的乘积和。

如果两个通道经常同时变大、同时变小，乘积和往往较大。对于已经减去均值的输入，除以样本数后可解释为协方差；没有减均值时，应叫二阶矩，不应直接等同于协方差。

GPTQ 的 $X^TX$ 使用的是输入二阶统计。

## A.4 梯度：某个参数稍微变动，损失往哪边走

对单变量 $L(w)=(w-a)^2$：

$$
\frac{dL}{dw}=2(w-a).
$$

如果 $w>a$，导数为正，说明继续增加 $w$ 会增加损失；所以向反方向更新：

$$
w_{new}=w-\eta\frac{dL}{dw}.
$$

$\eta$ 是学习率。

多个变量的梯度，就是分别求偏导后排成向量。例如：

$$
L=\delta_1^2+\delta_1\delta_2+\delta_2^2,
$$

$$
\nabla L=\begin{bmatrix}2\delta_1+\delta_2\\\delta_1+2\delta_2\end{bmatrix}.
$$

它告诉我们所有方向上的局部变化趋势。

## A.5 Hessian：损失的弯曲程度

继续对梯度求导：

$$
H=\begin{bmatrix}2&1\\1&2\end{bmatrix}.
$$

这就是 Hessian。对上面的例子：

$$
L=\tfrac12\delta^TH\delta.
$$

如果 Hessian 是对角矩阵，每个坐标互不耦合；非零非对角项意味着一个坐标的变化会影响另一个坐标的最佳选择。

对更一般的平滑函数，当前位置附近有泰勒近似：

$$
L(w+\delta)\approx L(w)+g^T\delta+\tfrac12\delta^TH\delta.
$$

这是局部近似。但 GPTQ 使用的线性层输出平方误差本身就是二次函数，因此对固定输入，该二次展开是精确的。

## A.6 逆矩阵与数值求解

如果 $H^{-1}H=I$，那么 $H^{-1}$ 叫逆矩阵。方程 $H\delta=v$ 的解可以写成 $\delta=H^{-1}v$。

但写成逆矩阵，不代表代码应总显式计算逆。数值实现常优先使用线性求解或 Cholesky 分解，减少误差和开销。

对二维矩阵：

$$
\begin{bmatrix}a&b\\c&d\end{bmatrix}^{-1}
=\frac1{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix},
$$

前提是 $ad-bc\ne0$。但行列式的绝对值小，不足以判断相对求解是否病态：例如 $H=10^{-8}I_2$ 的行列式为 $10^{-16}$，二范数条件数却是 1。相对敏感性更应看 $\kappa_2(H)=\sigma_{max}/\sigma_{min}$，同时检查数据尺度和有限精度范围。对半正定的重建 Hessian 加 $\lambda I$（$\lambda>0$）可使其正定并控制条件数，也改变了优化目标；它不是对任意矩阵都适用的无条件修复。

## A.7 拉格朗日乘子为什么能处理 GPTQ 的约束

对可微函数，在定义域内部取得无约束局部最小值时，一阶必要条件是梯度为 0；这并不充分排除鞍点或极大值。这里讨论正定二次损失，严格凸性进一步保证其驻点是唯一全局最小值。

有约束 $e_i^T\delta=\epsilon_i$ 时，只允许沿约束平面移动。最优点不要求完整梯度为 0，而要求梯度在允许移动的方向上没有分量；因此梯度必须与约束平面的法向量 $e_i$ 平行：

$$
\nabla L+\lambda e_i=0.
$$

这就是拉格朗日乘子条件的几何含义。

例如不使用任何矩阵工具，直接解第 02 章的二维问题。令 $\delta_1=0.4$：

$$
L=0.16+0.4\delta_2+\delta_2^2.
$$

对 $\delta_2$ 求导：

$$
0.4+2\delta_2=0\Rightarrow\delta_2=-0.2.
$$

这和逆 Hessian 公式完全一致。高维公式只是把这个“固定一个，优化其他”的步骤推广到很多变量。

## A.8 特征向量与 SVD 的直觉

特征向量满足 $Hv=\lambda v$：矩阵作用后，这个方向仍沿自身，只改变长度。$\lambda$ 表示缩放程度。

对于一般矩阵，SVD 写成：

$$
E=U\Sigma V^T=\sum_i\sigma_i u_iv_i^T.
$$

每个 $u_iv_i^T$ 是一个 rank-1 矩阵，可以看成一块简单的方向性图案。$\sigma_i$ 越大，这个图案贡献越大。

只保留前 $r$ 个，就是用 $r$ 个最重要图案近似原矩阵。在 Frobenius 误差下，截断 SVD 是最佳 rank-$r$ 近似，剩余误差为：

$$
\|E-E_r\|_F^2=\sum_{i>r}\sigma_i^2.
$$

LoftQ 用低秩 adapter 近似量化残差，正是利用这个性质。但“最佳 Frobenius 近似”不等于“最佳下游任务 adapter”。

## A.9 期望与无偏估计

期望是按概率加权的平均。若一个随机输出 $\widehat x$ 满足 $\mathbb E[\widehat x]=x$，就叫对 $x$ 无偏。

例如为了表示 0.3，随机以 30% 概率输出 1、70% 概率输出 0，平均值为 0.3，但每次误差都不为零。

无偏不等于低方差，也不等于误差小。尤其不能随意把期望穿过非线性：

$$
\mathbb E[\widehat x^2]\ne(\mathbb E[\widehat x])^2.
$$

上述例子左边为 0.3，右边为 0.09。TurboQuant 内积无偏不推出 softmax 输出无偏，原理相同。

## A.10 读公式的固定步骤

1. 写出每个变量的含义和形状。
2. 判断哪些是固定值，哪些可以优化。
3. 写清目标：权重误差、输出误差，还是任务损失。
4. 每个等号后面说明依据：展开乘法、链式法则、求导为零，或约束条件。
5. 标记近似号和假设，不把统计近似当成恒等式。
6. 用二维例子或 NumPy 检查方向、常数和边界情况。

## A.11 谱范数与最坏误差方向

矩阵 $E$ 的谱范数是最大奇异值：$\|E\|_2=\max_{\|u\|=1}\|Eu\|$。
因此 $\|E\|_2^2=\lambda_{max}(E^TE)$，衡量最坏输入方向，而 Frobenius 范数平方是所有奇异值平方之和。
一个误差矩阵可以平均误差较小，却在少数方向上很大，这正是鲁棒量化要额外检查的问题。

若方向按 $u^TCu=1$、$C\succ0$ 约束，置 $v=C^{1/2}u$，就把问题变成
$\max_{\|v\|=1}\|EC^{-1/2}v\|^2$。这里的白化只是在不同坐标度量之间转换，不是丢弃特征值较小的方向。

## A.12 Jensen 不等式能说明什么

凸函数 $\phi$ 满足 $\phi(\sum_i a_ix_i)\le\sum_i a_i\phi(x_i)$，其中 $a_i\ge0$ 且 $\sum_i a_i=1$。
它可以证明同构凸代价在总量固定时由均衡分配达到下界，也可以给出
$\|\sum_i a_ie_i\|^2\le\sum_i a_i\|e_i\|^2$。

但若每一项使用不同函数 $\phi_i$，不能把它们直接合并成同一个 Jensen 结论。
MoE 的 gate 若不归一化或出现符号权重，也需要重新写明条件。第 13、14 章分别提供了反例与适用范围。


<!-- chapter-demo: math -->
## 本章实操｜用三条路径核验约束最小二乘

<a id="chapter-demo"></a>

**实践目的：** 将KKT、线性求解和可行点枚举对应到同一个二次目标，并检查二阶矩。

**环境：** Python 3.10+、NumPy、PyTorch；支持 CPU/CUDA，Colab 选择 GPU 运行时。无需下载模型。在项目根目录执行：

```bash
python experiments/chapter_demo_lab.py --chapter math --device cuda
# 没有GPU时显式使用CPU；auto在CUDA可用时优先使用GPU
python experiments/chapter_demo_lab.py --chapter math --device cpu
# 也可直接执行源码模块（同样在项目根目录）
python -m experiments.demos.appendix_math
```

**Colab：** 可直接上传[逐章 GPU Notebook](../experiments/notebooks/colab_chapter_demos.ipynb)，它自带所需源码；按单元选择本章运行。修改源码后将 `PRACTICE=True`，或命令行加 `--practice`；此模式跳过正文一致性检查，保留数值断言、设备核验及实际源码哈希。指定 `cuda` 后不可用会报错，不会静默退回 CPU。随机输入在 CPU 准备后一次搬到所选设备；矩阵、量化、训练和统计计算使用该设备的 PyTorch 张量。文件 I/O、Python 流程控制和打印在主机；默认 FP64 核验精度，不代表原生低比特 Tensor Core 性能。

**完整源码：** [带逐步注释的 demo](../experiments/demos/appendix_math.py)。默认只打印结果；保存命令与全部章节入口见[实操导航](../README.md#chapter-demos)。

<details><summary>展开完整代码：每一步的行为、计算逻辑与检查条件</summary>

<!-- chapter-demo-code: math -->
```python
"""数学附录：用 KKT 系统、逆矩阵公式与直接代入核对同一个约束问题。"""
import torch
from experiments.demo_runtime import get_runtime, same

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype

# 第 1 步：二次损失 L=1/2 delta^T H delta，固定第一个误差为0.3。
H = rt.tensor([[4., 2.], [2., 3.]])
selector = rt.tensor([[1.], [0.]])
required = rt.tensor([.3])
assert torch.all(torch.linalg.eigvalsh(H) > 0)  # 正定保证该约束问题唯一最优。

# 第 2 步：拉格朗日驻点和约束联合成线性方程，未知量是delta和乘子。
system = torch.cat([torch.cat([H, selector], dim=1),
                    torch.cat([selector.T, H.new_zeros((1, 1))], dim=1)], dim=0)
rhs = torch.cat([torch.zeros(2, device=device, dtype=dtype), required])
solution = torch.linalg.solve(system, rhs)
delta, multiplier = solution[:2], solution[2:]
assert same(H@delta+selector@multiplier, 0., allow_scalar_broadcast=True)
assert same(selector.T@delta, required)

# 第 3 步：另一种写法先求解 H v = e1，再按所需第一坐标重新缩放。
# 使用solve而非显式求逆；v正是逆矩阵的第一列。
column = torch.linalg.solve(H, selector).flatten()
closed_form = required[0]*column/column[0]
assert same(delta, closed_form)
assert same(delta, [.3, -.2])
loss = .5*delta@H@delta
assert same(loss, .12)
print('KKT解 / 乘子 / 最小损失:', delta, multiplier, loss)

# 第 4 步：沿所有可行点的参数delta2采样，独立检查没有比解析解更低的点。
second = torch.linspace(-1, 1, 201, device=device, dtype=dtype)
samples = torch.column_stack([torch.ones_like(second)*required[0], second])
losses = .5*torch.einsum('ni,ij,nj->n', samples, H, samples)
assert torch.all(losses >= loss-1e-12)

# 第 5 步：演示未中心化二阶矩和协方差的区别，解释GPTQ统计对象。
X = rt.tensor([[1., 2.], [2., 3.], [3., 4.]])
mean = X.mean(dim=0)
second_moment = X.T@X/len(X)
centered = X-mean
covariance = centered.T@centered/len(X)
assert same(second_moment, covariance+torch.outer(mean, mean))
print('二阶矩 / 中心化协方差:', second_moment, covariance)

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()
```

</details>

**运行后观察：** delta=[0.3,-0.2]、损失0.12；KKT残差为零，二阶矩=协方差+均值外积。

**动手修改：** 修改H并检查正定性，再改变约束值；奇异H需要重新定义求解方式，不能照搬逆。

**适用边界：** 小矩阵数值核验辅助理解，不能替代一般数学证明。

<!-- /chapter-demo -->

<!-- chapter-assessment -->
## A.13 本章测试与参考答案

<a id="chapter-assessment"></a>

核心卷 8 题，仍按 100 分；另有简单、困难、特别困难拓展题各 1 道，不计入核心卷。难度相对于本章先修知识划分，题号保持原有顺序。核心卷建议 60–90 分钟，特别困难拓展可另留 30–60 分钟。

| 难度 | 按难度选题 |
|---|---|
| 简单 | [A-01](#exercise-a-01) · [A-02](#exercise-a-02) · [A-03](#exercise-a-03) · [A-07](#exercise-a-07) · [A-E01](#exercise-a-e01) |
| 困难 | [A-04](#exercise-a-04) · [A-05](#exercise-a-05) · [A-06](#exercise-a-06) · [A-08](#exercise-a-08) · [A-E02](#exercise-a-e02) |
| 特别困难 | [A-E03](#exercise-a-e03) |

每题提供至少三条参考路径，分别可能是解析解、等价推导、独立数值核验或满足条件的工程方案。固定条件下唯一的数值结果不因解法不同而改变；数值实验不能替代一般性证明。未明确要求比较多解时，可任选或组合路径回答全部子问；验证说明须补足题干要求的推导。开放题接受条件明确、证据充分的其他方案。

<a id="exercise-a-01"></a>

### A-01｜矩阵方向与形状（简单｜计算，10 分）

$X=\begin{bmatrix}1&2\\0&-1\end{bmatrix}$、$W=\begin{bmatrix}2&0\\1&-1\\0&3\end{bmatrix}$。计算 $XW^T$ 并给出形状。为什么 $XW$ 不合法？

<details><summary>参考答案与评分要点</summary>

**参考路径一：**

$XW^T=\begin{bmatrix}2&-1&6\\0&1&-3\end{bmatrix}$，形状2×3。X为2×2、W为3×2，XW的内部维2和3不同；转置改变轴顺序，不是任意符号装饰。

**路径二：逐行内积。** 每行X分别与W三行点积，得到[2,−1,6]和[0,1,−3]；矩阵输出列对应不同输出神经元。

**路径三：线性组合列。** X的第一行是e₁+2e₂，因此输出是W第一列加两倍第二列；第二行是−e₂，所以为第二列取负。这也核对转置方向。

**评分要点：** 正确结果6分；形状与乘法条件4分。

</details>

<a id="exercise-a-02"></a>

### A-02｜二阶矩不等于协方差（简单｜计算，10 分）

两条样本为 (1,2)、(3,4)。按分母n=2计算均值、未中心化二阶矩和总体协方差，验证 $C=\mathbb E[x^Tx]-\mu^T\mu$。

<details><summary>参考答案与评分要点</summary>

**参考路径一：**

$\mu=(2,3)$；二阶矩 $\begin{bmatrix}5&7\\7&10\end{bmatrix}$，减去 $\begin{bmatrix}4&6\\6&9\end{bmatrix}$ 后，协方差全为1。若采用n−1无偏样本协方差约定，数值又不同；本题分母固定n。

**路径二：先中心化。** 两样本减μ后为(−1,−1)和(1,1)，各外积都是全1，平均仍全1。

**路径三：均值—残差分解。** 写x=μ+r且E[r]=0，展开E[xᵀx]=μᵀμ+E[rᵀr]，交叉项消失。这证明关系不只在本例成立。

**评分要点：** 均值2分；两个矩阵6分；关系与分母2分。

</details>

<a id="exercise-a-03"></a>

### A-03｜SSE、MSE、Frobenius 与谱范数（简单｜计算，15 分）

$E=\operatorname{diag}(3,4)$。求 SSE、四元素 MSE、Frobenius 范数和谱范数；哪个反映单位输入方向上的最大放大？

<details><summary>参考答案与评分要点</summary>

**参考路径一：**

SSE=25，MSE=25/4=6.25，Frobenius范数5，谱范数4。谱范数给出 $\max_{\|x\|=1}\|Ex\|$，而Frobenius汇总全部奇异值能量；均方与平方和也不能混用。

**路径二：奇异值。** 对角矩阵奇异值3、4，平方和25、最大4；Frobenius为√25=5，MSE再除四个元素。

**几何路径：** 单位圆经E变成半轴3和4的椭圆，最大长度4；总平方能量25不是最大方向幅值，所以两个范数回答不同问题。

**评分要点：** 四指标各2分；最坏方向与区别7分。

</details>

<a id="exercise-a-04"></a>

### A-04｜梯度和 Hessian（困难｜推导与计算，15 分）

$L(u,v)=2u^2+uv+3v^2$。求梯度、Hessian，并在 (1,−1) 上计算梯度和L；判断H是否正定。

<details><summary>参考答案与评分要点</summary>

**参考路径一：**

$\nabla L=(4u+v,u+6v)^T$；$H=\begin{bmatrix}4&1\\1&6\end{bmatrix}$。在(1,−1)，梯度(3,−5)，L=4；H首顺序主子式4>0、行列式23>0，故正定。

**解法二：矩阵二次型。** L=1/2 [u,v]H[u,v]ᵀ，H对称，梯度H[u,v]ᵀ；用该乘法核对(3,−5)。

**正定替代证明：** $L=2(u+v/4)^2+(23/8)v^2$，非零(u,v)时严格正，所以H正定，不必显式求特征值。

**评分要点：** 导数6分；点值5分；正定判断4分。

</details>

<a id="exercise-a-05"></a>

### A-05｜固定一个坐标后的优化（困难｜推导，15 分）

最小化 $L=2u^2+uv+3v^2$，约束u=0.5。求最优v、最小L，并解释为什么此时完整梯度不必为零。

<details><summary>参考答案与评分要点</summary>

**参考路径一：**

代入得 $L=0.5+0.5v+3v^2$，导数 $0.5+6v=0$，$v=-1/12$，最小 $23/48\approx0.479167$。只允许沿v移动，最优时该方向梯度为零，u方向由约束乘子抵消，不能要求所有分量都零。

**解法二：完成平方。** 固定u后L=$3(v+1/12)^2+23/48$，最优v与损失直接得出。

**解法三：KKT条件。** ∇L+λ(1,0)=0、u=0.5，第二行0.5+6v=0，第一行由λ补足。约束法向梯度不必为0。

**评分要点：** 代入与最优6分；损失5分；约束几何4分。

</details>

<a id="exercise-a-06"></a>

### A-06｜SVD 截断误差（困难｜计算与判断，15 分）

某矩阵奇异值为 [5,2,1]，rank-1 与 rank-2 最佳 Frobenius 近似的残余平方误差分别是多少？是否一定也是任务损失最优补偿？

<details><summary>参考答案与评分要点</summary>

**参考路径一：**

rank-1残余2²+1²=5，rank-2残余1。该最优性是固定矩阵、Frobenius度量下的秩约束近似；输入加权、下游非线性、因子量化和执行约束都可能改变目标。

**路径二：保留与总能量差。** 总能量25+4+1=30，rank1保留25剩5，rank2保留29剩1。

**验证路径：对角代表。** 对diag(5,2,1)直接删去小奇异方向，计算平方距离；一般矩阵通过正交不变性转到此坐标。这不是任意任务度量的最优性证明。

**评分要点：** 两残余6分；数学条件5分；任务与部署边界4分。

</details>

<a id="exercise-a-07"></a>

### A-07｜无偏不等于零误差（简单｜计算，10 分）

随机量 $\widehat x$ 以0.4概率为1、0.6概率为0，用来估计x=0.4。求均值、方差和 $\mathbb E[\widehat x^2]$；为何不能把平方移到期望外？

<details><summary>参考答案与评分要点</summary>

**参考路径一：**

均值0.4，方差0.24，平方期望0.4；均值平方0.16，两者差为方差0.24。非线性函数一般不能与期望交换，无偏不说明每次精确。

**解法二：二值幂恒等式。** 因xhat∈{0,1}，xhat²=xhat，所以平方期望0.4；均值平方0.16，相差0.24。

**解法三：中心化平方。** 误差为0.6（概率0.4）和−0.4（概率0.6），均方0.4×.36+0.6×.16=.24；无偏只消掉误差均值。

**评分要点：** 三统计6分；方差关系与非线性4分。

</details>

<a id="exercise-a-08"></a>

### A-08｜Jensen 与混合专家误差（困难｜推导条件，10 分）

说明 $\|\sum_i g_ie_i\|^2\le\sum_i g_i\|e_i\|^2$ 所需条件。若g=(2,−1)，给一个标量反例。

<details><summary>参考答案与评分要点</summary>

**参考路径一：**

需要 $g_i\ge0$ 且和为1，用平方范数凸性。取 $e_1=0,e_2=1$，左侧1，右侧−1，不满足不等式。带符号或未归一化权重应重新求界，不能直接套用概率混合结论。

**证明二：方差恒等式。** 在非负归一gate下，$\sum g_i\|e_i\|^2-\|\sum g_ie_i\|^2=\frac12\sum_{i,j}g_ig_j\|e_i-e_j\|^2\ge0$。

**反例路径：** g=(2,−1)虽和1却有负权，右侧不再保证非负；取题中e₁=0,e₂=1得−1<1。若权重全正但和不为1，应先归一化再恢复总尺度。

**评分要点：** 条件和凸性5分；反例4分；适用边界1分。

</details>

<a id="exercise-a-e01"></a>

### A-E01｜两个误差指标的单位（简单｜拓展题，单独计分）

误差向量(0.1,−0.3)，求SSE、MSE和L2范数；为什么MSE不是L2范数？

<details><summary>参考答案与评分要点</summary>

**解法一：定义。** SSE0.1，MSE0.05，L2=√0.1≈0.316228。

**解法二：换算。** MSE×元素数=SSE，L2平方=SSE；两种操作不同。

**量纲路径：** SSE/MSE都是原量纲的平方，L2保原量纲。不同元素数时SSE与MSE的归一化也影响比较。

**评分要点：** 单独按 10 分评价：结论与条件 3 分、推理或设计过程 5 分、独立核验与边界 2 分。评分覆盖题干要求的全部子问；未要求比较多解时，可任选或组合参考路径完成作答。数值核验不能替代题干明确要求的证明，多解数量不额外加分。

</details>

<a id="exercise-a-e02"></a>

### A-E02｜半正定矩阵与正定求解（困难｜拓展题，单独计分）

X=[[1,1],[2,2]]。求XᵀX的秩和特征值，判断是否可逆；加入0.1I后怎样变化？

<details><summary>参考答案与评分要点</summary>

**解法一：直接矩阵。** XᵀX=[[5,5],[5,5]]，秩1，特征值10、0，不可逆；加阻尼后为10.1、0.1，可逆。

**解法二：和差方向。** (1,1)方向被放大10，(1,−1)被消掉；阻尼对每个方向加0.1。

**几何路径：** 两列输入完全相同，数据无法区分某些权重变化；阻尼引入正则选择，并不新增缺失数据。解算改善不等于统计信息补全。

**评分要点：** 单独按 10 分评价：结论与条件 3 分、推理或设计过程 5 分、独立核验与边界 2 分。评分覆盖题干要求的全部子问；未要求比较多解时，可任选或组合参考路径完成作答。数值核验不能替代题干明确要求的证明，多解数量不额外加分。

</details>

<a id="exercise-a-e03"></a>

### A-E03｜多个线性约束下的最小二次误差（特别困难｜拓展题，单独计分）

H正定，S满列秩，求最小化 $\frac12\delta^TH\delta$ 且 $S^T\delta=\epsilon$ 的解与最小损失。至少用KKT和白化几何解释，并说明S不满秩时哪一步需要改变。

<details><summary>参考答案与评分要点</summary>

**证明一：KKT。** Hδ+Sλ=0，代约束得 $\lambda=-(S^TH^{-1}S)^{-1}\epsilon$；故 $\delta^*=H^{-1}S(S^TH^{-1}S)^{-1}\epsilon$，最小损失 $\frac12\epsilon^T(S^TH^{-1}S)^{-1}\epsilon$。

**证明二：最小范数。** 置v=H^{1/2}δ、B=H^{-1/2}S，目标为||v||²/2、约束Bᵀv=ε。最优v在B列空间，写v=Bα，解BᵀBα=ε，再回原坐标。

**退化与核验：** S列相关时SᵀH⁻¹S不再可逆，需先检查约束相容、删除冗余或使用恰当广义逆。可用小矩阵验证约束和驻点，但不能在不相容约束上盲用逆矩阵公式。

**评分要点：** 单独按 10 分评价：结论与条件 3 分、推理或设计过程 5 分、独立核验与边界 2 分。评分覆盖题干要求的全部子问；未要求比较多解时，可任选或组合参考路径完成作答。数值核验不能替代题干明确要求的证明，多解数量不额外加分。

</details>

**复习建议：** 计算错误先回到本章的形状、单位与约定；工程判断缺少依据时，补写目标函数、对照配置和验证边界。建议达到 80 分后继续；这是一项自学建议，不是经教学验证的能力认证标准。

[课程考核与综合练习](11_练习与答案.md#course-assessment-guide) · [返回学习目录](../README.md)

<!-- /chapter-assessment -->


---

[返回学习目录](../README.md)
