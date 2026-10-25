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
