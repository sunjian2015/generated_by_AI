"""快速冒烟测试: 验证 gaussian_splatting_min.py 的核心函数可微且能收敛。"""
import torch
import torch.nn.functional as F
from gaussian_splatting_min import GaussianModel, render, look_at, build_covariance_3d

device = "cpu"
H = W = 48
fx = fy = 40.0
cx, cy = W / 2, H / 2

# 目标: 彩色圆盘
ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
r2 = xs ** 2 + ys ** 2
target = torch.zeros(H, W, 3)
mask = r2 < 0.5
target[..., 0] = torch.where(mask, torch.tensor(0.9), torch.tensor(0.05))
target[..., 1] = torch.where(mask, torch.tensor(0.4), torch.tensor(0.05))
target[..., 2] = torch.where(mask, torch.tensor(0.2), torch.tensor(0.1))

g = GaussianModel(num_points=60, device=device)
with torch.no_grad():
    g.xyz[:] = torch.randn(60, 3) * 0.6
    g.xyz[:, 2] += 2.0

view = look_at([0, 0, 0], [0, 0, 1], [0, 1, 0], device)
opt = torch.optim.Adam(g.parameters(), lr=0.03)

losses = []
for step in range(80):
    img = render(g, view, fx, fy, cx, cy, W, H)
    loss = F.l1_loss(img, target)
    opt.zero_grad()
    loss.backward()
    # 验证梯度确实回传到了各参数
    assert g.xyz.grad is not None and g.xyz.grad.abs().sum() > 0, "xyz 无梯度!"
    assert g.quat.grad is not None, "quat 无梯度!"
    opt.step()
    losses.append(loss.item())
    if step % 20 == 0:
        print(f"step {step:3d}  loss {loss.item():.5f}")

print(f"final loss {losses[-1]:.5f}  (start {losses[0]:.5f})")
assert losses[-1] < losses[0], "loss 未下降!"
# 也顺带检查协方差对称正定
Sigma = build_covariance_3d(g.scale, g.quat)
sym_err = (Sigma - Sigma.transpose(1, 2)).abs().max().item()
eig_min = torch.linalg.eigvalsh(Sigma).min().item()
print(f"cov symmetric err={sym_err:.2e}  min eigenvalue={eig_min:.3e} (>0 表示正定)")
print("SMOKE TEST PASSED")
