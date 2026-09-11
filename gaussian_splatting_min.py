"""
Minimal 3D Gaussian Splatting in pure PyTorch (teaching implementation).

这是一个教学级实现，用纯 PyTorch 完成可微的高斯 splatting 前向渲染与优化，
帮助理解 3DGS 的核心数学。它不使用 CUDA tile 光栅化，因此不追求速度，
但完整覆盖了：协方差参数化 -> 投影 -> 2D 高斯 -> 深度排序 -> alpha 混合。

依赖: torch (>=1.12)
可选: numpy, imageio (仅用于保存图像)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def quaternion_to_rotation(q: torch.Tensor) -> torch.Tensor:
    """四元数 (N,4) -> 旋转矩阵 (N,3,3)。q 顺序为 (w, x, y, z)。"""
    q = F.normalize(q, dim=-1)  # 保证单位四元数
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    N = q.shape[0]
    R = torch.empty(N, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_covariance_3d(scale: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """由缩放 (N,3) 与四元数 (N,4) 构造 3D 协方差 Sigma = R S S^T R^T, 形状 (N,3,3)。"""
    R = quaternion_to_rotation(quat)            # (N,3,3)
    S = torch.diag_embed(scale)                 # (N,3,3)
    M = R @ S                                    # (N,3,3)
    Sigma = M @ M.transpose(1, 2)                # (N,3,3)
    return Sigma


# ---------------------------------------------------------------------------
# 高斯模型
# ---------------------------------------------------------------------------
class GaussianModel(nn.Module):
    """一组可优化的各向异性 3D 高斯。"""

    def __init__(self, num_points: int, device="cpu"):
        super().__init__()
        # 位置 mean
        self.xyz = nn.Parameter(torch.randn(num_points, 3, device=device) * 0.5)
        # 缩放: 用 log 参数化, 取 exp 后恒为正
        self.log_scale = nn.Parameter(
            torch.log(torch.ones(num_points, 3, device=device) * 0.1)
        )
        # 旋转四元数 (w,x,y,z), 初始化为单位旋转
        quat = torch.zeros(num_points, 4, device=device)
        quat[:, 0] = 1.0
        self.quat = nn.Parameter(quat)
        # 不透明度: 用 logit 参数化, sigmoid 后落在 (0,1)
        self.opacity_logit = nn.Parameter(torch.zeros(num_points, 1, device=device))
        # 颜色 (这里用简化的 RGB, 不展开完整球谐), logit -> sigmoid
        self.color_logit = nn.Parameter(torch.randn(num_points, 3, device=device))

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    @property
    def opacity(self):
        return torch.sigmoid(self.opacity_logit)

    @property
    def color(self):
        return torch.sigmoid(self.color_logit)


# ---------------------------------------------------------------------------
# 可微渲染
# ---------------------------------------------------------------------------
def render(
    gaussians: GaussianModel,
    view_matrix: torch.Tensor,   # (4,4) world->camera
    fx: float, fy: float,        # 焦距(像素)
    cx: float, cy: float,        # 主点
    width: int, height: int,
    bg_color: torch.Tensor = None,
):
    """
    把 3D 高斯渲染成图像 (H,W,3)。完整流程:
      1. 变换到相机坐标
      2. 透视投影得到 2D 中心 mu'
      3. 用 EWA 雅可比把 3D 协方差投影为 2D 协方差 Sigma'
      4. 按深度排序
      5. 对每个像素做 alpha 混合
    """
    device = gaussians.xyz.device
    if bg_color is None:
        bg_color = torch.zeros(3, device=device)

    xyz = gaussians.xyz                           # (N,3)
    N = xyz.shape[0]

    # --- 1. world -> camera ---
    R = view_matrix[:3, :3]                       # (3,3)
    t = view_matrix[:3, 3]                        # (3,)
    cam_xyz = xyz @ R.T + t                        # (N,3)
    depth = cam_xyz[:, 2]                           # (N,) 相机空间 z

    # 只保留相机前方的点
    front_mask = depth > 1e-4

    # --- 2. 透视投影: 屏幕坐标 (像素) ---
    inv_z = 1.0 / torch.clamp(depth, min=1e-4)
    u = fx * cam_xyz[:, 0] * inv_z + cx            # (N,)
    v = fy * cam_xyz[:, 1] * inv_z + cy            # (N,)
    mu2d = torch.stack([u, v], dim=-1)             # (N,2)

    # --- 3. 3D 协方差投影为 2D 协方差 ---
    Sigma3d = build_covariance_3d(gaussians.scale, gaussians.quat)  # (N,3,3)
    # 先把协方差旋到相机坐标系: Sigma_cam = R Sigma R^T
    Sigma_cam = R @ Sigma3d @ R.T                   # 广播: (3,3)@(N,3,3)@(3,3)->(N,3,3)

    # 投影雅可比 J (EWA splatting 的仿射近似), 对每个高斯不同
    x, y, z = cam_xyz[:, 0], cam_xyz[:, 1], torch.clamp(depth, min=1e-4)
    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * x / (z * z)
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * y / (z * z)

    Sigma2d = J @ Sigma_cam @ J.transpose(1, 2)     # (N,2,2)
    # 加一点正则保证可逆 (对应屏幕上最小 1 像素的模糊)
    Sigma2d[:, 0, 0] += 0.3
    Sigma2d[:, 1, 1] += 0.3

    # 2D 协方差求逆 (解析公式, 比 torch.inverse 稳定)
    a = Sigma2d[:, 0, 0]
    b = Sigma2d[:, 0, 1]
    c = Sigma2d[:, 1, 1]
    det = a * c - b * b
    det = torch.clamp(det, min=1e-8)
    inv = torch.zeros_like(Sigma2d)
    inv[:, 0, 0] = c / det
    inv[:, 0, 1] = -b / det
    inv[:, 1, 0] = -b / det
    inv[:, 1, 1] = a / det                          # (N,2,2) = Sigma'^{-1}

    opacity = gaussians.opacity.squeeze(-1)          # (N,)
    color = gaussians.color                          # (N,3)

    # --- 4. 按深度排序 (从近到远) ---
    order = torch.argsort(depth)
    # 把无效(相机后方)的高斯放到最后: 用大深度替换
    valid = front_mask[order]

    # --- 5. 逐像素 alpha 混合 ---
    # 构造像素网格
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pix = torch.stack([xs, ys], dim=-1)              # (H,W,2)

    out = torch.zeros(height, width, 3, device=device)
    T = torch.ones(height, width, device=device)     # 累积透射率 prod(1-alpha)

    # 为了内存可控, 逐个高斯前向合成 (已按深度排序)
    for idx in order:
        if not front_mask[idx]:
            continue
        d = pix - mu2d[idx]                          # (H,W,2) 像素到高斯中心
        S_inv = inv[idx]                             # (2,2)
        # 马氏距离平方: d^T Sigma'^{-1} d
        power = (
            d[..., 0] * (S_inv[0, 0] * d[..., 0] + S_inv[0, 1] * d[..., 1])
            + d[..., 1] * (S_inv[1, 0] * d[..., 0] + S_inv[1, 1] * d[..., 1])
        )
        g = torch.exp(-0.5 * power)                  # (H,W) 高斯权重
        alpha = torch.clamp(opacity[idx] * g, max=0.999)  # 有效不透明度
        w = alpha * T                                 # 该高斯对像素的贡献权重
        out = out + w.unsqueeze(-1) * color[idx]      # 累加颜色
        T = T * (1 - alpha)                           # 更新透射率

    # 背景合成
    out = out + T.unsqueeze(-1) * bg_color
    return out


# ---------------------------------------------------------------------------
# 简单训练演示: 用随机高斯拟合一张目标图像
# ---------------------------------------------------------------------------
def look_at(eye, target, up, device):
    """构造 world->camera 的 view matrix (4,4)。"""
    eye = torch.tensor(eye, dtype=torch.float32, device=device)
    target = torch.tensor(target, dtype=torch.float32, device=device)
    up = torch.tensor(up, dtype=torch.float32, device=device)
    f = F.normalize(target - eye, dim=0)             # 前向
    r = F.normalize(torch.cross(f, up, dim=0), dim=0)
    u = torch.cross(r, f, dim=0)
    R = torch.stack([r, u, f], dim=0)                # camera 轴 (行)
    t = -R @ eye
    V = torch.eye(4, device=device)
    V[:3, :3] = R
    V[:3, 3] = t
    return V


def demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = W = 128
    fx = fy = 100.0
    cx, cy = W / 2, H / 2

    # 目标图像: 一个彩色圆盘
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    r2 = xs ** 2 + ys ** 2
    target = torch.zeros(H, W, 3, device=device)
    mask = r2 < 0.5
    target[..., 0] = torch.where(mask, 0.9, 0.05)
    target[..., 1] = torch.where(mask, 0.3 + 0.5 * xs.abs(), 0.05)
    target[..., 2] = torch.where(mask, 0.2, 0.1)

    gaussians = GaussianModel(num_points=200, device=device)
    # 把高斯初始化在相机前方 z=2 附近
    with torch.no_grad():
        gaussians.xyz[:] = torch.randn(200, 3, device=device) * 0.6
        gaussians.xyz[:, 2] += 2.0

    view = look_at(eye=[0, 0, 0], target=[0, 0, 1], up=[0, 1, 0], device=device)

    opt = torch.optim.Adam(gaussians.parameters(), lr=0.02)

    for step in range(300):
        img = render(gaussians, view, fx, fy, cx, cy, W, H)
        loss = F.l1_loss(img, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.5f}")

    print("训练完成。final loss =", loss.item())

    # 可选: 保存结果
    try:
        import numpy as np
        import imageio
        out = (img.clamp(0, 1).detach().cpu().numpy() * 255).astype("uint8")
        tgt = (target.clamp(0, 1).detach().cpu().numpy() * 255).astype("uint8")
        imageio.imwrite("gs_render.png", out)
        imageio.imwrite("gs_target.png", tgt)
        print("已保存 gs_render.png 与 gs_target.png")
    except ImportError:
        print("(安装 imageio 可保存图像结果)")


if __name__ == "__main__":
    demo()
