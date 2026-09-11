'''论文：https://arxiv.org/pdf/2605.30073'''
"""NAVA 核心思想的最小教学复现（非官方 6B 完整实现）。"""
from dataclasses import dataclass
from typing import Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def cross_modal_mask(nv: int, na: int, device) -> Tensor:
    """屏蔽视频与音频之间的注意力，用于构造“无对齐条件”分支。"""
    mask = torch.zeros(nv + na, nv + na, dtype=torch.bool, device=device)
    mask[:nv, nv:] = True
    mask[nv:, :nv] = True
    return mask


class FFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HAL(nn.Module):
    """Hybrid Alignment Layer：模态独立投影 + 联合自注意力。"""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.v_norm, self.a_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.v_proj, self.a_proj = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.joint_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ctx_norm = nn.LayerNorm(dim)
        self.v_cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.a_cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.v_ffn_norm, self.a_ffn_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.v_ffn, self.a_ffn = FFN(dim), FFN(dim)

    def forward(self, v: Tensor, a: Tensor, c: Tensor, drop_alignment=False):
        # v:[B,Nv,D], a:[B,Na,D], c:[B,Nc,D]
        nv, na = v.shape[1], a.shape[1]
        x = torch.cat([self.v_proj(self.v_norm(v)), self.a_proj(self.a_norm(a))], 1)
        mask = cross_modal_mask(nv, na, x.device) if drop_alignment else None
        dx, _ = self.joint_attn(x, x, x, attn_mask=mask, need_weights=False)
        v, a = v + dx[:, :nv], a + dx[:, nv:]

        c = self.ctx_norm(c)
        dv, _ = self.v_cross(v, c, c, need_weights=False)
        da, _ = self.a_cross(a, c, c, need_weights=False)
        v, a = v + dv, a + da
        return v + self.v_ffn(self.v_ffn_norm(v)), a + self.a_ffn(self.a_ffn_norm(a))


class UFL(nn.Module):
    """Unified Fusion Layer：音视频共享参数，进行高层语义融合。"""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1, self.norm2, self.ctx_norm = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn_norm, self.ffn = nn.LayerNorm(dim), FFN(dim)

    def forward(self, v: Tensor, a: Tensor, c: Tensor):
        nv = v.shape[1]
        x = torch.cat([v, a], 1)
        q = self.norm1(x)
        dx, _ = self.self_attn(q, q, q, need_weights=False)
        x = x + dx
        dc, _ = self.cross_attn(self.norm2(x), self.ctx_norm(c), self.ctx_norm(c), need_weights=False)
        x = x + dc
        x = x + self.ffn(self.ffn_norm(x))
        return x[:, :nv], x[:, nv:]


@dataclass
class Output:
    video_velocity: Tensor
    audio_velocity: Tensor


class MinimalNAVA(nn.Module):
    def __init__(self, dv=96, da=64, dc=128, dim=256, heads=8, n_hal=2, n_ufl=2):
        super().__init__()
        self.v_in, self.a_in, self.c_in = nn.Linear(dv, dim), nn.Linear(da, dim), nn.Linear(dc, dim)
        self.time = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.hal = nn.ModuleList([HAL(dim, heads) for _ in range(n_hal)])
        self.ufl = nn.ModuleList([UFL(dim, heads) for _ in range(n_ufl)])
        self.v_out, self.a_out = nn.Linear(dim, dv), nn.Linear(dim, da)

    def forward(self, zv: Tensor, za: Tensor, context: Tensor, t: Tensor, drop_alignment=False):
        # zv:[B,Nv,dv], za:[B,Na,da], context:[B,Nc,dc], t:[B]
        te = self.time(t[:, None])[:, None]
        v, a, c = self.v_in(zv) + te, self.a_in(za) + te, self.c_in(context)
        for block in self.hal:
            v, a = block(v, a, c, drop_alignment)
        for block in self.ufl:
            v, a = block(v, a, c)
        return Output(self.v_out(v), self.a_out(a))


def insert_timbre_tokens(text: Tensor, timbre: Tensor, insert_after: Sequence[int]) -> Tensor:
    """把说话人音色 token 插入对应台词 span 之后。"""
    assert timbre.shape[1] == len(insert_after)
    pieces, start = [], 0
    for i, pos in enumerate(insert_after):
        pieces += [text[:, start:pos + 1], timbre[:, i:i + 1]]
        start = pos + 1
    pieces.append(text[:, start:])
    return torch.cat(pieces, dim=1)


def factorized_cfg(full: Output, no_text: Output, no_align: Output, no_timbre: Output,
                   s_text=5.0, s_align=5.0, s_timbre=5.0) -> Output:
    """论文式思想：分别缩放文本、对齐、音色三个引导方向。"""
    def mix(y, yt, ya, ys):
        return yt + s_text * (y - yt) + s_align * (y - ya) + s_timbre * (y - ys)
    return Output(
        mix(full.video_velocity, no_text.video_velocity, no_align.video_velocity, no_timbre.video_velocity),
        mix(full.audio_velocity, no_text.audio_velocity, no_align.audio_velocity, no_timbre.audio_velocity),
    )


def flow_matching_loss(model: MinimalNAVA, clean_v: Tensor, clean_a: Tensor, context: Tensor):
    """假设：采用直线路径 rectified-flow 的最小近似。"""
    b = clean_v.shape[0]
    t = torch.rand(b, device=clean_v.device)
    eps_v, eps_a = torch.randn_like(clean_v), torch.randn_like(clean_a)
    zt_v = (1 - t[:, None, None]) * clean_v + t[:, None, None] * eps_v
    zt_a = (1 - t[:, None, None]) * clean_a + t[:, None, None] * eps_a
    pred = model(zt_v, zt_a, context, t)
    return F.mse_loss(pred.video_velocity, eps_v - clean_v) + F.mse_loss(pred.audio_velocity, eps_a - clean_a)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MinimalNAVA(dim=128, heads=4)
    v, a = torch.randn(2, 12, 96), torch.randn(2, 20, 64)
    text, timbre = torch.randn(2, 16, 128), torch.randn(2, 2, 128)
    context = insert_timbre_tokens(text, timbre, [4, 10])
    out = model(v, a, context, torch.rand(2))
    loss = flow_matching_loss(model, v, a, context)
    print(out.video_velocity.shape, out.audio_velocity.shape, float(loss))
