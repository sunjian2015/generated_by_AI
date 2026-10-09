"""
TDM (Trajectory Distribution Matching, ICCV 2025) 核心算法精简版 —— 仅保留算法逻辑，不可直接运行。
论文: https://arxiv.org/abs/2503.06674   提炼自 train_tdm_demo.py

=============================================================================================
关键参数示例 (对应 README 的训练命令: --cfg 4.5 --total_steps 900 --use_separate --use_huber)
=============================================================================================
  total_steps = 900        # 截断的最大时刻 T。注意不是 1000：TDM 不用最噪的那一段
  K (steps)   = 4          # 生成器的少步数 (4 NFE)
  cfg         = 4.5        # 蒸馏进生成器的教师引导强度 (推理时 guidance_scale=1)
  bsz         = 16, latent = [bsz, 4, 64, 64]   # PixArt-512 + VAE 下采样 8x
  lr          = 2e-5 (fake score) / 2e-5 / 5 (生成器)，betas=(0., 0.95)
  DDPM beta   : linear, beta_start=1e-4, beta_end=0.02, 1000 步, prediction_type='epsilon'

  alphas / sigmas 就是 DDPM 的 sqrt(alphas_cumprod) 与 sqrt(1-alphas_cumprod)，长度 1000：
      alphas = sqrt(cumprod(1 - linspace(1e-4, 0.02, 1000)))
      sigmas = sqrt(1 - alphas**2)
  实际数值 (x_t = alphas[t] * x_0 + sigmas[t] * eps)：
      t=  0   alphas=0.9999  sigmas=0.0100   SNR=9999      几乎干净
      t= 20   alphas=0.9969  sigmas=0.0791   SNR=158.6     代码里的时刻下界
      t=224   alphas=0.7690  sigmas=0.6392   SNR=1.447
      t=449   alphas=0.3564  sigmas=0.9344   SNR=0.1455
      t=674   alphas=0.0993  sigmas=0.9951   SNR=0.00995
      t=899   alphas=0.0166  sigmas=0.9999   SNR=0.000275  轨迹起点，近似纯噪声

  K=4 / total_steps=900 时的离散时刻 (t_k = k * 900//4 - 1)：
      k:       1     2     3     4
      t_k:   224   449   674   899        <- 第 k 段的起点 (轨迹点 x_{t_k})
      t_mid:   0   225   450   675        <- 第 k 段的终点 = t_k - 225 + 1
  生成器采样轨迹: 899 -> 674 -> 449 -> 224 -> x_0

=============================================================================================
TDM vs DMD2 总览 (DMD2 一侧依据其论文/官方实现，本仓库内无 DMD2 代码)
=============================================================================================
                    | DMD2                                  | TDM
  ------------------+---------------------------------------+------------------------------------
  匹配的对象         | 生成器的**终点** x_0 加噪后的分布        | 轨迹**每一段的转移**分布 q(x_{t_{k-1}})
  x_t 的构造         | x_t = a_t·G(...)+ s_t·eps, eps 全新高斯 | G 走一步 ODE 到 t_mid (保留模型噪声),
                    |                                       | 再从 t_mid 加噪到 t
  x_t 的噪声         | 纯标准高斯                             | 模型噪声与新噪声的**混合** -> 非标准高斯
  重要性权重         | 不需要                                 | **需要** is_weight 修正密度偏移
  真实图像           | **需要** (GAN loss 的判别器分支)        | 完全不需要 (image-free)
  额外损失           | distribution matching + **GAN loss**   | 只有 distribution matching
  更新比例           | two time-scale，fake:gen ≈ 5:1         | 1:1
  t 的采样范围       | 全局共享 [t_min, t_max]                | 按段隔离 [t_mid, t_k] (--use_separate)
  fake score 加权    | 普通去噪 MSE                           | min-SNR(5) × is_weight
  生成器损失         | MSE 到 (x0 - grad).detach()            | 同一形式，可选 Huber (--use_huber)
  ------------------+---------------------------------------+------------------------------------
  一句话: DMD2 只约束"生成器最终生成什么"，TDM 约束"生成器每一步怎么走"，
          监督更密集 -> 500 迭代 / 2 A800 小时即可超过教师，且无需任何图像。

三个网络:
  unet_sd   : 冻结的教师 (real score)，多步扩散模型
  unet      : 待训练的少步生成器 G，从教师初始化
  unet_fake : 待训练的 fake score，拟合当前生成器分布的 score
"""

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- 基础工具
def extract_into_tensor(a, t, x_shape):
    """按 timestep 取 schedule 值并 reshape 成 [bsz,1,1,1] 以便广播"""
    b, *_ = t.shape
    return a.gather(-1, t).reshape(b, *((1,) * (len(x_shape) - 1)))


def predicted_origin(model_output, timesteps, sample, alphas, sigmas):
    """由 eps-prediction 反解 x0:  x0 = (x_t - sigma_t * eps) / alpha_t
    注意该式对 eps 是仿射的，所以 "先对 eps 做 CFG 再解 x0" 等价于 "先解 x0 再做 CFG"。"""
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    return (sample - sigmas * model_output) / alphas


def compute_snr(scheduler, timesteps):
    """SNR = (alpha_t / sigma_t)^2 = ac / (1 - ac)。例: t=224 -> 1.447, t=674 -> 0.00995

    diffusers.training_utils.compute_snr 写成 (sqrt(ac)/sqrt(1-ac))**2，是为了让
    alpha/sigma 与前向过程 x_t = alpha_t*x_0 + sigma_t*eps 里的记号字面对应；
    这里直接用等价的方差比，数值上完全一致 (float32 下最大相对差 2.8e-7，纯舍入)。
    """
    ac = scheduler.alphas_cumprod.to(timesteps.device)
    return ac[timesteps] / (1 - ac[timesteps])


# ----------------------------------------------------------------------------- 1. 生成器自采样整条轨迹
@torch.no_grad()
def generate_trajectory(unet, scheduler, noise, cond, mask, steps, total_steps, alphas, sigmas):
    """
    用当前生成器 G 自采样一条 K 步轨迹，返回轨迹上所有点。这些点是后续"分布匹配的起点"。
    steps=4, total_steps=900 时: t 依次为 899 -> 674 -> 449 -> 224
    返回 [x_899, x_674, x_449, x_224, x_0]

    vs DMD2: DMD2 的 "backward simulation" 也会模拟自己的轨迹来取得中间输入 x_{t_k}，
             这一步两者思路一致 (都为了消除 train/test 输入不匹配)。
             区别在下面 sample_segment —— 拿到 x_{t_k} 之后用它做什么。
    """
    t = torch.full((noise.shape[0],), total_steps - 1, device=noise.device).long()   # 899
    x_t = noise
    traj = []
    for _ in range(steps):
        traj.append(x_t)
        eps = unet(x_t, timestep=t, encoder_hidden_states=cond, encoder_attention_mask=mask)[0]
        eps = eps.chunk(2, dim=1)[0]                 # PixArt 输出 [eps, var] 共 8 通道，只取 eps
        x_0 = predicted_origin(eps, t, x_t, alphas, sigmas)
        t = t - total_steps // steps                 # 899 -> 674 -> 449 -> 224
        x_t = scheduler.add_noise(x_0, eps, t)       # 用模型自己预测的 eps 重加噪 = 确定性 ODE 步
    traj.append(x_0)
    return traj


# ----------------------------------------------------------------------------- 2. 单步预测 / 段间加噪
class Predictor:
    def __init__(self, scheduler, alphas, sigmas, uncond_emb, uncond_mask):
        self.sch, self.alphas, self.sigmas = scheduler, alphas, sigmas
        self.uncond_emb, self.uncond_mask = uncond_emb, uncond_mask

    def predict(self, model, x_t, t, cond, mask, cfg=None):
        """一次前向 -> (eps, x0)。cfg 不为 None 时做 classifier-free guidance"""
        eps = model(x_t, timestep=t, encoder_hidden_states=cond,
                    encoder_attention_mask=mask)[0].chunk(2, dim=1)[0]
        if cfg is not None:
            eps_u = model(x_t, timestep=t, encoder_hidden_states=self.uncond_emb,
                          encoder_attention_mask=self.uncond_mask)[0].chunk(2, dim=1)[0]
            eps = eps_u + cfg * (eps - eps_u)
        return eps, predicted_origin(eps, t.long(), x_t, self.alphas, self.sigmas)

    def add_noise(self, x, noise, t1, t2):
        """
        从**已在 t1** 的样本继续加噪到 t2 (t2 > t1)，保持前向边缘分布一致:
            x_{t2} = (a2/a1)·x_{t1} + sqrt(s2^2 - (a2/a1·s1)^2)·noise
        例 t1=225, t2=674: a1=0.7673,s1=0.6413 -> a2=0.0993,s2=0.9951
            系数 a2/a1=0.1294，beta=sqrt(0.9951^2-(0.1294*0.6413)^2)=0.9917

        vs DMD2: DMD2 直接从干净的 x_0 加噪 (等价于 t1=0 的特例)，用的是
                 scheduler.add_noise(x_0, fresh_noise, t)。TDM 必须支持 t1>0，
                 因为它的起点是轨迹中间点 x_{t_mid} 而非干净样本。
        """
        a1, s1 = extract_into_tensor(self.alphas, t1, x.shape), extract_into_tensor(self.sigmas, t1, x.shape)
        a2, s2 = extract_into_tensor(self.alphas, t2, x.shape), extract_into_tensor(self.sigmas, t2, x.shape)
        beta = (s2 ** 2 - (a2 / a1 * s1) ** 2) ** 0.5
        return x / a1 * a2 + beta * noise

    def mixed_noise(self, model_eps, noise, t1, t2):
        """
        x_{t2} 中等效的"总噪声" = (模型噪声被搬运过来的部分 + 新采噪声) / s2。
        由于 model_eps 是网络输出、不是标准高斯，这个混合噪声的密度偏离 N(0,I)，
        需要用它来构造重要性权重 —— 这是 TDM 特有的、DMD2 中不存在的一项。
        """
        a1, s1 = extract_into_tensor(self.alphas, t1, noise.shape), extract_into_tensor(self.sigmas, t1, noise.shape)
        a2, s2 = extract_into_tensor(self.alphas, t2, noise.shape), extract_into_tensor(self.sigmas, t2, noise.shape)
        beta = (s2 ** 2 - (a2 / a1 * s1) ** 2) ** 0.5
        return (model_eps / a1 * a2 * s1 + beta * noise) / s2


# ----------------------------------------------------------------------------- 3. TDM 的核心：采样一段轨迹
def sample_segment(unet, predictor, traj, cond, mask, bsz, K, total_steps, device, use_separate=True):
    """
    ============================ 这里就是 TDM 与 DMD2 的分水岭 ============================
    TDM:  随机取第 k 段 -> 生成器从 x_{t_k} 一步预测 x0 -> 按 ODE 回到**段终点 t_mid**
          -> 从 t_mid 加噪到段内随机 t。匹配的是"一步转移后的分布"。
    DMD2: 生成器从 x_{t_k} 预测 x0 后**直接当作最终干净样本**，
          x_t = scheduler.add_noise(x0, torch.randn_like(x0), t)，t ~ U[t_min, t_max]。
          匹配的是"终点分布"。
    差别的后果: TDM 的 x_t 里保留了生成器自己的 eps，噪声非标准高斯 -> 需要 is_weight；
              但换来了对每一段转移的显式监督，收敛快得多。
    ===================================================================================
    K=4, total_steps=900 时: k=1..4 -> t_k = 224/449/674/899, t_mid = 0/225/450/675
    """
    k = torch.randint(1, K + 1, (bsz,), device=device).long()
    x_tk = torch.stack([traj[k[i]][i] for i in range(bsz)])   # traj 已 reverse，索引 k <-> t_k
    t_k = k * total_steps // K - 1                            # 段起点: 224 / 449 / 674 / 899
    t_mid = t_k - total_steps // K + 1                        # 段终点:   0 / 225 / 450 / 675

    # 匹配时刻 t 的采样区间 —— README 提到的两种模式:
    #   use_separate=True : t ~ [t_mid, t_k-10]，各段完全隔离 (推荐，无需给 fake score 加 step 条件)
    #   use_separate=False: t ~ [t_mid, T-10] ，各段共享上界 (建议给 fake score 加 step 条件)
    # 下界 max(20, ·) 避开 t<20 的极高 SNR 区 (t=20 时 SNR 已达 158.6，再小数值不稳)
    upper = (t_k - 10) if use_separate else (total_steps - 10)
    t = torch.stack([torch.randint(max(20, t_mid[i]), upper[i] if use_separate else upper,
                                   (1,), device=device)[0] for i in range(bsz)]).long()

    eps_g, x0_g = predictor.predict(unet, x_tk, t_k, cond, mask)    # 生成器一步 -> x0 预测
    x_mid = predictor.sch.add_noise(x0_g, eps_g, t_mid)             # 确定性 ODE 转移到段终点
    fresh = torch.randn_like(x_mid)
    x_t = predictor.add_noise(x_mid, fresh, t_mid, t)               # 再扩散到 t 处做 score 匹配
    return x0_g, eps_g, x_t, t, t_mid, fresh


# ----------------------------------------------------------------------------- 4. 训练主循环
def train_step(batch, unet, unet_fake, unet_sd, predictor, scheduler,
               optimizer_g, optimizer_d, cond, mask, uncond_emb, uncond_mask,
               cfg=4.5, K=4, total_steps=900, device="cuda"):
    """
    vs DMD2 的训练结构差异:
      - DMD2 是 two time-scale: 每更新生成器 1 次，先更新 fake score 5 次 (dfake_gen_update_ratio=5)，
        且额外训练一个 GAN 判别器头 (挂在 fake score 的 bottleneck 上)，需要真实图像 batch。
      - TDM 这里是严格 1:1 交替，没有 GAN loss、没有真实图像，batch 里只有 prompt。
    """
    bsz = cond.shape[0]
    noise = torch.randn(bsz, 4, 64, 64, device=device)

    # (0) 用当前 G 自采样轨迹（无梯度）；reverse 后索引 k 对应时刻 t_k
    with torch.no_grad():
        traj = generate_trajectory(unet, scheduler, noise, cond, mask, K, total_steps,
                                   predictor.alphas, predictor.sigmas)
        traj.reverse()          # [x_0, x_224, x_449, x_674, x_899]

    # ---------------- (A) 更新 fake score：让它成为当前生成器分布的 score ----------------
    with torch.no_grad():
        x0_g, eps_g, x_t, t, t_mid, fresh = sample_segment(
            unet, predictor, traj, cond, mask, bsz, K, total_steps, device)
        # TDM 独有的重要性权重:
        #   x_t 的等效噪声是 mixed (模型噪声+新噪声)，并非标准高斯；
        #   用密度比 N(mixed)/N(fresh) 修正这一偏移，否则 fake score 拟合的是错的分布。
        #   DMD2 的 x_t 噪声是纯 randn，密度比恒为 1，因此完全没有这一项。
        mixed = predictor.mixed_noise(eps_g, fresh, t_mid, t)
        is_w = torch.exp(-0.5 * (mixed ** 2).view(bsz, -1).mean(1)) / \
               torch.exp(-0.5 * (fresh ** 2).view(bsz, -1).mean(1))

    _, x0_fake = predictor.predict(unet_fake, x_t, t, cond, mask)
    snr = compute_snr(scheduler, t).clamp(max=5)          # min-SNR-5 截断，DMD2 用普通去噪 MSE
    loss_fake = (F.mse_loss(x0_fake.float(), x0_g.float(), reduction="none").mean([1, 2, 3])
                 * snr * is_w).mean()
    loss_fake.backward()
    optimizer_d.step(); optimizer_d.zero_grad()

    # ---------------- (B) 更新生成器：分布匹配梯度 ----------------
    # 注意: 原实现在 (A)(B) 各自独立重采样一次 segment，不共用同一批 x_t
    x0_g, eps_g, x_t, t, t_mid, _ = sample_segment(
        unet, predictor, traj, cond, mask, bsz, K, total_steps, device)

    with torch.no_grad():
        _, x0_real = predictor.predict(unet_sd, x_t, t, cond, mask)              # 教师，有条件
        _, x0_uncond = predictor.predict(unet_sd, x_t, t, uncond_emb, uncond_mask)  # 教师，无条件
        _, x0_fake = predictor.predict(unet_fake, x_t, t, cond, mask)            # fake score
        # DMD 梯度 ∇KL ≈ s_fake - s_real，写成 "目标样本" 形式便于用 MSE 反传。
        # 后一项是 CFG: x0_real + (cfg-1)(x0_real - x0_uncond) 恰好等于用 cfg 引导的教师 x0，
        # 所以这里与 DMD2 "用带 CFG 的教师算 pred_real" 在数学上等价，只是写法拆开了。
        target = x0_g.detach() + (x0_real - x0_fake) + (cfg - 1) * (x0_real - x0_uncond)

    # 自适应归一化 (与 DMD/DMD2 相同): 按 |x0_g - x0_real_cfg| 的量级缩放，
    # 使不同 t (SNR 跨 4 个数量级) 上的梯度尺度一致
    x0_real_cfg = x0_real + (cfg - 1) * (x0_real - x0_uncond)
    w = torch.abs(x0_g.double() - x0_real_cfg.double()).mean([1, 2, 3], keepdim=True).detach()

    use_huber, huber_c = True, 1e-3          # --use_huber，对离群梯度更稳
    if use_huber:
        loss_g = ((torch.sqrt((x0_g.float() - target.float()) ** 2 + huber_c ** 2) - huber_c) / w).mean()
    else:
        loss_g = (F.mse_loss(x0_g.float(), target.float(), reduction="none") / w).mean()

    loss_g.backward()
    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
    optimizer_g.step(); optimizer_g.zero_grad()
    return loss_fake, loss_g
