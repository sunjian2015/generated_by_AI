import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, UNet2DConditionModel
from typing import Optional

class TDMTrainer:
    """
    Trajectory Distribution Matching (TDM) 训练器

    核心思路：
    1. 用 K 步学生生成器 f_θ 从噪声生成轨迹 {x_{t_0}, x_{t_1}, ..., x_{t_{K-1}}}
    2. 在每个中间时刻 t_i，对 x_{t_i} 加噪到 τ ∈ [t_i, t_{i+1}]，得到 x_τ
    3. 训练 fake score s_φ 拟合 x_τ 的分布（通过去噪目标）
    4. 训练生成器 f_θ 最小化 ||x_{t_i} - x̃_{t_i}||_Huber，其中 x̃_{t_i} 由真假 score 差值修正
    """

    def __init__(
        self,
        unet_student: UNet2DConditionModel,  # 学生生成器 f_θ
        unet_fake_score: UNet2DConditionModel,  # fake score s_φ
        unet_teacher: UNet2DConditionModel,  # 教师模型（可选，用于获取真实 score）
        noise_scheduler: DDIMScheduler,
        K: int = 4,  # 采样步数
        T: int = 1000,  # 总扩散步数
        huber_c: float = 0.00054,  # Pseudo-Huber 参数，论文建议 c = 0.00054 * sqrt(d)
        use_huber: bool = True,
    ):
        self.unet_student = unet_student
        self.unet_fake_score = unet_fake_score
        self.unet_teacher = unet_teacher
        self.scheduler = noise_scheduler
        self.K = K
        self.T = T
        self.huber_c = huber_c
        self.use_huber = use_huber

        # 计算 K 步对应的时刻：t_i = T * i / K
        self.timesteps = torch.linspace(T - 1, 0, K + 1).long()  # [t_0=999, t_1=750, t_2=500, t_3=250, t_4=0]

    def sample_student_trajectory(
        self,
        noise: torch.Tensor,  # shape: (B, C, H, W)，初始噪声
        prompt_embeds: torch.Tensor,  # shape: (B, seq_len, dim)
        cfg_scale: float = 1.0,
    ) -> list:
        """
        用学生模型生成 K 步 ODE 轨迹（确定性采样，如 DDIM）

        返回：[(x_{t_0}, t_0), (x_{t_1}, t_1), ..., (x_{t_{K-1}}, t_{K-1})]
        """
        trajectory = []
        x_t = noise

        for i in range(self.K):
            t = self.timesteps[i]
            t_next = self.timesteps[i + 1]

            # 预测噪声 ε_θ(x_t, t)（论文 Eq. 1 相关）
            with torch.no_grad():
                noise_pred = self.unet_student(x_t, t, encoder_hidden_states=prompt_embeds).sample

                # 如果使用 CFG（classifier-free guidance）
                if cfg_scale != 1.0:
                    noise_pred_uncond = self.unet_student(x_t, t, encoder_hidden_states=torch.zeros_like(prompt_embeds)).sample
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)

            # 保存当前轨迹点（假设：轨迹存储的是去噪后的 x_{t_i}）
            # 论文中 x_{t_i} 是中间噪声样本，需根据 scheduler 推进
            trajectory.append((x_t.clone(), t))

            # DDIM 单步更新：x_{t_next} = sqrt(α_{t_next}) * x̂_0 + sqrt(1-α_{t_next}) * ε
            alpha_t = self.scheduler.alphas_cumprod[t]
            alpha_t_next = self.scheduler.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0)

            # 预测 x_0
            x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            # 推进到 t_next
            x_t = torch.sqrt(alpha_t_next) * x_0_pred + torch.sqrt(1 - alpha_t_next) * noise_pred

        return trajectory

    def compute_snr(self, tau):
        """SNR(t) = ᾱ_t / (1 - ᾱ_t)，timesteps: (B,) long → 返回 (B,)"""
        alphas_cumprod = self.scheduler.alphas_cumprod.to(tau.device)
        a = alphas_cumprod[tau].float()
        return a / (1.0 - a)

    def train_fake_score(
        self,
        trajectory: list,
        prompt_embeds: torch.Tensor,
        use_importance_sampling: bool = True,
    ) -> torch.Tensor:
        """
        训练 fake score s_φ（论文 Eq. 7 和 Algorithm 1 第 8-10 行）

        目标：min E_{x_{t_i} ~ p_{θ,t_i}} E_{x_τ ~ q(x_τ|x_{t_i})} ||s_φ(x_τ, τ) - x̂_{t_i}||²

        重要性采样：从 q(x_τ | x_{t_i}) 采样，而非 q(x_τ | x̂_{t_i})
        """
        loss_score = 0.0

        for i in range(len(trajectory) - 1):  # K-1 个区间 [t_i, t_{i+1}]
            x_ti, t_i = trajectory[i]
            t_i_next = self.timesteps[i + 1]

            # 在区间 [t_i, t_{i+1}] 内均匀采样时刻 τ
            # 假设：use_separate=True 时区间不重叠（论文 3.1 节强调）
            tau = torch.randint(t_i_next.item(), t_i.item(), (x_ti.shape[0],), device=x_ti.device)

            # 重要性采样：从 q(x_τ | x_{t_i}) 加噪
            # 假设 x_{t_i} 是噪声样本，x̂_{t_i} 是对应的干净样本（需从 x_{t_i} 预测）
            with torch.no_grad():
                noise_pred_ti = self.unet_student(x_ti, t_i, encoder_hidden_states=prompt_embeds).sample
                x_0_pred = (x_ti - torch.sqrt(1 - self.scheduler.alphas_cumprod[t_i]) * noise_pred_ti) / \
                            torch.sqrt(self.scheduler.alphas_cumprod[t_i])

            # 从 x_{t_i} 加噪到 x_τ：x_τ = sqrt(α_τ/α_{t_i}) * x_{t_i} + sqrt(1 - α_τ/α_{t_i}) * ε
            # 简化：直接用 scheduler.add_noise
            noise_epsilon = torch.randn_like(x_ti)
            alpha_ti = self.scheduler.alphas_cumprod[t_i]
            alpha_tau = self.scheduler.alphas_cumprod[tau]

            # 等效总噪声 ε_mixed：使 x_τ = sqrt(α_τ)·x̂_{t_i} + sqrt(1-α_τ)·ε_mixed
            eps_mixed = (x_tau - torch.sqrt(alpha_tau) * x_0_pred) / torch.sqrt(1 - alpha_tau)

            # 重要性权重 w = q(x_τ|x̂_{t_i}) / q(x_τ|x_{t_i})，两个高斯密度之比
            bsz = x_ti.shape[0]
            is_weight = torch.exp(-0.5 * (eps_mixed ** 2).view(bsz, -1).mean(dim=1)) / \
                        torch.exp(-0.5 * (noise_epsilon ** 2).view(bsz, -1).mean(dim=1))


            if use_importance_sampling:
                # 重要性采样：q(x_τ | x_{t_i}) = q(x_τ | x̂_{t_i}) * q(x̂_{t_i} | x_{t_i}) / q(x_{t_i})
                # 实际操作：从 x_{t_i} 扩散到 x_τ（论文强调从轨迹样本扩散）
                ratio = alpha_tau / alpha_ti
                x_tau = torch.sqrt(ratio) * x_ti + torch.sqrt(1 - ratio) * noise_epsilon
            else:
                # 不使用重要性采样：从 x̂_{t_i}（干净样本）加噪
                x_tau = self.scheduler.add_noise(x_0_pred, noise_epsilon, tau)
                is_weight = 1

            # Fake score 预测目标：去噪到 x̂_{t_i}
            fake_score_pred = self.unet_fake_score(x_tau, tau, encoder_hidden_states=prompt_embeds).sample

            # 去噪损失（假设预测噪声）
            # 论文 Eq. 7：||f_φ(x_τ, τ) - x̂_{t_i}||²
            # 实际代码通常预测噪声，需转换：x̂_{t_i} = (x_τ - sqrt(1-α_τ) * ε_pred) / sqrt(α_τ)
            x_0_pred_fake = (x_tau - torch.sqrt(1 - alpha_tau) * fake_score_pred) / torch.sqrt(alpha_tau)

            # loss_score += F.mse_loss(x_0_pred_fake, x_0_pred.detach())

            loss = ((x_0_pred_fake - x_0_pred.detach()) ** 2).view(bsz, -1).mean(dim=1)
            snr = compute_snr(self.scheduler, tau)
            snr = torch.stack([snr, 5 * torch.ones_like(tau)], dim=1).min(dim=1)[0]
            loss_score += (loss * snr * is_weight.detach()).mean()

        return loss_score / (len(trajectory) - 1)

    def train_generator(
        self,
        trajectory: list,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        训练生成器 f_θ（论文 Eq. 11-12 和 Algorithm 1 第 11-13 行）

        目标：min Σ ||x_{t_i} - x̃_{t_i}||_Huber
        其中 x̃_{t_i} = x_{t_i} + λ_τ [s_real(x_τ, τ) - s_fake(x_τ, τ)]
        """
        loss_gen = 0.0

        for i in range(len(trajectory) - 1):
            x_ti, t_i = trajectory[i]
            t_i_next = self.timesteps[i + 1]

            # 采样 τ ∈ [t_i, t_{i+1}]
            tau = torch.randint(t_i_next.item(), t_i.item(), (x_ti.shape[0],), device=x_ti.device)

            # 从 x_{t_i} 加噪到 x_τ
            noise_epsilon = torch.randn_like(x_ti)
            alpha_ti = self.scheduler.alphas_cumprod[t_i]
            alpha_tau = self.scheduler.alphas_cumprod[tau]
            x_tau = torch.sqrt(alpha_tau / alpha_ti) * x_ti + \
                    torch.sqrt(alpha_tau * (1 - alpha_ti / alpha_ti)) * noise_epsilon

            # 计算真实 score（教师模型）和 fake score
            with torch.no_grad():
                real_score = self.unet_teacher(x_tau, tau, encoder_hidden_states=prompt_embeds).sample
            fake_score = self.unet_fake_score(x_tau, tau, encoder_hidden_states=prompt_embeds).sample

            # 修正样本：x̃_{t_i} = x_{t_i} + λ_τ [s_real - s_fake]
            # 假设：λ_τ = σ_τ²（扩散系数），论文 Eq. 6 隐含
            lambda_tau = (1 - alpha_tau) / torch.sqrt(alpha_tau)
            score_diff = real_score - fake_score
            x_ti_revised = x_ti + lambda_tau * score_diff

            # Pseudo-Huber 损失（论文 Eq. 11）
            if self.use_huber:
                diff = x_ti - x_ti_revised.detach()
                loss_step = (torch.sqrt(diff ** 2 + self.huber_c ** 2) - self.huber_c).mean()
            else:
                loss_step = F.mse_loss(x_ti, x_ti_revised.detach())

            loss_gen += loss_step

        return loss_gen / (len(trajectory) - 1)


# ========== 训练主循环示例 ==========
def training_loop():
    """
    训练 TDM 的完整流程（简化版 Algorithm 1）
    """
    # 初始化模型（假设已加载预训练权重）
    unet_student = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    unet_fake_score = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    unet_teacher = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    noise_scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    trainer = TDMTrainer(unet_student, unet_fake_score, unet_teacher, noise_scheduler, K=4)

    optimizer_student = torch.optim.AdamW(unet_student.parameters(), lr=2e-6)
    optimizer_fake = torch.optim.AdamW(unet_fake_score.parameters(), lr=2e-5)

    for step in range(20000):  # 论文：SD-v1.5 训练 20k 步
        # 1. 采样 prompt 和噪声（data-free，只需 prompt）
        # 假设：从 JourneyDB 数据集采样 prompt，用 text encoder 编码
        prompt_embeds = torch.randn(4, 77, 768).cuda()  # 示例：batch=4
        noise = torch.randn(4, 4, 64, 64).cuda()  # latent space

        # 2. 生成学生轨迹（K 步确定性采样）
        trajectory = trainer.sample_student_trajectory(noise, prompt_embeds, cfg_scale=3.5)

        # 3. 训练 fake score（论文 Eq. 7）
        optimizer_fake.zero_grad()
        loss_score = trainer.train_fake_score(trajectory, prompt_embeds, use_importance_sampling=True)
        loss_score.backward()
        torch.nn.utils.clip_grad_norm_(unet_fake_score.parameters(), max_norm=1.0)
        optimizer_fake.step()

        # 4. 训练生成器（论文 Eq. 11）
        optimizer_student.zero_grad()
        loss_gen = trainer.train_generator(trajectory, prompt_embeds)
        loss_gen.backward()
        torch.nn.utils.clip_grad_norm_(unet_student.parameters(), max_norm=1.0)
        optimizer_student.step()

        if step % 100 == 0:
            print(f"Step {step}: loss_score={loss_score.item():.4f}, loss_gen={loss_gen.item():.4f}")


# ========== 推理示例：4 步生成 ==========
@torch.no_grad()
def inference_4step(unet_student, prompt_embeds, noise_scheduler):
    """
    用训练好的 4 步生成器推理
    输入：prompt_embeds (B, seq_len, dim), 初始噪声 ε ~ N(0, I)
    输出：生成的图像 latent
    """
    x_t = torch.randn(1, 4, 64, 64).cuda()  # 初始噪声
    timesteps = torch.linspace(999, 0, 5).long()  # [999, 750, 500, 250, 0]

    for i in range(4):
        t = timesteps[i]
        t_next = timesteps[i + 1]

        # 预测噪声
        noise_pred = unet_student(x_t, t, encoder_hidden_states=prompt_embeds).sample

        # DDIM 更新
        alpha_t = noise_scheduler.alphas_cumprod[t]
        alpha_t_next = noise_scheduler.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0).cuda()
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x_t = torch.sqrt(alpha_t_next) * x_0_pred + torch.sqrt(1 - alpha_t_next) * noise_pred

    return x_t  # 返回生成的 latent，需用 VAE decoder 解码为图像

# 关键注释说明：

# 1. 形状标注：
# - noise: (B, C, H, W) = (batch, 4, 64, 64)，SD-v1.5 latent space
# - prompt_embeds: (B, seq_len, dim) = (batch, 77, 768)
# - trajectory: 列表长度 K，每个元素 (x_{t_i}, t_i)
# 2. 假设与简化：
# - 假设 1：使用 DDIM 作为 ODE solver（论文也支持 DPMSolver）
# - 假设 2：use_separate=True 时区间不重叠（论文 3.1 强调）
# - 假设 3：真实 score 来自教师模型的预测噪声，转换为 score via Eq. (1)
# - 简化：未实现采样步数感知（Eq. 8），完整版需将 K 作为条件注入 UNet
# 3. 论文细节对应：
# - train_fake_score → Algorithm 1 第 234 行 + Eq. (7)
# - train_generator → Algorithm 1 第 12 行 + Eq. (11)
# - Pseudo-Huber 损失 → Eq. (11)，c = 0.00054 * sqrt(d)
# - 重要性采样 → 从 x_{t_i} 而非 x̂_{t_i} 扩散（消融实验 +2.08 HPS）