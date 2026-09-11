"""使用因果 Transformer 自回归生成离散视频 token 的最小示例。

依赖:
    pip install torch

运行:
    python autoregressive_video_minimal.py

核心流程:
    1. 生成简单的移动方块视频，形状为 [T, H, W]。
    2. 将像素量化为有限个离散 token。
    3. 按时间、行、列顺序将视频展平为 token 序列。
    4. 使用 teacher forcing 训练因果 Transformer 预测下一个 token。
    5. 从 BOS 开始，每次采样一个 token，自回归生成完整视频。

为保持最小实现，代码直接建模低分辨率离散像素。真实视频生成模型通常先用
VQ-VAE/VQGAN 将视频压缩为 latent token，再用同样的自回归方法生成 latent
token，最后通过 decoder 还原成 RGB 视频。
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_FRAMES = 6
HEIGHT = 8
WIDTH = 8
NUM_LEVELS = 4
VIDEO_TOKEN_COUNT = NUM_FRAMES * HEIGHT * WIDTH

# 像素 token 使用 [0, NUM_LEVELS - 1]，BOS 是额外的序列起始 token。
BOS_ID = NUM_LEVELS
INPUT_VOCAB_SIZE = NUM_LEVELS + 1

BATCH_SIZE = 32
TRAIN_STEPS = 300
LEARNING_RATE = 3e-4

MODEL_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.0


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
# 合成视频数据
# -----------------------------------------------------------------------------
def make_moving_square_video():
    """生成一个方块匀速移动并在边界反弹的离散视频。

    返回:
        video: LongTensor [T, H, W]

    视频已是离散 token，不需要额外 tokenizer。背景为 0，方块主体为 3，
    边缘为 2。位置和速度随机，使模型学习跨帧运动规律，而不是记住单个视频。
    """
    square_size = 2
    x = random.randint(0, WIDTH - square_size)
    y = random.randint(0, HEIGHT - square_size)
    dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])

    frames = []
    for _ in range(NUM_FRAMES):
        frame = torch.zeros(HEIGHT, WIDTH, dtype=torch.long)
        frame[y : y + square_size, x : x + square_size] = NUM_LEVELS - 1

        # 给左上角一个较低灰度 token，使方块方向不完全对称。
        frame[y, x] = NUM_LEVELS - 2
        frames.append(frame)

        next_x = x + dx
        next_y = y + dy
        if next_x < 0 or next_x + square_size > WIDTH:
            dx = -dx
            next_x = x + dx
        if next_y < 0 or next_y + square_size > HEIGHT:
            dy = -dy
            next_y = y + dy
        x, y = next_x, next_y

    return torch.stack(frames, dim=0)


def make_batch(batch_size):
    """构造视频 batch，并按 [time, row, column] 展平为序列。"""
    videos = torch.stack([make_moving_square_video() for _ in range(batch_size)])
    return videos.view(batch_size, VIDEO_TOKEN_COUNT).to(DEVICE)


# -----------------------------------------------------------------------------
# 因果视频 Transformer
# -----------------------------------------------------------------------------
class AutoregressiveVideoTransformer(nn.Module):
    """把整个离散视频当作一条 token 序列建模。

    序列顺序为:
        frame 0 的所有像素 -> frame 1 的所有像素 -> ... -> frame T-1

    因此后面帧的 token 可以关注所有已经生成的前面帧，也可以关注当前帧中
    raster scan 顺序下已经生成的像素，但不能看到未来 token。
    """

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(INPUT_VOCAB_SIZE, MODEL_DIM)
        self.position_embedding = nn.Embedding(VIDEO_TOKEN_COUNT, MODEL_DIM)

        layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=4 * MODEL_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)
        self.final_norm = nn.LayerNorm(MODEL_DIM)

        # 输出只预测合法像素 token，BOS 仅允许出现在输入中。
        self.lm_head = nn.Linear(MODEL_DIM, NUM_LEVELS, bias=False)

    def forward(self, input_ids):
        """预测每个输入位置之后的一个视频 token。

        参数:
            input_ids: LongTensor [B, L]，L <= VIDEO_TOKEN_COUNT

        返回:
            logits: FloatTensor [B, L, NUM_LEVELS]
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > VIDEO_TOKEN_COUNT:
            raise ValueError("输入序列超过最大视频 token 数")

        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.token_embedding(input_ids)
        hidden = hidden + self.position_embedding(positions)[None, :, :]

        # mask[i, j] = True 表示位置 i 不能读取未来位置 j。
        # 对角线保留，使当前位置可以读取自己的输入 token。训练输入已经右移，
        # 所以当前位置仍然看不到对应的 target token。
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask)
        hidden = self.final_norm(hidden)
        return self.lm_head(hidden)


# -----------------------------------------------------------------------------
# Teacher forcing 训练
# -----------------------------------------------------------------------------
def make_teacher_forcing_inputs(video_tokens):
    """将目标视频右移一位，在序列开头插入 BOS。

    假设视频 token 是:
        target = [x0, x1, x2, ..., xN]

    模型输入为:
        input  = [BOS, x0, x1, ..., xN-1]

    在位置 t，模型只能看到 BOS 和 x0...x(t-1)，并预测 xt。这就是最大似然
    自回归训练，也叫 teacher forcing，因为历史 token 使用真实数据而非模型
    自己的预测。
    """
    bos = torch.full(
        (video_tokens.size(0), 1),
        BOS_ID,
        dtype=torch.long,
        device=video_tokens.device,
    )
    return torch.cat([bos, video_tokens[:, :-1]], dim=1)


def train(model):
    """最小化所有时空 token 的 next-token cross entropy。

    自回归分解为:
        p(video) = product_t p(x_t | x_0, ..., x_{t-1})

    对应负对数似然:
        loss = -mean_t log p_theta(x_t | x_<t)

    `cross_entropy` 内部会对 logits 做 log-softmax，并取真实 token 的负对数
    概率。展平 batch 和 sequence 后，每个时空位置都是一个分类目标。
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    for step in range(1, TRAIN_STEPS + 1):
        targets = make_batch(BATCH_SIZE)
        inputs = make_teacher_forcing_inputs(targets)
        logits = model(inputs)

        loss = F.cross_entropy(
            logits.reshape(-1, NUM_LEVELS),
            targets.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % 50 == 0:
            print(f"step={step:03d} next_token_loss={loss.item():.4f}")


# -----------------------------------------------------------------------------
# 自回归视频采样
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_video(model, temperature=0.8):
    """从 BOS 开始逐 token 生成完整视频。

    每一轮只取模型最后一个位置的 logits，因为它表示：
        p(next_video_token | all_generated_tokens)

    采样出的 token 追加回上下文，再预测下一个 token。这里为了清晰，每一步都
    重算整个上下文；实际大模型会缓存每层 attention 的 KV，避免重复计算。
    """
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    model.eval()
    sequence = torch.tensor([[BOS_ID]], dtype=torch.long, device=DEVICE)

    for _ in range(VIDEO_TOKEN_COUNT):
        next_token_logits = model(sequence)[:, -1, :] / temperature
        probabilities = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        sequence = torch.cat([sequence, next_token], dim=1)

    video_tokens = sequence[:, 1:]
    video = video_tokens.view(NUM_FRAMES, HEIGHT, WIDTH)
    return video.cpu()


def tokens_to_uint8(video_tokens):
    """将 [0, NUM_LEVELS-1] 的离散 token 映射到可视化灰度值 [0, 255]。"""
    scale = 255.0 / (NUM_LEVELS - 1)
    return (video_tokens.float() * scale).round().to(torch.uint8)


def print_video(video_tokens):
    """用终端字符显示每一帧，便于不安装视频库时观察结果。"""
    shades = " .o#"
    for frame_index, frame in enumerate(video_tokens):
        print(f"\nframe {frame_index}")
        for row in frame:
            print("".join(shades[int(value)] for value in row))


def main():
    seed_everything(SEED)
    model = AutoregressiveVideoTransformer().to(DEVICE)

    train(model)
    generated_tokens = generate_video(model, temperature=0.8)
    generated_uint8 = tokens_to_uint8(generated_tokens)

    print_video(generated_tokens)

    # 保存 [T, H, W] uint8 张量。真实项目可进一步用 torchvision.io.write_video
    # 或 imageio 编码为 mp4/gif；这里保持只依赖 PyTorch。
    torch.save(
        {
            "token_video": generated_tokens,
            "uint8_video": generated_uint8,
            "shape": (NUM_FRAMES, HEIGHT, WIDTH),
        },
        "generated_video.pt",
    )


if __name__ == "__main__":
    main()
