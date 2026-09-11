"""只依赖 PyTorch 和 Transformers 的最小 GPT + Reward Model + PPO 示例。

运行:
    conda activate torch
    python ppo_gpt_minimal.py

为保证离线可运行，所有模型都由一个很小的 GPT-2 配置随机初始化，不下载权重。
这份代码关注 RLHF/PPO 的数据流和 loss，而不是生成自然语言的实际质量。
"""

import copy
import random

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import (
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2ForTokenClassification,
    GPT2LMHeadModel,
)


# -----------------------------------------------------------------------------
# 1. 配置
# -----------------------------------------------------------------------------
SEED = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
RESPONSE_LENGTH = 6
PPO_UPDATES = 30
PPO_EPOCHS = 4

POLICY_LR = 3e-4
VALUE_LR = 5e-4
REWARD_LR = 1e-3
REWARD_TRAIN_STEPS = 80

GAMMA = 1.0
GAE_LAMBDA = 0.95
POLICY_CLIP = 0.2
VALUE_CLIP = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
KL_COEF = 0.05
MAX_GRAD_NORM = 1.0
SAMPLE_TEMPERATURE = 1.0


# 一个离散的单词级玩具词表。GPT 只处理 token id，并不要求一定使用 BPE。
TOKENS = [
    "<pad>",
    "<bos>",
    "write",
    "a",
    "message",
    "good",
    "great",
    "happy",
    "kind",
    "bad",
    "sad",
    "mean",
    "okay",
    "today",
    "friend",
    ".",
]
TOKEN_TO_ID = {token: i for i, token in enumerate(TOKENS)}
PAD_ID = TOKEN_TO_ID["<pad>"]
BOS_ID = TOKEN_TO_ID["<bos>"]
PROMPT_TOKENS = ["<bos>", "write", "a", "message"]
PROMPT_IDS = torch.tensor(
    [TOKEN_TO_ID[token] for token in PROMPT_TOKENS], dtype=torch.long
)
POSITIVE_IDS = torch.tensor(
    [TOKEN_TO_ID[token] for token in ("good", "great", "happy", "kind")]
)
NEGATIVE_IDS = torch.tensor([TOKEN_TO_ID[token] for token in ("bad", "sad", "mean")])
GENERATABLE_IDS = torch.tensor(
    [i for i, token in enumerate(TOKENS) if token not in {"<pad>", "<bos>"}]
)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def decode(token_ids):
    """仅用于日志展示。"""
    return " ".join(TOKENS[int(token_id)] for token_id in token_ids)


def make_gpt_config():
    """创建约 10 万参数的小 GPT-2 配置，便于 CPU 上快速演示。"""
    return GPT2Config(
        vocab_size=len(TOKENS),
        n_positions=len(PROMPT_TOKENS) + RESPONSE_LENGTH,
        n_embd=48,
        n_layer=2,
        n_head=4,
        # PPO 要精确比较同一 token 在新旧策略下的概率。关闭 dropout，避免
        # 重要性采样 ratio 混入与参数更新无关的随机网络 mask。
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        summary_first_dropout=0.0,
        bos_token_id=BOS_ID,
        eos_token_id=TOKEN_TO_ID["."],
        pad_token_id=PAD_ID,
        use_cache=False,
    )


# -----------------------------------------------------------------------------
# 2. 四个模型
# -----------------------------------------------------------------------------
def build_models():
    """构造 PPO 文本训练常见的四个模型角色。

    policy:
        GPT2LMHeadModel，actor。输出每个位置上下一个 token 的 logits。

    reference:
        policy 初始权重的冻结副本。它不训练，只约束 policy 不要偏离初始模型。

    reward_model:
        GPT2ForSequenceClassification，读取 prompt + 完整 response，输出一个标量
        R(x, y)。在 PPO 前先通过监督回归训练，PPO 阶段完全冻结。

    value_model:
        GPT2ForTokenClassification，num_labels=1。每个 token 位置输出一个标量
        V(s_t)，作为独立 critic，不与 policy 共享参数。
    """
    policy = GPT2LMHeadModel(make_gpt_config()).to(DEVICE)
    reference = copy.deepcopy(policy).to(DEVICE)

    reward_config = make_gpt_config()
    reward_config.num_labels = 1
    reward_model = GPT2ForSequenceClassification(reward_config).to(DEVICE)

    value_config = make_gpt_config()
    value_config.num_labels = 1
    value_model = GPT2ForTokenClassification(value_config).to(DEVICE)

    reference.requires_grad_(False)
    reference.eval()
    return policy, reference, reward_model, value_model


# -----------------------------------------------------------------------------
# 3. 奖励模型的监督训练
# -----------------------------------------------------------------------------
def synthetic_preference_score(response_ids):
    """只用于生成奖励模型的监督标签，不会被 PPO 直接调用。

    标签 = 正向词数量 - 负向词数量，并额外惩罚连续重复 token。
    在真实项目中，这批 (prompt, response, score) 数据应来自人工偏好数据：

      - point-wise reward model: 用 MSE 拟合人工标量分数；
      - pair-wise reward model: 用 -log sigmoid(r_chosen-r_rejected) 做排序训练。

    此处采用 point-wise MSE，代码更短，也能得到一个真正训练过的神经奖励模型。
    """
    positive = (response_ids[..., None] == POSITIVE_IDS).any(dim=-1).sum(dim=1)
    negative = (response_ids[..., None] == NEGATIVE_IDS).any(dim=-1).sum(dim=1)
    repeated = (response_ids[:, 1:] == response_ids[:, :-1]).sum(dim=1)
    return positive.float() - negative.float() - 0.25 * repeated.float()


def make_reward_batch(batch_size):
    """构造固定长度的 prompt/response 监督样本。"""
    sampled_indices = torch.randint(
        0, len(GENERATABLE_IDS), (batch_size, RESPONSE_LENGTH)
    )
    response_ids = GENERATABLE_IDS[sampled_indices]
    labels = synthetic_preference_score(response_ids)
    prompt_ids = PROMPT_IDS.unsqueeze(0).repeat(batch_size, 1)
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    return full_ids.to(DEVICE), labels.to(DEVICE)


def train_reward_model(reward_model):
    """用标量回归训练奖励模型。

    Reward loss:
        L_RM = mean((R_phi(prompt, response) - human_score)^2)

    奖励模型在 PPO 前训练；训练完成后冻结。PPO 不应该通过 reward score
    反向传播到奖励模型，否则 policy 和 reward model 可能串通改变评分标准。
    """
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=REWARD_LR)
    reward_model.train()

    for step in range(1, REWARD_TRAIN_STEPS + 1):
        input_ids, labels = make_reward_batch(batch_size=64)
        predicted_scores = reward_model(input_ids=input_ids).logits.squeeze(-1)
        reward_loss = F.mse_loss(predicted_scores, labels)

        optimizer.zero_grad()
        reward_loss.backward()
        torch.nn.utils.clip_grad_norm_(reward_model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        if step in {1, REWARD_TRAIN_STEPS}:
            print(f"reward_model step={step:03d} mse={reward_loss.item():.4f}")

    reward_model.requires_grad_(False)
    reward_model.eval()


# -----------------------------------------------------------------------------
# 4. Rollout：用旧策略逐 token 采样
# -----------------------------------------------------------------------------
def mask_non_generation_logits(logits):
    """禁止生成控制 token；采样与 PPO 重算概率时必须使用同一个 mask。"""
    logits = logits.clone()
    logits[..., PAD_ID] = -torch.inf
    logits[..., BOS_ID] = -torch.inf
    return logits


def action_distribution(logits):
    """得到实际行为策略分布。

    这里从 temperature softmax 对应的完整 Categorical 分布采样，没有 top-k、
    top-p 等截断。若增加截断，PPO 更新阶段也必须应用完全相同的截断，否则
    old_logprob 和 new_logprob 不属于同一种行为分布，重要性采样比率会失真。
    """
    logits = mask_non_generation_logits(logits) / SAMPLE_TEMPERATURE
    return Categorical(logits=logits)


@torch.no_grad()
def sample_rollout(policy, reference, reward_model, value_model):
    """生成一批 response，并保存 PPO 所需的固定 rollout 数据。

    每个生成 token 是一个 action：
        state s_t = prompt + 已生成的 token
        action a_t = 下一个 token

    old_logprobs 是行为策略 pi_old 真正采样 a_t 时的 log probability。一次
    rollout 后会做多个 PPO epoch；期间 old_logprobs 必须保持不变。
    """
    prompt_ids = PROMPT_IDS.to(DEVICE).unsqueeze(0).repeat(BATCH_SIZE, 1)
    full_ids = prompt_ids

    actions = []
    old_logprobs = []
    ref_logprobs = []
    old_values = []

    policy.eval()
    value_model.eval()
    for _ in range(RESPONSE_LENGTH):
        old_logits = policy(input_ids=full_ids).logits[:, -1, :]
        ref_logits = reference(input_ids=full_ids).logits[:, -1, :]
        values = value_model(input_ids=full_ids).logits[:, -1, 0]

        old_dist = action_distribution(old_logits)
        ref_dist = action_distribution(ref_logits)
        action = old_dist.sample()

        actions.append(action)
        old_logprobs.append(old_dist.log_prob(action))
        ref_logprobs.append(ref_dist.log_prob(action))
        old_values.append(values)
        full_ids = torch.cat([full_ids, action[:, None]], dim=1)

    response_ids = torch.stack(actions, dim=1)

    # 奖励模型只在完整序列上给一个 sequence-level scalar score。
    scores = reward_model(input_ids=full_ids).logits.squeeze(-1)
    return {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "old_logprobs": torch.stack(old_logprobs, dim=1),
        "ref_logprobs": torch.stack(ref_logprobs, dim=1),
        "old_values": torch.stack(old_values, dim=1),
        "scores": scores,
    }


# -----------------------------------------------------------------------------
# 5. Reward shaping 与 GAE
# -----------------------------------------------------------------------------
def make_token_rewards(batch):
    """将序列奖励和逐 token KL penalty 合成 token reward。

    对行为策略实际采到的 token，KL 的 Monte Carlo 样本为：

        sampled_kl_t = log pi_old(a_t|s_t) - log pi_ref(a_t|s_t)

    它单个样本可能为负，但在 a_t ~ pi_old 下的期望正是：

        E[sampled_kl_t] = KL(pi_old(.|s_t) || pi_ref(.|s_t)) >= 0

    每个 token 都受到 KL 惩罚；奖励模型对完整文本的 score 只加在最后一步：

        r_t = -beta * sampled_kl_t
        r_T = r_T + R_phi(prompt, response)
    """
    sampled_kl = batch["old_logprobs"] - batch["ref_logprobs"]
    rewards = -KL_COEF * sampled_kl
    rewards[:, -1] += batch["scores"]
    return rewards, sampled_kl


def compute_gae(rewards, old_values):
    """计算每个 token 的 Generalized Advantage Estimation。

        delta_t = r_t + gamma * V_old(s_{t+1}) - V_old(s_t)
        A_t = delta_t + gamma * lambda * A_{t+1}
        return_t = A_t + V_old(s_t)

    response 固定长度，并在最后一个 token 后终止，因此 V(s_{T+1}) = 0。
    returns 是 value model 的回归目标，advantages 决定 token 概率应增大还是减小。
    """
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros(rewards.size(0), device=DEVICE)
    next_value = torch.zeros(rewards.size(0), device=DEVICE)

    for t in reversed(range(RESPONSE_LENGTH)):
        delta = rewards[:, t] + GAMMA * next_value - old_values[:, t]
        next_advantage = delta + GAMMA * GAE_LAMBDA * next_advantage
        advantages[:, t] = next_advantage
        next_value = old_values[:, t]

    returns = advantages + old_values
    return advantages, returns


# -----------------------------------------------------------------------------
# 6. 对固定 rollout 重新计算新策略概率和新 value
# -----------------------------------------------------------------------------
def evaluate_actions(policy, value_model, prompt_ids, response_ids):
    """一次并行 forward 对齐所有生成 token。

    完整序列是 [prompt, a1, a2, ..., aT]。因果 LM 在位置 i 的输出预测位置
    i+1，所以模型输入去掉最后一个 token 后：

        prompt 最后位置的 logits -> a1
        a1 位置的 logits          -> a2
        ...

    start = prompt_length - 1 是文本 PPO 中关键的 off-by-one 对齐位置。
    """
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    model_inputs = full_ids[:, :-1]

    all_logits = policy(input_ids=model_inputs).logits
    all_values = value_model(input_ids=model_inputs).logits[..., 0]
    start = prompt_ids.size(1) - 1
    end = start + response_ids.size(1)

    action_logits = all_logits[:, start:end, :]
    new_values = all_values[:, start:end]
    new_dist = action_distribution(action_logits)
    new_logprobs = new_dist.log_prob(response_ids)
    entropy = new_dist.entropy()
    return new_logprobs, entropy, new_values


# -----------------------------------------------------------------------------
# 7. PPO losses
# -----------------------------------------------------------------------------
def compute_ppo_losses(policy, value_model, batch, advantages, returns):
    """计算 policy、value、entropy 三类 PPO loss。

    Policy loss 与重要性采样：
        rollout 来自 pi_old，当前参数对应 pi_theta。对固定的旧样本，使用

          ratio_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
                  = exp(new_logprob_t - old_logprob_t)

        修正新旧策略的分布差异。ratio=1 表示概率未变化。PPO 再裁剪 ratio：

          L_clip = -mean(min(ratio*A, clip(ratio,1-eps,1+eps)*A))

        A>0 时希望提高该 token 概率，A<0 时希望降低概率；min 与 clip 共同
        限制一次更新能够获得的收益，避免复用同一 rollout 时策略变化过大。

    Value loss：
        critic 拟合 GAE return。value 也相对 rollout 时的 V_old 做裁剪，并取
        clipped/unclipped 两个平方误差的较大者，防止 critic 单步变化过大。

    Entropy bonus：
        最大化 token 分布熵以保留探索。由于优化器最小化 loss，因此使用负号。
    """
    new_logprobs, entropy, new_values = evaluate_actions(
        policy, value_model, batch["prompt_ids"], batch["response_ids"]
    )

    # 重要性采样比率。用 log probability 相减再 exp，比直接概率相除更稳定。
    log_ratio = new_logprobs - batch["old_logprobs"]
    ratio = log_ratio.exp()
    unclipped_objective = ratio * advantages
    clipped_objective = ratio.clamp(1.0 - POLICY_CLIP, 1.0 + POLICY_CLIP) * advantages
    policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()

    value_change = new_values - batch["old_values"]
    clipped_values = batch["old_values"] + value_change.clamp(-VALUE_CLIP, VALUE_CLIP)
    value_error = (new_values - returns).pow(2)
    clipped_value_error = (clipped_values - returns).pow(2)
    value_loss = 0.5 * torch.maximum(value_error, clipped_value_error).mean()

    entropy_bonus = entropy.mean()
    actor_loss = policy_loss - ENTROPY_COEF * entropy_bonus
    critic_loss = VALUE_COEF * value_loss

    with torch.no_grad():
        # 非负的 sampled reverse-KL 近似，仅用于观察 PPO 新旧策略更新幅度。
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > POLICY_CLIP).float().mean()

    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy_bonus.item(),
        "approx_kl": approx_kl.item(),
        "clip_fraction": clip_fraction.item(),
    }
    return actor_loss, critic_loss, metrics


def main():
    seed_everything(SEED)
    policy, reference, reward_model, value_model = build_models()

    print(f"device={DEVICE}; training reward model")
    train_reward_model(reward_model)

    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=POLICY_LR)
    value_optimizer = torch.optim.AdamW(value_model.parameters(), lr=VALUE_LR)

    for update in range(1, PPO_UPDATES + 1):
        # 每次更新重新 rollout，保证 PPO 数据来自最新的行为策略。
        batch = sample_rollout(policy, reference, reward_model, value_model)
        rewards, sampled_ref_kl = make_token_rewards(batch)
        advantages, returns = compute_gae(rewards, batch["old_values"])

        # rollout、reward、GAE 都是固定训练目标，不参与梯度计算。
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()
        returns = returns.detach()

        policy.train()
        value_model.train()
        for _ in range(PPO_EPOCHS):
            actor_loss, critic_loss, metrics = compute_ppo_losses(
                policy, value_model, batch, advantages, returns
            )

            # 两个 optimizer 强调 actor 和 critic 是参数独立的完整模型。
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), MAX_GRAD_NORM)
            policy_optimizer.step()
            value_optimizer.step()

        if update == 1 or update % 5 == 0:
            sample = decode(batch["response_ids"][0].tolist())
            print(
                f"update={update:02d} "
                f"reward={batch['scores'].mean().item():+.3f} "
                f"ref_kl={sampled_ref_kl.mean().item():+.4f} "
                f"policy_loss={metrics['policy_loss']:+.4f} "
                f"value_loss={metrics['value_loss']:.4f} "
                f"clipfrac={metrics['clip_fraction']:.3f} "
                f"sample={sample!r}"
            )


if __name__ == "__main__":
    main()
