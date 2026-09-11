"""使用 PyTorch + Transformers 实现的最小 GPT GRPO 训练示例。

运行:
    conda activate torch
    python grpo_gpt_minimal.py

为保证离线可运行，所有模型都使用很小的 GPT-2 配置随机初始化，不下载权重。
本示例用于学习 GRPO 的分组采样、组内相对 advantage、重要性采样、clip 和
reference KL；生成文本的自然语言质量不是目标。

标准 GRPO 不需要 value model：它使用同一 prompt 下多个 response 的组内相对
奖励作为 advantage。按题目要求，本文件仍实现并训练一个独立 GPT value model，
用于拟合序列回报和监控 critic loss，但它不参与 GRPO advantage 的计算。
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
# 1. 超参数与离散文本空间
# -----------------------------------------------------------------------------
SEED = 11
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_PROMPTS = 4
GROUP_SIZE = 8
RESPONSE_LENGTH = 6
GRPO_UPDATES = 30
GRPO_EPOCHS = 4

POLICY_LR = 3e-4
VALUE_LR = 5e-4
REWARD_LR = 1e-3
REWARD_TRAIN_STEPS = 100

POLICY_CLIP = 0.2
KL_COEF = 0.04
ENTROPY_COEF = 0.005
VALUE_COEF = 0.5
MAX_GRAD_NORM = 1.0
SAMPLE_TEMPERATURE = 1.0
ADVANTAGE_EPS = 1e-4


# GPT 只接收 token id，因此最小示例可以直接使用一个小型单词词表。
TOKENS = [
    "<pad>",
    "<bos>",
    "write",
    "positive",
    "negative",
    "neutral",
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
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}
PAD_ID = TOKEN_TO_ID["<pad>"]
BOS_ID = TOKEN_TO_ID["<bos>"]
EOS_ID = TOKEN_TO_ID["."]

PROMPT_TEXTS = [
    ["<bos>", "write", "positive", "message"],
    ["<bos>", "write", "negative", "message"],
    ["<bos>", "write", "neutral", "message"],
    ["<bos>", "write", "positive", "message"],
]
PROMPT_IDS = torch.tensor(
    [[TOKEN_TO_ID[token] for token in prompt] for prompt in PROMPT_TEXTS],
    dtype=torch.long,
)
PROMPT_LENGTH = PROMPT_IDS.size(1)

POSITIVE_IDS = torch.tensor(
    [TOKEN_TO_ID[token] for token in ("good", "great", "happy", "kind")]
)
NEGATIVE_IDS = torch.tensor([TOKEN_TO_ID[token] for token in ("bad", "sad", "mean")])
GENERATABLE_IDS = torch.tensor(
    [index for index, token in enumerate(TOKENS) if token not in {"<pad>", "<bos>"}]
)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def decode(token_ids):
    return " ".join(TOKENS[int(token_id)] for token_id in token_ids)


def make_gpt_config():
    """约 6 万参数的小 GPT-2；关闭 dropout 以精确比较新旧策略概率。"""
    return GPT2Config(
        vocab_size=len(TOKENS),
        n_positions=PROMPT_LENGTH + RESPONSE_LENGTH,
        n_embd=48,
        n_layer=2,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        summary_first_dropout=0.0,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
        use_cache=False,
    )


# -----------------------------------------------------------------------------
# 2. 四个模型角色
# -----------------------------------------------------------------------------
def build_models():
    """构造策略、参考、奖励和价值模型。

    policy:
        GPT2LMHeadModel，待训练 actor，输出 pi_theta(token | context)。

    reference:
        policy 初始参数的冻结副本，提供 pi_ref，用于 KL 正则化。它不会更新。

    reward_model:
        GPT2ForSequenceClassification(num_labels=1)，读取 prompt + response，输出
        序列标量 R_phi(x, y)。先监督训练，GRPO 阶段冻结。

    value_model:
        GPT2ForTokenClassification(num_labels=1)，在每个上下文位置输出 V(s_t)。
        标准 GRPO 不使用 critic；这里按题目要求完整训练它，仅用于回报预测诊断。
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
# 3. 奖励模型监督训练
# -----------------------------------------------------------------------------
def synthetic_human_score(prompt_ids, response_ids):
    """生成奖励模型的监督标签；GRPO 本身不会直接调用此规则。

    positive prompt 偏好正向词，negative prompt 偏好负向词，neutral prompt
    偏好 okay/today/friend。连续重复 token 会被惩罚。真实系统应将这里换成人工
    标注的偏好分数或 chosen/rejected 数据。
    """
    positive_count = (
        (response_ids[..., None] == POSITIVE_IDS).any(dim=-1).sum(dim=1).float()
    )
    negative_count = (
        (response_ids[..., None] == NEGATIVE_IDS).any(dim=-1).sum(dim=1).float()
    )
    neutral_ids = torch.tensor(
        [TOKEN_TO_ID[token] for token in ("okay", "today", "friend")]
    )
    neutral_count = (
        (response_ids[..., None] == neutral_ids).any(dim=-1).sum(dim=1).float()
    )
    repeated = (response_ids[:, 1:] == response_ids[:, :-1]).sum(dim=1).float()

    prompt_type = prompt_ids[:, 2]
    scores = torch.where(
        prompt_type == TOKEN_TO_ID["positive"],
        positive_count - negative_count,
        torch.where(
            prompt_type == TOKEN_TO_ID["negative"],
            negative_count - positive_count,
            neutral_count - 0.5 * (positive_count + negative_count),
        ),
    )
    return scores - 0.25 * repeated


def make_reward_batch(batch_size):
    """创建奖励模型的 point-wise 回归训练数据。"""
    prompt_indices = torch.randint(0, len(PROMPT_IDS), (batch_size,))
    prompt_ids = PROMPT_IDS[prompt_indices]
    sampled_indices = torch.randint(
        0, len(GENERATABLE_IDS), (batch_size, RESPONSE_LENGTH)
    )
    response_ids = GENERATABLE_IDS[sampled_indices]
    labels = synthetic_human_score(prompt_ids, response_ids)
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    return full_ids.to(DEVICE), labels.to(DEVICE)


def train_reward_model(reward_model):
    """训练完整神经奖励模型，然后冻结。

    point-wise reward loss:
        L_RM = mean((R_phi(prompt, response) - human_score)^2)

    常见替代方案是 pair-wise Bradley-Terry loss：
        L_pair = -log sigmoid(R(chosen) - R(rejected))

    本例选择 MSE，以便最小代码仍能清晰展示独立 reward model 的训练和使用。
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

    # GRPO 训练 policy 时不允许梯度进入 reward model，否则评分标准会同时变化。
    reward_model.requires_grad_(False)
    reward_model.eval()


# -----------------------------------------------------------------------------
# 4. Group rollout：每个 prompt 采样 G 个 response
# -----------------------------------------------------------------------------
def mask_control_token_logits(logits):
    """禁止生成 PAD/BOS；采样与 loss 重算必须使用完全相同的分布。"""
    logits = logits.clone()
    logits[..., PAD_ID] = -torch.inf
    logits[..., BOS_ID] = -torch.inf
    return logits


def action_distribution(logits):
    """实际 token 行为分布；未使用 top-k/top-p，避免重要性比率失配。"""
    return Categorical(logits=mask_control_token_logits(logits) / SAMPLE_TEMPERATURE)


@torch.no_grad()
def sample_group_rollouts(policy, reference, reward_model):
    """对每个 prompt 独立采样 GROUP_SIZE 条 response。

    batch 排列为：
        prompt_0 的 G 条 response，prompt_1 的 G 条 response，...

    old_logprobs 是真正执行采样的行为策略 pi_old 的概率。在接下来的多个
    GRPO epoch 中保持不变，作为重要性采样比率的分母。
    """
    prompt_ids = PROMPT_IDS.to(DEVICE).repeat_interleave(GROUP_SIZE, dim=0)
    full_ids = prompt_ids
    sampled_tokens = []
    old_logprobs = []

    policy.eval()
    for _ in range(RESPONSE_LENGTH):
        old_logits = policy(input_ids=full_ids).logits[:, -1, :]
        old_dist = action_distribution(old_logits)
        action = old_dist.sample()

        sampled_tokens.append(action)
        old_logprobs.append(old_dist.log_prob(action))
        full_ids = torch.cat([full_ids, action[:, None]], dim=1)

    response_ids = torch.stack(sampled_tokens, dim=1)
    old_logprobs = torch.stack(old_logprobs, dim=1)
    scores = reward_model(input_ids=full_ids).logits.squeeze(-1)

    # reference logprob 不参与采样，但 GRPO loss 需要它计算 pi_theta 对 pi_ref
    # 的 KL。固定 response 可以一次 teacher-forcing forward 并行计算所有位置。
    ref_logprobs = evaluate_policy_logprobs(reference, prompt_ids, response_ids)
    return {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "old_logprobs": old_logprobs,
        "ref_logprobs": ref_logprobs,
        "scores": scores,
    }


# -----------------------------------------------------------------------------
# 5. GRPO 的组内相对 advantage
# -----------------------------------------------------------------------------
def compute_group_advantages(scores):
    """对每个 prompt 的 G 个奖励做组内标准化。

    对第 i 个 prompt 的第 j 个 response：

        A_ij = (r_ij - mean(r_i1...r_iG)) / (std(r_i1...r_iG) + eps)

    这正是 GRPO 区别于 PPO critic/GAE 的关键：同组 response 互为 baseline，
    高于本组平均奖励的序列得到正 advantage，低于平均值的得到负 advantage。

    返回 [num_prompts * group_size]。序列内所有 token 共享该 sequence advantage。
    """
    grouped_scores = scores.view(NUM_PROMPTS, GROUP_SIZE)
    group_mean = grouped_scores.mean(dim=1, keepdim=True)
    # unbiased=False 可在小组大小下获得稳定的总体标准差定义。
    group_std = grouped_scores.std(dim=1, keepdim=True, unbiased=False)
    advantages = (grouped_scores - group_mean) / (group_std + ADVANTAGE_EPS)
    return advantages.reshape(-1).detach()


# -----------------------------------------------------------------------------
# 6. 固定 response 的 token 概率与 value 对齐
# -----------------------------------------------------------------------------
def response_positions(prompt_ids, response_ids):
    """构造 teacher-forcing 输入并返回生成 token 对应的切片边界。

    对 [prompt, a1, a2, ..., aT]，因果 LM 在 prompt 最后位置预测 a1，
    在 a1 位置预测 a2。因此 start = prompt_length - 1。
    """
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    model_inputs = full_ids[:, :-1]
    start = prompt_ids.size(1) - 1
    end = start + response_ids.size(1)
    return model_inputs, start, end


def evaluate_policy_logprobs(model, prompt_ids, response_ids):
    """用指定策略计算固定 response 中每个 token 的 log probability。"""
    model_inputs, start, end = response_positions(prompt_ids, response_ids)
    logits = model(input_ids=model_inputs).logits[:, start:end, :]
    distribution = action_distribution(logits)
    return distribution.log_prob(response_ids)


def evaluate_policy(policy, prompt_ids, response_ids):
    """当前策略的 token logprob 和 entropy。"""
    model_inputs, start, end = response_positions(prompt_ids, response_ids)
    logits = policy(input_ids=model_inputs).logits[:, start:end, :]
    distribution = action_distribution(logits)
    return distribution.log_prob(response_ids), distribution.entropy()


def evaluate_values(value_model, prompt_ids, response_ids):
    """价值模型在生成每个 token 前的状态价值 V(s_t)。"""
    model_inputs, start, end = response_positions(prompt_ids, response_ids)
    all_values = value_model(input_ids=model_inputs).logits[..., 0]
    return all_values[:, start:end]


# -----------------------------------------------------------------------------
# 7. GRPO policy loss 与辅助 value loss
# -----------------------------------------------------------------------------
def compute_losses(policy, value_model, batch, sequence_advantages):
    """计算 GRPO actor loss 和独立价值模型 loss。

    1. 重要性采样与 clipped surrogate：

       ratio_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
               = exp(new_logprob_t - old_logprob_t)

       surrogate_t = min(
           ratio_t * A,
           clip(ratio_t, 1-eps, 1+eps) * A
       )

       rollout 来自冻结在采样时刻的 pi_old，但参数更新的是 pi_theta。ratio
       修正两者差异；clip 限制复用同一组 rollout 时策略单步变化过大。

    2. Reference KL：

       令 log_ratio_ref = log pi_ref(a|s) - log pi_theta(a|s)，使用

           KL_estimator = exp(log_ratio_ref) - log_ratio_ref - 1

       这是采样于 pi_theta 时对 KL(pi_theta || pi_ref) 的非负估计器。与简单
       log pi_theta-log pi_ref 相比，它逐样本非负且数值更适合放入 loss：

           L_actor = -mean(surrogate - beta * KL_estimator)

    3. Value loss：

       标准 GRPO 不使用 value model。这里单独令每个 token 状态预测最终序列
       reward，并用 MSE 训练：

           L_value = mean((V_psi(s_t) - stop_grad(sequence_reward))^2)

       它只用于满足完整模型示例和诊断，不影响 sequence_advantages。
    """
    new_logprobs, entropy = evaluate_policy(
        policy, batch["prompt_ids"], batch["response_ids"]
    )
    advantages = sequence_advantages[:, None].expand_as(new_logprobs)

    # pi_theta / pi_old：GRPO clipped objective 的重要性采样比率。
    log_ratio_old = new_logprobs - batch["old_logprobs"]
    ratio_old = log_ratio_old.exp()
    unclipped_objective = ratio_old * advantages
    clipped_objective = (
        ratio_old.clamp(1.0 - POLICY_CLIP, 1.0 + POLICY_CLIP) * advantages
    )
    surrogate = torch.minimum(unclipped_objective, clipped_objective)

    # 参考模型概率是 rollout 后预先算好的常量；当前策略概率每个 epoch 都重算。
    log_ratio_ref = batch["ref_logprobs"] - new_logprobs
    reference_kl = log_ratio_ref.exp() - log_ratio_ref - 1.0
    entropy_bonus = entropy.mean()
    policy_loss = -(surrogate - KL_COEF * reference_kl).mean()
    actor_loss = policy_loss - ENTROPY_COEF * entropy_bonus

    predicted_values = evaluate_values(
        value_model, batch["prompt_ids"], batch["response_ids"]
    )
    value_targets = batch["scores"][:, None].expand_as(predicted_values).detach()
    value_loss = F.mse_loss(predicted_values, value_targets)
    critic_loss = VALUE_COEF * value_loss

    with torch.no_grad():
        # 这是 pi_old 与更新后 pi_theta 的 KL 近似，用于观察更新幅度。
        old_policy_kl = ((ratio_old - 1.0) - log_ratio_old).mean()
        clip_fraction = ((ratio_old - 1.0).abs() > POLICY_CLIP).float().mean()

    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy_bonus.item(),
        "reference_kl": reference_kl.mean().item(),
        "old_policy_kl": old_policy_kl.item(),
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

    for update in range(1, GRPO_UPDATES + 1):
        # 每轮先用最新 policy 为每个 prompt 采样一组 response。
        batch = sample_group_rollouts(policy, reference, reward_model)
        group_advantages = compute_group_advantages(batch["scores"])

        policy.train()
        value_model.train()
        for _ in range(GRPO_EPOCHS):
            actor_loss, critic_loss, metrics = compute_losses(
                policy, value_model, batch, group_advantages
            )

            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), MAX_GRAD_NORM)
            policy_optimizer.step()
            value_optimizer.step()

        if update == 1 or update % 5 == 0:
            grouped_scores = batch["scores"].view(NUM_PROMPTS, GROUP_SIZE)
            best_index = grouped_scores[0].argmax().item()
            sample = decode(batch["response_ids"][best_index].tolist())
            print(
                f"update={update:02d} "
                f"reward={batch['scores'].mean().item():+.3f} "
                f"group_std={grouped_scores.std(dim=1, unbiased=False).mean().item():.3f} "
                f"ref_kl={metrics['reference_kl']:.4f} "
                f"policy_loss={metrics['policy_loss']:+.4f} "
                f"value_loss={metrics['value_loss']:.4f} "
                f"clipfrac={metrics['clip_fraction']:.3f} "
                f"best_positive_sample={sample!r}"
            )


if __name__ == "__main__":
    main()
