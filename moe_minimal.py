"""A runnable, vectorized Mixture-of-Experts language-model example.

The routing path contains no Python loop over tokens or experts.  It uses
Top-K token-choice routing, capacity dropping, and batched expert matmuls.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMoE(nn.Module):
    """Top-K sparse MoE FFN with vectorized dispatch and combine."""

    def __init__(
        self,
        d_model: int,
        d_expert: int,
        n_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("top_k must be in [1, n_experts]")

        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Expert parameters are stacked as [expert, input/output dimension].
        # This layout lets one batched matmul process all selected experts.
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_expert))
        self.b1 = nn.Parameter(torch.zeros(n_experts, d_expert))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_expert, d_model))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Flatten [batch, sequence, hidden] into the token dimension.
        original_shape = x.shape
        tokens = x.reshape(-1, original_shape[-1])
        n_tokens = tokens.shape[0]

        # The router produces a distribution for every token, then selects K
        # experts.  topk is a fused tensor operation, not a Python selection loop.
        router_logits = self.router(tokens)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_gates, topk_experts = torch.topk(router_probs, self.top_k, dim=-1)
        topk_gates = topk_gates / topk_gates.sum(dim=-1, keepdim=True)

        # Make one row per token-expert route: [token_count * K].
        route_experts = topk_experts.reshape(-1)
        route_gates = topk_gates.reshape(-1)
        route_tokens = (
            torch.arange(n_tokens, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )

        # Calculate each route's position inside its expert without iterating
        # over experts. Sorting groups routes; cummax finds each group start.
        n_routes = route_experts.numel()
        sort_order = torch.argsort(route_experts, stable=True)
        sorted_experts = route_experts[sort_order]
        route_numbers = torch.arange(n_routes, device=x.device)
        new_group = torch.ones(n_routes, dtype=torch.bool, device=x.device)
        new_group[1:] = sorted_experts[1:] != sorted_experts[:-1]
        group_starts = torch.where(new_group, route_numbers, torch.zeros_like(route_numbers))
        group_starts = torch.cummax(group_starts, dim=0).values
        sorted_positions = route_numbers - group_starts
        route_positions = torch.empty_like(sorted_positions)
        route_positions[sort_order] = sorted_positions

        # Capacity is a practical memory bound used by large MoE systems.
        capacity = max(1, math.ceil(self.capacity_factor * n_routes / self.n_experts))
        keep = route_positions < capacity
        valid_routes = torch.nonzero(keep, as_tuple=False).squeeze(1)
        valid_tokens = route_tokens[valid_routes]
        valid_experts = route_experts[valid_routes]
        valid_gates = route_gates[valid_routes]

        # Sparse dispatch: only selected token-expert pairs enter expert FFNs.
        # Indexing [E, ...] by valid_experts creates [valid_routes, ...], so
        # bmm applies a different expert weight matrix to every selected route.
        dispatched = tokens[valid_tokens]
        hidden = torch.bmm(
            dispatched.unsqueeze(1), self.w1[valid_experts]
        ).squeeze(1)
        hidden = F.silu(hidden + self.b1[valid_experts])
        expert_output = torch.bmm(
            hidden.unsqueeze(1), self.w2[valid_experts]
        ).squeeze(1)
        expert_output = expert_output + self.b2[valid_experts]

        # Sparse combine: scatter-add K weighted expert results back to tokens.
        output = torch.zeros_like(tokens)
        output.index_add_(0, valid_tokens, expert_output * valid_gates.unsqueeze(-1))

        # Switch-style auxiliary loss encourages balanced probability and load.
        # These reductions are vectorized across all experts.
        importance = router_probs.mean(dim=0)
        load = F.one_hot(topk_experts, num_classes=self.n_experts).float().mean(dim=(0, 1))
        load_balance_loss = self.n_experts * torch.sum(importance * load)
        router_z_loss = torch.square(torch.logsumexp(router_logits, dim=-1)).mean()

        return (
            output.reshape(original_shape),
            load_balance_loss,
            router_z_loss,
        )


class MoEBlock(nn.Module):
    """One Transformer block: causal attention followed by a sparse MoE FFN."""

    def __init__(self, d_model: int, n_heads: int, moe: SparseMoE) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln_attn = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln_moe = nn.LayerNorm(d_model)
        self.moe = moe

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv(self.ln_attn(x))
        qkv = qkv.reshape(batch_size, sequence_length, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        # PyTorch's fused scaled dot-product attention supplies the causal mask.
        attention = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attention = attention.transpose(1, 2).reshape(batch_size, sequence_length, self.d_model)
        x = x + self.out_proj(attention)

        moe_output, aux_loss, z_loss = self.moe(self.ln_moe(x))
        return x + moe_output, aux_loss, z_loss


class TinyMoELanguageModel(nn.Module):
    """Small causal LM showing where an MoE FFN sits in a real decoder block."""

    def __init__(
        self,
        vocab_size: int = 256,
        max_sequence_length: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_experts: int = 4,
        d_expert: int = 256,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_sequence_length, d_model)
        self.block = MoEBlock(
            d_model,
            n_heads,
            SparseMoE(d_model, d_expert, n_experts, top_k=top_k),
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        _, sequence_length = input_ids.shape
        positions = torch.arange(sequence_length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x, aux_loss, z_loss = self.block(x)
        logits = self.lm_head(self.final_norm(x))

        language_model_loss = None
        if labels is not None:
            # Next-token prediction for a causal language model.
            language_model_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)
            )
        return logits, language_model_loss, aux_loss, z_loss


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMoELanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # A single optimization step proves that forward, routing, and backward run.
    input_ids = torch.randint(0, 256, (2, 32), device=device)
    logits, lm_loss, aux_loss, z_loss = model(input_ids, labels=input_ids)
    total_loss = lm_loss + 0.01 * aux_loss + 0.001 * z_loss
    total_loss.backward()
    optimizer.step()

    print(f"device={device}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(
        f"lm_loss={lm_loss.item():.4f} "
        f"aux_loss={aux_loss.item():.4f} "
        f"z_loss={z_loss.item():.4f}"
    )


if __name__ == "__main__":
    main()
