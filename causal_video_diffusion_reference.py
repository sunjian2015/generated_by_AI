"""
Causal Video Diffusion with KV Cache - Minimal PyTorch Reference Implementation

Based on CausVid (https://arxiv.org/abs/2412.07772) and Self-Forcing papers.

Key concepts:
1. Autoregressive (AR) generation: Generate video frames one-by-one (or in blocks)
2. KV Cache: Cache attention keys/values from already-generated frames
3. Cache clean frames: After denoising each frame, store the CLEAN (denoised) frame's KV
4. Causal mask: Each frame only attends to past frames (block-wise causal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import math


# ============================================================================
# 1. Simplified Transformer with Causal Attention and KV Cache
# ============================================================================

class CausalSelfAttention(nn.Module):
    """Self-attention with KV caching for causal video generation."""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        current_start: int = 0,
        current_end: int = 0,
        use_cache: bool = False
    ):
        """
        Args:
            x: [B, L, D] input tokens
            kv_cache: dict with 'k' [B, max_len, num_heads, head_dim] and 'v'
            current_start: start position in the sequence
            current_end: end position (current_start + L)
            use_cache: whether to use/update cache
        """
        B, L, D = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)  # [B, L, H, D_h]
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        
        if use_cache and kv_cache is not None:
            # Update cache with current K, V at positions [current_start:current_end]
            kv_cache['k'][:, current_start:current_end] = k
            kv_cache['v'][:, current_start:current_end] = v
            
            # Attention over ALL past + current tokens [0:current_end]
            k_all = kv_cache['k'][:, :current_end]  # [B, current_end, H, D_h]
            v_all = kv_cache['v'][:, :current_end]
            
            # Compute attention: Q @ K^T
            # q: [B, L, H, D_h] -> [B, H, L, D_h]
            # k_all: [B, current_end, H, D_h] -> [B, H, D_h, current_end]
            q = q.transpose(1, 2)  # [B, H, L, D_h]
            k_all = k_all.transpose(1, 2)  # [B, H, current_end, D_h]
            v_all = v_all.transpose(1, 2)  # [B, H, current_end, D_h]
            
            attn = (q @ k_all.transpose(-2, -1)) * self.scale  # [B, H, L, current_end]
            attn = F.softmax(attn, dim=-1)
            out = attn @ v_all  # [B, H, L, D_h]
            out = out.transpose(1, 2).reshape(B, L, D)  # [B, L, D]
        else:
            # Standard self-attention (training mode with causal mask)
            q = q.transpose(1, 2)  # [B, H, L, D_h]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
            
            # Apply causal mask (lower triangular)
            causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(causal_mask, float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
            out = attn @ v  # [B, H, L, D_h]
            out = out.transpose(1, 2).reshape(B, L, D)
        
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with causal self-attention and cross-attention."""
    
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CausalSelfAttention(dim, num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,  # text/image embeddings
        kv_cache: Optional[Dict] = None,
        crossattn_cache: Optional[Dict] = None,
        current_start: int = 0,
        current_end: int = 0,
        use_cache: bool = False
    ):
        # Self-attention with KV cache
        x = x + self.self_attn(
            self.norm1(x),
            kv_cache=kv_cache,
            current_start=current_start,
            current_end=current_end,
            use_cache=use_cache
        )
        
        # Cross-attention (with text/image)
        x_norm = self.norm2(x)
        x = x + self.cross_attn(x_norm, context, context, need_weights=False)[0]
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x


class SimpleDiffusionTransformer(nn.Module):
    """Simplified diffusion transformer for video generation."""
    
    def __init__(
        self,
        in_channels: int = 4,
        dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        context_dim: int = 768
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, dim)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Context projection
        self.context_proj = nn.Linear(context_dim, dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, in_channels)
        
    def forward(
        self,
        x: torch.Tensor,  # [B, F, C, H, W] or [B, L, C] flattened
        timestep: torch.Tensor,  # [B] or [B, F]
        context: torch.Tensor,  # [B, L_ctx, context_dim]
        kv_caches: Optional[List[Dict]] = None,
        crossattn_caches: Optional[List[Dict]] = None,
        current_start: int = 0,
        use_cache: bool = False
    ):
        """
        Args:
            x: input latent (noisy)
            timestep: diffusion timestep
            context: conditioning (text embeddings, etc.)
            kv_caches: list of KV cache dicts for each layer
            current_start: starting position in sequence (for cache indexing)
            use_cache: whether to use KV cache (inference) or not (training)
        """
        # Flatten spatial dimensions if needed
        if x.dim() == 5:  # [B, F, C, H, W]
            B, F, C, H, W = x.shape
            x = x.permute(0, 1, 3, 4, 2).reshape(B, F * H * W, C)  # [B, L, C]
        else:
            B, L, C = x.shape
            F, H, W = None, None, None
        
        # Project input
        x = self.input_proj(x)  # [B, L, dim]
        
        # Add time embedding
        if timestep.dim() == 1:
            t_emb = self.time_embed(self.get_timestep_embedding(timestep, self.dim))  # [B, dim]
            t_emb = t_emb.unsqueeze(1)  # [B, 1, dim]
        else:  # [B, F]
            t_emb = self.time_embed(self.get_timestep_embedding(timestep.flatten(), self.dim))
            t_emb = t_emb.view(B, -1, self.dim)  # [B, F, dim]
        
        x = x + t_emb
        
        # Project context
        context = self.context_proj(context)
        
        # Apply transformer blocks
        current_end = current_start + x.shape[1]
        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            crossattn_cache = crossattn_caches[i] if crossattn_caches is not None else None
            
            x = block(
                x,
                context,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                use_cache=use_cache
            )
        
        # Output
        x = self.out_norm(x)
        x = self.out_proj(x)  # [B, L, C]
        
        # Reshape back if needed
        if F is not None:
            x = x.view(B, F, H, W, C).permute(0, 1, 4, 2, 3)  # [B, F, C, H, W]
        
        return x
    
    @staticmethod
    def get_timestep_embedding(timesteps, embedding_dim):
        """Sinusoidal timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ============================================================================
# 2. Causal Video Generation Pipeline (Inference)
# ============================================================================

class CausalVideoInferencePipeline:
    """
    Autoregressive video generation with KV caching.
    
    Key insight: After denoising each frame block, we run a forward pass with
    timestep=0 using the CLEAN denoised output to cache its K/V for future frames.
    """
    
    def __init__(
        self,
        model: SimpleDiffusionTransformer,
        num_frames_per_block: int = 1,
        num_denoising_steps: int = 50,
        max_frames: int = 24
    ):
        self.model = model
        self.num_frames_per_block = num_frames_per_block
        self.denoising_steps = num_denoising_steps
        self.max_frames = max_frames
        
        # Denoising timestep schedule (e.g., [999, 950, 900, ..., 50, 0])
        self.timestep_schedule = torch.linspace(999, 0, num_denoising_steps + 1).long()
        
    def initialize_kv_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Initialize KV cache for all transformer layers."""
        # Assuming flattened spatial: F * H * W tokens per frame
        # For simplicity, assume 1560 tokens per frame (like in CausVid)
        tokens_per_frame = 1560
        max_seq_len = self.max_frames * tokens_per_frame
        
        kv_caches = []
        for _ in range(self.model.num_layers):
            kv_caches.append({
                'k': torch.zeros(batch_size, max_seq_len, self.model.num_layers, 
                               self.model.dim // self.model.num_layers, 
                               device=device, dtype=dtype),
                'v': torch.zeros(batch_size, max_seq_len, self.model.num_layers,
                               self.model.dim // self.model.num_layers,
                               device=device, dtype=dtype)
            })
        return kv_caches
    
    @torch.no_grad()
    def generate(
        self,
        noise: torch.Tensor,  # [B, num_frames, C, H, W]
        context: torch.Tensor,  # [B, L_ctx, context_dim]
        initial_frames: Optional[torch.Tensor] = None  # [B, num_init_frames, C, H, W]
    ) -> torch.Tensor:
        """
        Autoregressively generate video frames.
        
        Args:
            noise: Random noise for frames to generate
            context: Conditioning (e.g., text embeddings)
            initial_frames: Optional starting frames (e.g., first frame for I2V)
            
        Returns:
            Generated video frames [B, total_frames, C, H, W]
        """
        B, num_frames, C, H, W = noise.shape
        tokens_per_frame = H * W  # simplified
        device = noise.device
        dtype = noise.dtype
        
        # Initialize output
        num_init_frames = initial_frames.shape[1] if initial_frames is not None else 0
        total_frames = num_init_frames + num_frames
        output = torch.zeros(B, total_frames, C, H, W, device=device, dtype=dtype)
        
        # Initialize KV cache
        kv_caches = self.initialize_kv_cache(B, device, dtype)
        
        current_frame_idx = 0
        
        # Step 1: Cache initial frames (if provided)
        if initial_frames is not None:
            output[:, :num_init_frames] = initial_frames
            # Forward pass with t=0 to cache clean initial frames
            for i in range(0, num_init_frames, self.num_frames_per_block):
                end_idx = min(i + self.num_frames_per_block, num_init_frames)
                frame_block = initial_frames[:, i:end_idx]
                
                _ = self.model(
                    frame_block,
                    timestep=torch.zeros(B, device=device).long(),
                    context=context,
                    kv_caches=kv_caches,
                    current_start=current_frame_idx * tokens_per_frame,
                    use_cache=True
                )
                current_frame_idx += (end_idx - i)
        
        # Step 2: Autoregressively denoise and cache each frame block
        num_blocks = (num_frames + self.num_frames_per_block - 1) // self.num_frames_per_block
        
        for block_idx in range(num_blocks):
            start_frame = block_idx * self.num_frames_per_block
            end_frame = min(start_frame + self.num_frames_per_block, num_frames)
            
            # Get noisy input for this block
            noisy_block = noise[:, start_frame:end_frame]
            
            # Spatial denoising loop (multi-step diffusion)
            for step_idx in range(len(self.timestep_schedule) - 1):
                t_current = self.timestep_schedule[step_idx]
                t_next = self.timestep_schedule[step_idx + 1]
                
                # Timestep tensor [B, num_frames_in_block]
                t = t_current * torch.ones(B, end_frame - start_frame, device=device).long()
                
                # Predict clean latent (x0 prediction)
                pred_clean = self.model(
                    noisy_block,
                    timestep=t,
                    context=context,
                    kv_caches=kv_caches,
                    current_start=current_frame_idx * tokens_per_frame,
                    use_cache=True
                )
                
                # DDIM-style update: add noise for next timestep
                if step_idx < len(self.timestep_schedule) - 2:
                    noise_sample = torch.randn_like(noisy_block)
                    # Simplified: noisy_block = alpha_next * pred_clean + sigma_next * noise
                    alpha_next = (1000 - t_next) / 1000.0
                    sigma_next = t_next / 1000.0
                    noisy_block = alpha_next * pred_clean + sigma_next * noise_sample
                else:
                    # Final step: use clean prediction
                    noisy_block = pred_clean
            
            # Step 3: Store denoised output
            denoised_block = noisy_block
            output[:, current_frame_idx:current_frame_idx + (end_frame - start_frame)] = denoised_block
            
            # Step 4: **CRITICAL** - Cache the CLEAN denoised frames for future blocks
            # Run forward pass with timestep=0 to cache K/V of clean frames
            _ = self.model(
                denoised_block,
                timestep=torch.zeros(B, device=device).long(),
                context=context,
                kv_caches=kv_caches,
                current_start=current_frame_idx * tokens_per_frame,
                use_cache=True
            )
            
            current_frame_idx += (end_frame - start_frame)
        
        return output


# ============================================================================
# 3. Training Pipeline (Self-Forcing / DMD)
# ============================================================================

class SelfForcingTrainer:
    """
    Training pipeline for causal video diffusion using Distribution Matching Distillation (DMD).
    
    Key idea: 
    - Run the generator in AR mode to produce synthetic samples
    - Train a discriminator (fake_score) on these samples
    - Update generator to fool the discriminator (DMD loss)
    """
    
    def __init__(
        self,
        generator: SimpleDiffusionTransformer,
        discriminator: SimpleDiffusionTransformer,
        teacher: Optional[SimpleDiffusionTransformer] = None,
        num_frames_per_block: int = 1,
        num_denoising_steps: int = 4,  # Few-step distillation
        guidance_scale: float = 7.5
    ):
        self.generator = generator
        self.discriminator = discriminator  # fake_score model
        self.teacher = teacher  # real_score model (frozen pre-trained)
        
        self.num_frames_per_block = num_frames_per_block
        self.num_denoising_steps = num_denoising_steps
        self.guidance_scale = guidance_scale
        
        # Simplified timestep schedule for few-step generation
        self.timestep_schedule = torch.linspace(999, 0, num_denoising_steps + 1).long()
        
    def run_generator_rollout(
        self,
        noise: torch.Tensor,  # [B, F, C, H, W]
        context: torch.Tensor,
        kv_caches: List[Dict]
    ) -> torch.Tensor:
        """
        Autoregressively generate frames using the student generator.
        Returns: Generated clean frames [B, F, C, H, W]
        """
        B, num_frames, C, H, W = noise.shape
        tokens_per_frame = H * W
        device = noise.device
        
        output = torch.zeros_like(noise)
        current_frame_idx = 0
        num_blocks = (num_frames + self.num_frames_per_block - 1) // self.num_frames_per_block
        
        for block_idx in range(num_blocks):
            start_frame = block_idx * self.num_frames_per_block
            end_frame = min(start_frame + self.num_frames_per_block, num_frames)
            
            noisy_block = noise[:, start_frame:end_frame]
            
            # Multi-step denoising
            for step_idx in range(len(self.timestep_schedule) - 1):
                t_current = self.timestep_schedule[step_idx]
                t_next = self.timestep_schedule[step_idx + 1]
                
                t = t_current * torch.ones(B, end_frame - start_frame, device=device).long()
                
                # Randomly stop gradient at different steps (Self-Forcing strategy)
                # For simplicity, always compute gradients at the last step
                should_grad = (step_idx == len(self.timestep_schedule) - 2)
                
                with torch.set_grad_enabled(should_grad):
                    pred_clean = self.generator(
                        noisy_block,
                        timestep=t,
                        context=context,
                        kv_caches=kv_caches,
                        current_start=current_frame_idx * tokens_per_frame,
                        use_cache=True
                    )
                
                # Update noisy_block for next step
                if step_idx < len(self.timestep_schedule) - 2:
                    alpha_next = (1000 - t_next) / 1000.0
                    sigma_next = t_next / 1000.0
                    noisy_block = alpha_next * pred_clean.detach() + sigma_next * torch.randn_like(noisy_block)
                else:
                    noisy_block = pred_clean
            
            denoised_block = noisy_block
            output[:, current_frame_idx:current_frame_idx + (end_frame - start_frame)] = denoised_block
            
            # Cache clean output with t=0 (detached for efficiency)
            with torch.no_grad():
                _ = self.generator(
                    denoised_block,
                    timestep=torch.zeros(B, device=device).long(),
                    context=context,
                    kv_caches=kv_caches,
                    current_start=current_frame_idx * tokens_per_frame,
                    use_cache=True
                )
            
            current_frame_idx += (end_frame - start_frame)
        
        return output
    
    def compute_dmd_loss(
        self,
        generated_frames: torch.Tensor,
        context: torch.Tensor,
        unconditional_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Distribution Matching Distillation (DMD) loss.
        
        DMD loss = 0.5 * ||x0 - (x0 - grad)||^2
        where grad = (fake_score - teacher_score) / normalizer
        """
        B, F, C, H, W = generated_frames.shape
        
        # Sample random timestep for score evaluation
        t = torch.randint(20, 980, (B, F), device=generated_frames.device).long()
        
        # Add noise to generated frames
        noise = torch.randn_like(generated_frames)
        alpha = (1000 - t.float()) / 1000.0
        sigma = t.float() / 1000.0
        noisy_frames = alpha.view(B, F, 1, 1, 1) * generated_frames + \
                      sigma.view(B, F, 1, 1, 1) * noise
        
        # Compute fake score (discriminator)
        pred_fake = self.discriminator(noisy_frames, timestep=t, context=context)
        
        # Compute real score (teacher, frozen)
        with torch.no_grad():
            pred_real_cond = self.teacher(noisy_frames, timestep=t, context=context)
            pred_real_uncond = self.teacher(noisy_frames, timestep=t, context=unconditional_context)
            pred_real = pred_real_uncond + self.guidance_scale * (pred_real_cond - pred_real_uncond)
        
        # Compute gradient
        grad = pred_fake - pred_real
        
        # Normalize gradient
        normalizer = torch.abs(generated_frames - pred_real).mean(dim=[1, 2, 3, 4], keepdim=True)
        grad = grad / (normalizer + 1e-8)
        
        # DMD loss
        target = (generated_frames - grad).detach()
        loss = 0.5 * F.mse_loss(generated_frames, target)
        
        return loss
    
    def training_step(
        self,
        noise: torch.Tensor,
        context: torch.Tensor,
        unconditional_context: torch.Tensor,
        optimize_generator: bool = True
    ):
        """
        One training iteration.
        
        Args:
            noise: Random noise [B, F, C, H, W]
            context: Conditional embeddings (e.g., text)
            unconditional_context: Null embeddings for CFG
            optimize_generator: If True, compute generator loss; else discriminator loss
        """
        B = noise.shape[0]
        device = noise.device
        dtype = noise.dtype
        
        # Initialize KV cache
        kv_caches = []
        for _ in range(self.generator.num_layers):
            kv_caches.append({
                'k': torch.zeros(B, 1560 * 24, 8, 64, device=device, dtype=dtype),
                'v': torch.zeros(B, 1560 * 24, 8, 64, device=device, dtype=dtype)
            })
        
        # Step 1: Run generator rollout
        if optimize_generator:
            generated_frames = self.run_generator_rollout(noise, context, kv_caches)
            
            # Compute DMD loss
            loss = self.compute_dmd_loss(generated_frames, context, unconditional_context)
            return loss
        else:
            # Step 2: Train discriminator
            with torch.no_grad():
                generated_frames = self.run_generator_rollout(noise, context, kv_caches)
            
            # Sample timestep
            t = torch.randint(20, 980, (B, noise.shape[1]), device=device).long()
            
            # Add noise
            noise_sample = torch.randn_like(generated_frames)
            alpha = (1000 - t.float()) / 1000.0
            sigma = t.float() / 1000.0
            noisy_frames = alpha.view(B, -1, 1, 1, 1) * generated_frames + \
                          sigma.view(B, -1, 1, 1, 1) * noise_sample
            
            # Predict noise with discriminator
            pred_fake = self.discriminator(noisy_frames, timestep=t, context=context)
            
            # Denoising loss (MSE between predicted and true clean)
            loss = F.mse_loss(pred_fake, generated_frames)
            
            return loss


# ============================================================================
# 4. Usage Example
# ============================================================================

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = SimpleDiffusionTransformer(
        in_channels=4,  # VAE latent channels
        dim=512,
        num_heads=8,
        num_layers=12,
        context_dim=768  # text encoder dim
    ).to(device)
    
    print("=" * 80)
    print("Causal Video Diffusion - Autoregressive Generation with KV Cache")
    print("=" * 80)
    
    # ========================================================================
    # INFERENCE EXAMPLE
    # ========================================================================
    print("\n[1] INFERENCE: Autoregressive video generation")
    print("-" * 80)
    
    inference_pipeline = CausalVideoInferencePipeline(
        model=model,
        num_frames_per_block=1,  # Generate 1 frame at a time
        num_denoising_steps=50,
        max_frames=24
    )
    
    # Dummy inputs
    batch_size = 2
    num_frames = 8
    noise = torch.randn(batch_size, num_frames, 4, 8, 8, device=device)  # Simplified spatial dims
    context = torch.randn(batch_size, 77, 768, device=device)  # Text embeddings
    
    print(f"Input noise shape: {noise.shape}")
    print(f"Context shape: {context.shape}")
    
    # Generate
    with torch.no_grad():
        generated_video = inference_pipeline.generate(noise, context)
    
    print(f"Generated video shape: {generated_video.shape}")
    print(f"✓ Successfully generated {generated_video.shape[1]} frames autoregressively")
    
    print("\nKey mechanism:")
    print("  1. For each frame block:")
    print("     a. Run multi-step denoising (t=999 -> 0)")
    print("     b. Get clean denoised frame x0")
    print("     c. Run forward pass with t=0 to cache K/V of CLEAN frame")
    print("  2. Next frame attends to all past CLEAN frames via KV cache")
    print("  3. Cache stores: K/V from clean (denoised) frames, NOT noisy ones")
    
    # ========================================================================
    # TRAINING EXAMPLE
    # ========================================================================
    print("\n" + "=" * 80)
    print("[2] TRAINING: Self-Forcing with DMD")
    print("-" * 80)
    
    # Initialize models
    generator = SimpleDiffusionTransformer(4, 512, 8, 12, 768).to(device)
    discriminator = SimpleDiffusionTransformer(4, 512, 8, 12, 768).to(device)
    teacher = SimpleDiffusionTransformer(4, 512, 8, 12, 768).to(device)
    teacher.eval()  # Frozen teacher
    
    trainer = SelfForcingTrainer(
        generator=generator,
        discriminator=discriminator,
        teacher=teacher,
        num_frames_per_block=1,
        num_denoising_steps=4,  # Few-step distillation
        guidance_scale=7.5
    )
    
    # Dummy training data
    train_noise = torch.randn(batch_size, 6, 4, 8, 8, device=device)
    train_context = torch.randn(batch_size, 77, 768, device=device)
    train_uncond_context = torch.randn(batch_size, 77, 768, device=device)
    
    print(f"Training noise shape: {train_noise.shape}")
    
    # Simulate one training iteration
    print("\nGenerator training step:")
    gen_loss = trainer.training_step(
        train_noise, train_context, train_uncond_context, optimize_generator=True
    )
    print(f"  Generator loss: {gen_loss.item():.4f}")
    
    print("\nDiscriminator training step:")
    disc_loss = trainer.training_step(
        train_noise, train_context, train_uncond_context, optimize_generator=False
    )
    print(f"  Discriminator loss: {disc_loss.item():.4f}")
    
    print("\nTraining flow:")
    print("  1. Run generator AR rollout -> generate frames")
    print("  2. Compute DMD loss = 0.5 * ||x0 - (x0 - grad)||^2")
    print("     where grad = (fake_score - teacher_score) / normalizer")
    print("  3. Update generator to minimize DMD loss")
    print("  4. Update discriminator on generated samples")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: KV Cache in Autoregressive Diffusion")
    print("=" * 80)
    print("""
KV Cache 中存储的内容：
-----------------------
✓ 存储的是 **去噪后的干净帧** 的 K/V，而不是噪声帧
✓ 每生成一帧后，用 timestep=0 运行一次前向传播来缓存干净帧的 K/V
✓ 下一帧的注意力机制通过 KV cache 访问所有过去的干净帧

Autoregressive 去噪流程：
-------------------------
1. 对于每一帧 (或帧块):
   a. 初始化为噪声 x_T
   b. 多步去噪: x_T -> x_{T-1} -> ... -> x_0 (干净帧)
   c. 在每一步，模型通过 causal attention 访问之前已生成的干净帧的 KV cache
   
2. 生成当前帧的 x_0 后:
   a. 将其存入输出
   b. 用 t=0 运行模型，将干净帧的 K/V 存入 cache
   
3. 生成下一帧时，重复步骤 1-2

关键点：
--------
• Cache 更新时机: 每帧去噪完成后 (t=0 时)
• Cache 内容: 干净的去噪帧的 attention keys/values
• 因果性: 当前帧只能看到过去帧的 KV，通过 causal mask 或 KV cache 范围实现
• 效率: 避免重新计算已生成帧的 attention，大幅加速推理

训练策略 (Self-Forcing):
------------------------
• Distribution Matching Distillation (DMD): 用 teacher/discriminator 指导 generator
• 随机退出策略: 在不同去噪步骤随机停止梯度传播，增加训练鲁棒性
• Few-step distillation: 训练模型用 4-8 步而不是 50-1000 步完成去噪
""")
    
    print("\n✓ Reference implementation complete!")
    print("=" * 80)
