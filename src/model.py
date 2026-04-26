"""Decoder-only Transformer language model for SVG generation.

Architecture follows GPT-2 style (Pre-LN Transformer):
    Token Embedding + Positional Embedding
    → [TransformerBlock × n_layer]
        ├── LayerNorm → CausalSelfAttention → Residual
        └── LayerNorm → MLP (d → 4d → d) → Residual
    → LayerNorm → Linear → logits

Reference: nanoGPT by Karpathy (https://github.com/karpathy/nanoGPT)
Key differences from nanoGPT:
    - Removed GPT-2 pretrained loading (not needed for SVG)
    - Simplified config with explicit d_ff
    - Bias disabled by default (modern practice)
    - µP support via the mup package (MuReadout, MuAdamW, set_base_shapes)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from mup import MuReadout
except ImportError:
    MuReadout = None


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    block_size: int = 1024  # max sequence length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    d_ff: int = 512  # FFN intermediate size (typically 4 * n_embd)
    dropout: float = 0.0
    bias: bool = False
    # µP (Maximal Update Parameterization) settings
    mup: bool = False
    mup_base_width: int = 128  # Tiny model's n_embd as base width


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch default requires bias)."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Q, K, V projections combined into one linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection (W_O)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # µP: attention scaling = 1/d_head instead of 1/√d_head
        d_head = config.n_embd // config.n_head
        self.attn_scale = 1.0 / d_head if config.mup else None  # None = PyTorch default 1/√d_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Compute Q, K, V for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Flash attention (PyTorch >= 2.0)
        # µP uses scale=1/d_head; SP uses default scale=1/√d_head
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
            scale=self.attn_scale,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.d_ff, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Output head: MuReadout for µP (handles logit scaling automatically),
        # nn.Linear for standard parameterization
        if config.mup:
            if MuReadout is None:
                raise ImportError("mup package is required for µP mode: pip install mup")
            self.lm_head = MuReadout(config.n_embd, config.vocab_size, bias=False)
            # No weight tying under µP — embedding and readout have different
            # scaling requirements (readout needs 1/width_mult output scaling)
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            # Weight tying: share embedding and output projection weights
            self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return number of parameters (excluding position embeddings by default)."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        _linear_types = (nn.Linear,) if MuReadout is None else (nn.Linear, MuReadout)
        if isinstance(module, _linear_types):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            idx: token indices, shape (B, T)
            targets: target token indices, shape (B, T), or None for inference

        Returns:
            (logits, loss) where loss is None if targets is None
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Sequence length {t} exceeds block_size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)      # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)       # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # MuReadout automatically divides by width_mult in µP mode
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference: only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay only on 2D+ params.

        Following nanoGPT: weight tensors (matmuls, embeddings) get decay,
        biases and LayerNorm params do not.

        Note: For µP mode, use MuAdamW from train.py instead of this method.
        MuAdamW handles per-layer LR scaling automatically.
        """
        decay_params = [p for n, p in self.named_parameters()
                        if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters()
                          if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with top-k and/or nucleus (top-p) sampling.

        Args:
            idx: conditioning sequence, shape (B, T)
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: if set, only sample from top-k tokens
            top_p: if set, nucleus sampling — keep smallest set with cumprob >= top_p
            eos_token_id: if set, stop when this token is generated

        Returns:
            Extended sequence, shape (B, T + generated)
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                # Scatter back to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Stop if EOS is generated (batch_size=1 only)
            if eos_token_id is not None and idx.size(0) == 1 and idx_next.item() == eos_token_id:
                break

        return idx
