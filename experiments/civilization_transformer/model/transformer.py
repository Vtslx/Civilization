from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def layer_norm(values: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = values.mean(axis=-1, keepdims=True)
    variance = ((values - mean) ** 2).mean(axis=-1, keepdims=True)
    return (values - mean) / np.sqrt(variance + eps)


def gelu(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * values**3)))


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 64
    model_dim: int = 16
    hidden_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    max_seq_len: int = 32
    seed: int = 7

    def __post_init__(self) -> None:
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")


@dataclass
class ModelOutput:
    logits: np.ndarray
    hidden_states: list[np.ndarray]
    attention_weights: list[np.ndarray]
    final_hidden: np.ndarray


class TransformerBlock:
    def __init__(self, config: TransformerConfig, rng: np.random.Generator):
        self.config = config
        d = config.model_dim
        h = config.hidden_dim
        scale = 1.0 / np.sqrt(d)
        self.wq = rng.normal(0.0, scale, size=(d, d))
        self.wk = rng.normal(0.0, scale, size=(d, d))
        self.wv = rng.normal(0.0, scale, size=(d, d))
        self.wo = rng.normal(0.0, scale, size=(d, d))
        self.w1 = rng.normal(0.0, scale, size=(d, h))
        self.b1 = np.zeros(h)
        self.w2 = rng.normal(0.0, 1.0 / np.sqrt(h), size=(h, d))
        self.b2 = np.zeros(d)

    def forward(self, hidden: np.ndarray, attention_bias: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        batch, seq_len, model_dim = hidden.shape
        head_count = self.config.num_heads
        head_dim = model_dim // head_count

        q = hidden @ self.wq
        k = hidden @ self.wk
        v = hidden @ self.wv

        q = q.reshape(batch, seq_len, head_count, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, head_count, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, head_count, head_dim).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        if attention_bias is not None:
            scores = scores + attention_bias
        attention = softmax(scores, axis=-1)
        context = attention @ v
        context = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, model_dim)

        attended = context @ self.wo
        hidden = layer_norm(hidden + attended)
        mlp_out = gelu(hidden @ self.w1 + self.b1) @ self.w2 + self.b2
        hidden = layer_norm(hidden + mlp_out)
        return hidden, attention


class MiniTransformer:
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        scale = 1.0 / np.sqrt(config.model_dim)
        self.token_embedding = self.rng.normal(0.0, scale, size=(config.vocab_size, config.model_dim))
        self.position_embedding = self.rng.normal(0.0, scale, size=(config.max_seq_len, config.model_dim))
        self.blocks = [TransformerBlock(config, self.rng) for _ in range(config.num_layers)]
        self.lm_head = self.rng.normal(0.0, scale, size=(config.model_dim, config.vocab_size))

    def forward(
        self,
        input_ids: np.ndarray,
        extra_tokens: np.ndarray | None = None,
        attention_bias: np.ndarray | None = None,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if input_ids.max(initial=0) >= self.config.vocab_size or input_ids.min(initial=0) < 0:
            raise ValueError("input_ids contain token ids outside the vocabulary")

        hidden = self.token_embedding[input_ids]
        if extra_tokens is not None:
            if extra_tokens.ndim == 2:
                extra_tokens = np.broadcast_to(extra_tokens[None, :, :], (hidden.shape[0], *extra_tokens.shape))
            if extra_tokens.shape[0] != hidden.shape[0] or extra_tokens.shape[2] != self.config.model_dim:
                raise ValueError("extra_tokens must have shape [batch, memory_seq, model_dim] or [memory_seq, model_dim]")
            hidden = np.concatenate([hidden, extra_tokens], axis=1)

        seq_len = hidden.shape[1]
        if seq_len > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        hidden = hidden + self.position_embedding[:seq_len][None, :, :]
        hidden_states = [hidden.copy()]
        attentions: list[np.ndarray] = []
        for block in self.blocks:
            hidden, attention = block.forward(hidden, attention_bias=attention_bias)
            hidden_states.append(hidden.copy())
            attentions.append(attention.copy())

        logits = hidden @ self.lm_head
        return ModelOutput(logits=logits, hidden_states=hidden_states, attention_weights=attentions, final_hidden=hidden)
