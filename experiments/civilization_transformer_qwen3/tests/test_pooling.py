import pytest
import torch

from experiments.civilization_transformer_qwen3.analysis.hidden_states import (
    last_non_padding_pool,
    masked_mean_pool,
)


def test_masked_mean_pool_excludes_padding_tokens() -> None:
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])

    pooled = masked_mean_pool(hidden, mask)

    assert torch.equal(pooled, torch.tensor([[2.0, 3.0]]))


def test_last_non_padding_pool_selects_last_real_token() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
            [[5.0, 6.0], [200.0, 200.0], [300.0, 300.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    pooled = last_non_padding_pool(hidden, mask)

    assert torch.equal(pooled, torch.tensor([[3.0, 4.0], [5.0, 6.0]]))


def test_pooling_rejects_empty_attention_mask() -> None:
    hidden = torch.zeros((1, 2, 3))
    mask = torch.zeros((1, 2), dtype=torch.long)

    with pytest.raises(ValueError, match="empty sequence"):
        masked_mean_pool(hidden, mask)
    with pytest.raises(ValueError, match="empty sequence"):
        last_non_padding_pool(hidden, mask)
