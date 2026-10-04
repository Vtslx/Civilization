import torch

from civilization.research.torch_line.device import resolve_device
from civilization.research.torch_line.memory import MemoryEncoderTorch, MemoryItem
from civilization.research.torch_line.model import CivilizationBlockTorch, MiniTransformerTorch, TransformerConfigTorch
from civilization.research.torch_line.rules import RuleEngineTorch, RuleItem
from civilization.research.torch_line.state import StateConfig, StateEncoderTorch, StateGateTorch
from civilization.research.torch_line.training import run_minimal_training_loop


def _config() -> TransformerConfigTorch:
    return TransformerConfigTorch(vocab_size=48, model_dim=12, hidden_dim=24, num_heads=3, num_layers=2, max_seq_len=16, seed=11)


def _input_ids(device: torch.device) -> torch.Tensor:
    return torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)


def test_environment_resolves_available_device() -> None:
    device = resolve_device()
    assert device.type in {"cpu", "mps"}


def test_torch_transformer_exports_logits_hidden_states_and_attention_on_cpu() -> None:
    device = torch.device("cpu")
    config = _config()
    model = MiniTransformerTorch(config).to(device)
    output = model(_input_ids(device))

    assert output.logits.shape == (1, 4, config.vocab_size)
    assert len(output.hidden_states) == config.num_layers + 1
    assert len(output.attention_weights) == config.num_layers
    assert output.hidden_states[0].shape == (1, 4, config.model_dim)
    assert output.attention_weights[0].shape == (1, config.num_heads, 4, 4)
    torch.testing.assert_close(output.attention_weights[0].sum(dim=-1), torch.ones((1, config.num_heads, 4), device=device))


def test_memory_tokens_change_logits_and_receive_attention() -> None:
    device = torch.device("cpu")
    config = _config()
    model = MiniTransformerTorch(config).to(device)
    memory = MemoryEncoderTorch(config.model_dim, device=device).encode(
        [
            MemoryItem(
                id="m1",
                summary="causal memory",
                content="cause increases effect priority",
                relation_type="LEADS_TO",
                priority=1.0,
                confidence=0.9,
            )
        ]
    )

    without_memory = model(_input_ids(device))
    with_memory = model(_input_ids(device), extra_tokens=memory)

    assert with_memory.logits.shape[1] == 5
    assert torch.linalg.vector_norm(with_memory.logits[:, :4, :] - without_memory.logits).item() > 1e-6
    assert with_memory.attention_weights[-1][..., -1].mean().item() > 0.0


def test_state_encoder_and_gate_change_control_signals() -> None:
    device = torch.device("cpu")
    config = _config()
    encoder = StateEncoderTorch(config.model_dim, device=device)
    gate = StateGateTorch()
    strict = StateConfig(rigor=1.0, creativity=0.0, defensiveness=1.0)
    creative = StateConfig(rigor=0.0, creativity=1.0, defensiveness=0.0)

    strict_vector = encoder.encode(strict)
    creative_vector = encoder.encode(creative)
    strict_signals = gate.signals(strict)
    creative_signals = gate.signals(creative)

    assert strict_vector.shape == (config.model_dim,)
    assert torch.linalg.vector_norm(strict_vector - creative_vector).item() > 1e-6
    assert strict_signals.temperature < creative_signals.temperature
    assert strict_signals.rule_scale > creative_signals.rule_scale
    assert strict_signals.attention_bias > creative_signals.attention_bias


def test_rule_engine_audits_and_encodes_vectors() -> None:
    device = torch.device("cpu")
    config = _config()
    engine = RuleEngineTorch(
        [
            RuleItem(id="r-hard", type="hard", condition="unsafe", effect="block output", priority=1.0, source="test"),
            RuleItem(id="r-soft", type="soft", condition="evidence", effect="prefer citations", priority=0.5, source="test"),
            RuleItem(id="r-conflict", type="conflict", condition="always|never", effect="mark contradiction", priority=1.0, source="test"),
        ],
        model_dim=config.model_dim,
        device=device,
    )

    result = engine.evaluate("This unsafe answer includes evidence and says always plus never.")
    vectors = engine.encode_vectors()

    assert not result.passed
    assert len(result.violations) == 1
    assert len(result.soft_matches) == 1
    assert len(result.conflicts) == 1
    assert vectors.shape == (3, config.model_dim)


def test_civilization_block_torch_fuses_memory_state_and_rules() -> None:
    device = torch.device("cpu")
    config = _config()
    model = MiniTransformerTorch(config).to(device)
    hidden = model(_input_ids(device)).hidden_states[0]
    memory = MemoryEncoderTorch(config.model_dim, device=device).encode(
        [
            MemoryItem("m1", "priority memory", "strict causal relation", "REQUIRES", 1.0, 1.0),
            MemoryItem("m2", "secondary memory", "creative analogy", "RELATED_TO", 0.5, 0.8),
        ]
    )
    rules = RuleEngineTorch(
        [
            RuleItem("r1", "hard", "unsafe", "block", 1.0, "test"),
            RuleItem("r2", "soft", "evidence", "prefer", 0.7, "test"),
        ],
        model_dim=config.model_dim,
        device=device,
    ).encode_vectors()
    strict_state = StateConfig(rigor=1.0, creativity=0.0, defensiveness=1.0)
    loose_state = StateConfig(rigor=0.0, creativity=1.0, defensiveness=0.0)
    block = CivilizationBlockTorch(config).to(device)

    strict_output, strict_trace = block(hidden, memory, strict_state, rules)
    loose_output, loose_trace = block(hidden, memory, loose_state, rules)

    assert strict_output.shape == hidden.shape
    assert strict_trace.memory_attention.shape == (1, hidden.shape[1], memory.shape[0])
    assert strict_trace.state_rule_scale > loose_trace.state_rule_scale
    assert strict_trace.rule_influence_norm > 0.0
    assert torch.linalg.vector_norm(strict_output - loose_output).item() > 1e-6


def test_minimal_training_loop_reduces_loss_on_fixed_seed() -> None:
    result = run_minimal_training_loop(device="cpu", seed=123, steps=35, learning_rate=0.03)

    assert result.device == "cpu"
    assert result.seed == 123
    assert result.batch_shape == (8, 8)
    assert result.loss_decreased
    assert result.final_loss < result.initial_loss
    assert result.initial_loss > 0.0
    assert result.final_loss > 0.0
