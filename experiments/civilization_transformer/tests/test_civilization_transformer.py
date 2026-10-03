import unittest

import numpy as np

from experiments.civilization_transformer.memory import MemoryEncoder, MemoryItem
from experiments.civilization_transformer.model import CivilizationBlock, MiniTransformer, TransformerConfig
from experiments.civilization_transformer.rules import RuleEngine, RuleItem
from experiments.civilization_transformer.state import StateConfig, StateEncoder, StateGate


class CivilizationTransformerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TransformerConfig(vocab_size=48, model_dim=12, hidden_dim=24, num_heads=3, num_layers=2, max_seq_len=16)
        self.input_ids = np.array([[1, 2, 3, 4]], dtype=int)

    def test_phase_1_minimal_transformer_exports_logits_hidden_states_and_attention(self) -> None:
        model = MiniTransformer(self.config)
        output = model.forward(self.input_ids)

        self.assertEqual(output.logits.shape, (1, 4, self.config.vocab_size))
        self.assertEqual(len(output.hidden_states), self.config.num_layers + 1)
        self.assertEqual(len(output.attention_weights), self.config.num_layers)
        self.assertEqual(output.hidden_states[0].shape, (1, 4, self.config.model_dim))
        self.assertEqual(output.attention_weights[0].shape, (1, self.config.num_heads, 4, 4))
        np.testing.assert_allclose(output.attention_weights[0].sum(axis=-1), np.ones((1, self.config.num_heads, 4)), atol=1e-7)

    def test_phase_2_memory_tokens_change_output_and_receive_attention(self) -> None:
        model = MiniTransformer(self.config)
        encoder = MemoryEncoder(self.config.model_dim)
        memory = encoder.encode(
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

        without_memory = model.forward(self.input_ids)
        with_memory = model.forward(self.input_ids, extra_tokens=memory)

        self.assertEqual(with_memory.logits.shape[1], self.input_ids.shape[1] + 1)
        self.assertGreater(np.linalg.norm(with_memory.logits[:, :4, :] - without_memory.logits), 1e-6)
        memory_attention = with_memory.attention_weights[-1][..., -1]
        self.assertGreater(float(memory_attention.mean()), 0.0)

    def test_phase_3_state_encoder_and_gate_change_control_signals(self) -> None:
        encoder = StateEncoder(self.config.model_dim)
        gate = StateGate()
        strict = StateConfig(rigor=1.0, creativity=0.0, defensiveness=1.0)
        creative = StateConfig(rigor=0.0, creativity=1.0, defensiveness=0.0)

        strict_vector = encoder.encode(strict)
        creative_vector = encoder.encode(creative)
        strict_signals = gate.signals(strict)
        creative_signals = gate.signals(creative)

        self.assertEqual(strict_vector.shape, (self.config.model_dim,))
        self.assertGreater(np.linalg.norm(strict_vector - creative_vector), 1e-6)
        self.assertLess(strict_signals.temperature, creative_signals.temperature)
        self.assertGreater(strict_signals.rule_scale, creative_signals.rule_scale)
        self.assertGreater(strict_signals.attention_bias, creative_signals.attention_bias)

    def test_phase_4_rule_engine_audits_hard_soft_and_conflict_rules(self) -> None:
        engine = RuleEngine(
            [
                RuleItem(id="r-hard", type="hard", condition="unsafe", effect="block output", priority=1.0, source="test"),
                RuleItem(id="r-soft", type="soft", condition="evidence", effect="prefer citations", priority=0.5, source="test"),
                RuleItem(id="r-conflict", type="conflict", condition="always|never", effect="mark contradiction", priority=1.0, source="test"),
            ],
            model_dim=self.config.model_dim,
        )

        result = engine.evaluate("This unsafe answer includes evidence and says always plus never.")
        vectors = engine.encode_vectors()

        self.assertFalse(result.passed)
        self.assertEqual(len(result.violations), 1)
        self.assertEqual(len(result.soft_matches), 1)
        self.assertEqual(len(result.conflicts), 1)
        self.assertEqual(vectors.shape, (3, self.config.model_dim))

    def test_phase_5_civilization_block_fuses_memory_state_and_rules(self) -> None:
        model = MiniTransformer(self.config)
        base_output = model.forward(self.input_ids)
        hidden = base_output.hidden_states[0]

        memory = MemoryEncoder(self.config.model_dim).encode(
            [
                MemoryItem("m1", "priority memory", "strict causal relation", "REQUIRES", 1.0, 1.0),
                MemoryItem("m2", "secondary memory", "creative analogy", "RELATED_TO", 0.5, 0.8),
            ]
        )
        rules = RuleEngine(
            [
                RuleItem("r1", "hard", "unsafe", "block", 1.0, "test"),
                RuleItem("r2", "soft", "evidence", "prefer", 0.7, "test"),
            ],
            model_dim=self.config.model_dim,
        ).encode_vectors()
        strict_state = StateConfig(rigor=1.0, creativity=0.0, defensiveness=1.0)
        loose_state = StateConfig(rigor=0.0, creativity=1.0, defensiveness=0.0)
        block = CivilizationBlock(self.config)

        strict_output, strict_trace = block.forward(hidden, memory, strict_state, rules)
        loose_output, loose_trace = block.forward(hidden, memory, loose_state, rules)

        self.assertEqual(strict_output.shape, hidden.shape)
        self.assertEqual(strict_trace.memory_attention.shape, (1, hidden.shape[1], memory.shape[0]))
        self.assertGreater(strict_trace.state_rule_scale, loose_trace.state_rule_scale)
        self.assertGreater(strict_trace.rule_influence_norm, 0.0)
        self.assertGreater(np.linalg.norm(strict_output - loose_output), 1e-6)

    def test_phase_6_hidden_states_are_exportable_for_semantic_observation(self) -> None:
        model = MiniTransformer(self.config)
        causal = model.forward(np.array([[1, 2, 3, 4]], dtype=int)).final_hidden.mean(axis=1)
        negation = model.forward(np.array([[5, 6, 7, 8]], dtype=int)).final_hidden.mean(axis=1)
        conflict = model.forward(np.array([[9, 10, 11, 12]], dtype=int)).final_hidden.mean(axis=1)
        observations = np.concatenate([causal, negation, conflict], axis=0)
        centered = observations - observations.mean(axis=0, keepdims=True)
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)

        self.assertEqual(observations.shape, (3, self.config.model_dim))
        self.assertGreater(float(singular_values[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
