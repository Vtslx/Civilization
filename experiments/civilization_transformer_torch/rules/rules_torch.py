from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

from ..device import resolve_device


@dataclass(frozen=True)
class RuleItem:
    id: str
    type: str
    condition: str
    effect: str
    priority: float
    source: str

    def __post_init__(self) -> None:
        if self.type not in {"hard", "soft", "conflict"}:
            raise ValueError("RuleItem.type must be hard, soft, or conflict")


@dataclass(frozen=True)
class RuleResult:
    passed: bool
    violations: list[str]
    soft_matches: list[str]
    conflicts: list[str]


class RuleEngineTorch:
    def __init__(self, rules: list[RuleItem], model_dim: int = 16, device: str | torch.device | None = None):
        self.rules = rules
        self.model_dim = model_dim
        self.device = resolve_device(device)

    def evaluate(self, text: str) -> RuleResult:
        lowered = text.lower()
        violations: list[str] = []
        soft_matches: list[str] = []
        conflicts: list[str] = []
        for rule in self.rules:
            condition = rule.condition.lower()
            if rule.type == "hard" and condition in lowered:
                violations.append(f"{rule.id}: hard rule triggered by '{rule.condition}' -> {rule.effect}")
            elif rule.type == "soft" and condition in lowered:
                soft_matches.append(f"{rule.id}: soft preference matched '{rule.condition}' -> {rule.effect}")
            elif rule.type == "conflict":
                left, sep, right = condition.partition("|")
                if sep and left.strip() in lowered and right.strip() in lowered:
                    conflicts.append(f"{rule.id}: conflict between '{left.strip()}' and '{right.strip()}'")
        return RuleResult(passed=not violations and not conflicts, violations=violations, soft_matches=soft_matches, conflicts=conflicts)

    def _hash_embedding(self, text: str) -> torch.Tensor:
        vector = torch.zeros(self.model_dim, dtype=torch.float32, device=self.device)
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.model_dim
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = torch.linalg.vector_norm(vector)
        return vector / norm if norm.item() > 0 else vector

    def encode_vectors(self) -> torch.Tensor:
        if not self.rules:
            return torch.zeros((0, self.model_dim), dtype=torch.float32, device=self.device)
        vectors = []
        for rule in self.rules:
            type_weight = {"hard": 1.5, "soft": 0.75, "conflict": 1.25}[rule.type]
            vectors.append(self._hash_embedding(f"{rule.type} {rule.condition} {rule.effect}") * rule.priority * type_weight)
        return torch.stack(vectors, dim=0)
