# ADR-003: Reproducible OpenAI-Compatible Benchmark Calls

**Status:** Accepted  
**Date:** 2026-10-26

## Context

Repeated LongMemEval answerer runs using the same model alias and prompts have produced different answers. The benchmark adapter must support a provider-supplied seed while preserving the current behavior when no seed is requested. It also needs to run against a user-selected OpenAI-compatible gateway without storing credentials in the repository.

## Decision

- Configure custom gateways with `OPENAI_BASE_URL` and authenticate with `OPENAI_API_KEY` at runtime. Do not commit, log, or write the API key to benchmark results or memory.
- Keep `--openai-seed` optional. When supplied, pass it to OpenAI-compatible text and structured requests.
- Retry without the seed only when the provider explicitly rejects the seed field with an unsupported-parameter response. Record the seed and whether each client accepted it.
- Treat a seed as a reproducibility aid, not a guarantee of identical outputs. Compare complete runs with the same provider, model, seed, prompts, retrieval data, and benchmark cutoff before attributing a score change.

## Consequences

- Existing runs retain their request shape unless `--openai-seed` is set.
- Providers without seed support continue to work after one explicit-rejection fallback, and metadata reports that the seed was not accepted.
- Benchmark operators can select a gateway without adding provider credentials to source control.
- A seeded preflight is useful for validating a route, but only a matched full benchmark establishes score evidence.
