"""Host Civilization with a local Hugging Face model — no provider, no network.

The `local_transformers` runtime is model-agnostic: point it at any local causal
language model directory. Text-only decisions, no hidden states, no adapter
execution.

    python examples/local_model.py --model /path/to/local/model

Set CIVILIZATION_MODEL_PATH (or pass --model) to the directory that holds the
model weights, tokenizer files and config.json. If the directory is missing the
script explains what to place there instead of failing with a stack trace.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from civilization import CivilizationRequest, EmbeddedCivilization, EmbeddedConfig
from civilization.engine import DEFAULT_MODEL_PATH, MODEL_PATH_ENV


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.environ.get(MODEL_PATH_ENV, str(DEFAULT_MODEL_PATH)))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--state-dir", default="var/local-model-example")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        raise SystemExit(
            f"no local model at {model_path}\n"
            f"place one there, or point {MODEL_PATH_ENV} / --model at a local copy."
        )

    config = EmbeddedConfig(
        runtime="local_transformers",
        local_model_path=str(model_path),
        preferred_device=args.device,
        state_dir=args.state_dir,
        bearer_token_env=None,
    )
    service = EmbeddedCivilization(config)
    print(f"model:      {model_path}")
    print(f"device:     {args.device}")
    print(f"runtime:    {service.capabilities.to_dict()}")

    with service:
        prediction = service.predict(
            CivilizationRequest(
                text="Choose the supported operation: approve or reject.",
                answer_options=("approve", "reject"),
                session_id="local-model-example",
                task_name="local_model_example",
            )
        )
        print(f"decision:   {prediction.option_id} = {prediction.option_text!r}")
        print(f"runtime:    {prediction.trace.get('runtime')} on {prediction.trace.get('local_device')}")


if __name__ == "__main__":
    main()
