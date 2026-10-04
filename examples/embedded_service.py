"""Host Civilization inside your own process, with the backend as configuration.

The service chain does not change between backends: swap ``runtime`` and the
provider fields, and everything else — access control, queueing, jobs, exports,
Orion memory, the capability endpoint — stays the same.

    export PROVIDER_API_KEY=...                      # your provider's key
    python examples/embedded_service.py

    # or a local model instead of a provider
    python examples/embedded_service.py --local /path/to/model
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from civilization import EmbeddedCivilization, EmbeddedConfig


def build_config(args: argparse.Namespace) -> EmbeddedConfig:
    if args.local:
        return EmbeddedConfig(
            runtime="local_transformers",
            local_model_path=args.local,
            state_dir=args.state_dir,
            bearer_token_env=None,
        )
    return EmbeddedConfig(
        runtime="provider",
        provider_base_url=args.provider_base_url,
        provider_model=args.provider_model,
        provider_api_key_env=args.api_key_env,
        provider_headers={"x-provider-session": "embedded-example"},
        state_dir=args.state_dir,
        bearer_token_env=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", default=None, help="local model directory; use this instead of a provider")
    parser.add_argument(
        "--provider-base-url",
        default=os.environ.get("CIVILIZATION_PROVIDER_BASE_URL", "https://provider.example.com/v1"),
    )
    parser.add_argument("--provider-model", default=os.environ.get("CIVILIZATION_PROVIDER_MODEL", "your-model"))
    parser.add_argument("--api-key-env", default="PROVIDER_API_KEY")
    parser.add_argument("--state-dir", default="var/embedded-example")
    args = parser.parse_args()

    config = build_config(args)
    print(f"runtime: {config.runtime} | state: {Path(config.state_dir).resolve()}")

    service = EmbeddedCivilization(config)
    print(f"declared capabilities: {service.capabilities.to_dict()}")

    with service:
        client = service.start()
        print(f"service started on {client.base_url}")
        print(f"ready: {client.ready().get('ready')}")

        # In-process calls skip the network hop entirely.
        from civilization import CivilizationRequest

        prediction = service.predict(
            CivilizationRequest(
                text="The canary passed health checks and no critical alert is unresolved. Choose.",
                answer_options=("approve", "reject"),
                session_id="embedded-example",
                task_name="embedded_example",
                memory_items=("The canary passed health checks.",),
                rule_items=("Reject only when a critical verification is unresolved.",),
            )
        )
        print(f"decision: {prediction.option_id} = {prediction.option_text!r}")
        print(f"runtime:  {prediction.trace.get('runtime')}")


if __name__ == "__main__":
    main()
