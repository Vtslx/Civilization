"""Talk to a running Civilization service with the dependency-free client.

Start a service first, then run this file. Nothing here needs the engine
extras: the client uses the standard library only.

    civilization serve --provider-base-url https://provider.example.com/v1 --provider-model your-model
    python examples/remote_client.py
"""

from __future__ import annotations

import os

from civilization import CivilizationClient, CivilizationRequest

BASE_URL = os.environ.get("CIVILIZATION_BASE_URL", "http://127.0.0.1:8765")
TOKEN_ENV = "CIVILIZATION_API_TOKEN"


def main() -> None:
    client = CivilizationClient(BASE_URL, token_env=TOKEN_ENV if os.environ.get(TOKEN_ENV) else None)

    print(f"endpoint:      {client.base_url}")
    print(f"ready:         {client.ready().get('ready')}")

    capabilities = client.capabilities()
    print(f"runtime kind:  {capabilities.get('kind')}")
    print(f"adapter runs:  {capabilities.get('adapter_execution')}")
    print(f"hidden states: {capabilities.get('hidden_states')}")

    session = "remote-client-example"
    cell = client.write_memory(
        session_id=session,
        memory_system="episodic",
        content="The deployment signature and health checks are valid.",
        summary="deployment verification evidence",
    )
    print(f"memory cell:   {cell.get('cell_id')}")

    prediction = client.predict(
        CivilizationRequest(
            text="The deployment signature and health checks are valid. Choose the supported operation.",
            answer_options=("approve", "reject"),
            session_id=session,
            task_name="remote_client_example",
            memory_items=("The deployment signature and health checks are valid.",),
            rule_items=("Reject only when a critical verification is unresolved.",),
            state_values=(0.9, 0.1, 0.8),
        )
    )
    print(f"decision:      {prediction.option_id} = {prediction.option_text!r}")
    print(f"runtime used:  {prediction.trace.get('runtime')}")
    print(f"memory trace:  {prediction.memory_trace}")


if __name__ == "__main__":
    main()
