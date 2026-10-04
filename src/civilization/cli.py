"""Command line entry point: ``civilization``.

Four commands cover the whole path from "just downloaded the repository" to a
running service:

    civilization demo       run a complete decision + memory cycle with no
                            configuration, no model, and no network
    civilization doctor     report what this machine can run
    civilization serve      host the service (provider, local model, or adapter)
    civilization predict    send one decision to a running service

The client commands never need the engine extras; hosting a service does.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import signal
import sys
import threading
from typing import Any, Sequence

from . import __version__
from .client import CivilizationClient
from .models import CivilizationRequest, MemorySystem
from .runtimes import (
    capabilities_for,
    runtime_kinds,
)

PROGRAM = "civilization"
DEFAULT_BASE_URL = os.environ.get("CIVILIZATION_BASE_URL", "http://127.0.0.1:8765")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _print(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(payload)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_header(value: str) -> tuple[str, str]:
    name, separator, header_value = value.partition("=")
    if not separator or not name.strip():
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {value!r}")
    return name.strip(), header_value


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime",
        default=os.environ.get("CIVILIZATION_RUNTIME", "provider"),
        choices=list(runtime_kinds()),
        help="decision backend to host (default: provider)",
    )
    parser.add_argument(
        "--provider-base-url",
        default=os.environ.get("CIVILIZATION_PROVIDER_BASE_URL", ""),
        help="OpenAI-compatible base URL, for example https://provider.example.com/v1",
    )
    parser.add_argument(
        "--provider-model",
        default=os.environ.get("CIVILIZATION_PROVIDER_MODEL", ""),
        help="model name passed to the provider",
    )
    parser.add_argument(
        "--api-key-env",
        default=os.environ.get("CIVILIZATION_API_KEY_ENV", "OPENAI_API_KEY"),
        help="environment variable holding the provider API key (default: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        type=_parse_header,
        metavar="NAME=VALUE",
        help="extra provider header, repeatable",
    )
    parser.add_argument(
        "--local-model-path",
        default=os.environ.get("CIVILIZATION_MODEL_PATH", ""),
        help="local model directory, for the local_transformers / local_adapter runtimes",
    )
    parser.add_argument(
        "--allow-insecure-http",
        action="store_true",
        help="allow a plain-HTTP provider endpoint (loopback is always allowed)",
    )


def _embedded_config(args: argparse.Namespace) -> Any:
    """Build an EmbeddedConfig from CLI arguments, with actionable errors."""

    from .embedded import EmbeddedConfig

    runtime = args.runtime
    if runtime == "provider":
        if not args.provider_base_url:
            raise SystemExit(
                "a provider runtime needs --provider-base-url (or CIVILIZATION_PROVIDER_BASE_URL).\n"
                "Try `civilization demo` first: it runs the full path with no configuration."
            )
        if not args.provider_model:
            raise SystemExit("a provider runtime needs --provider-model (or CIVILIZATION_PROVIDER_MODEL).")
    if runtime in {"local_transformers", "local_adapter"} and not args.local_model_path:
        raise SystemExit(
            f"the {runtime} runtime needs --local-model-path (or CIVILIZATION_MODEL_PATH) "
            "pointing at a local model directory."
        )

    return EmbeddedConfig(
        runtime=runtime,
        provider_base_url=args.provider_base_url,
        provider_model=args.provider_model,
        provider_api_key_env=args.api_key_env,
        provider_headers=dict(args.header or []),
        allow_insecure_http=getattr(args, "allow_insecure_http", False),
        local_model_path=args.local_model_path,
        host=getattr(args, "host", "127.0.0.1"),
        port=getattr(args, "port", 0),
        state_dir=getattr(args, "state_dir", "var/state"),
        bearer_token_env=getattr(args, "token_env", "CIVILIZATION_API_TOKEN"),
        provider_timeout_seconds=getattr(args, "timeout", 120.0),
        max_tokens=getattr(args, "max_tokens", 512),
    )


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #
class _OfflineDemoRuntime:
    """Deterministic offline runtime used by ``civilization demo``.

    It is not a model. It answers the decision protocol with a fixed choice and
    reports the same trace shape a real runtime reports, so the demo exercises
    the full service path — validation, memory retrieval, injection, jobs,
    metrics — without a provider, a key, or network access.
    """

    def __init__(self) -> None:
        self.request_count = 0

    def predict(self, request: Any) -> Any:
        from .engine.stages.stage45_adapter_package import Stage45InferenceResponse

        self.request_count += 1
        scores = [0.0] * len(request.answer_options)
        scores[0] = 1.0
        return Stage45InferenceResponse(
            status="ok",
            task_name=request.task_name,
            seed=request.seed,
            predicted_option_id=0,
            scores={"api_choice": scores},
            trace={
                "runtime": "offline_demo",
                "production_full_mode": True,
                "note": "deterministic offline runtime: no model was queried",
                "memory_item_count": len(request.memory_items),
                "rule_item_count": len(request.rule_items),
                "adapter_execution": False,
                "hidden_states_available": False,
                "ablation_controls_executed": False,
            },
        )


ENGINE_MISSING_HINT = (
    "hosting the service needs the engine extra:\n"
    "    pip install 'astreusn-civilization-v1[embedded]'\n"
    "from a checkout:  pip install -e '.[embedded]'"
)


def _require_engine() -> None:
    """Fail with an instruction, not a traceback, when the engine is absent."""

    try:
        import numpy  # noqa: F401  (first dependency the service chain imports)
        import torch  # noqa: F401
    except ImportError as error:
        raise SystemExit(f"the service engine is not installed ({error.name}).\n{ENGINE_MISSING_HINT}") from error


def _command_demo(args: argparse.Namespace) -> int:
    _require_engine()
    from .embedded import EmbeddedCivilization, EmbeddedConfig
    from .runtimes import RuntimeCapabilities

    state_dir = Path(args.state_dir).expanduser()
    config = EmbeddedConfig(
        runtime="provider",
        provider_base_url="https://offline.demo.invalid/v1",
        provider_model="offline-demo",
        bearer_token_env=None,
        state_dir=str(state_dir),
        replay_after_request=False,
    )
    demo_capabilities = RuntimeCapabilities(
        kind="provider",
        label="offline-demo",
        provider_agnostic=False,
        local_weights=False,
        chat_completions=False,
        hidden_states=False,
        adapter_execution=False,
        notes=(
            "deterministic offline runtime used by `civilization demo`",
            "no model is queried and no network access is used",
        ),
    )
    print(f"{PROGRAM} {__version__} — offline demo (no model, no network, no configuration)\n")

    with EmbeddedCivilization(config, runtime_factory=_OfflineDemoRuntime, capabilities=demo_capabilities) as runtime:
        client = runtime.start()
        ready = client.ready()
        declared = client.capabilities()
        print(f"service ready:      {bool(ready.get('ready'))}")
        print(f"declared runtime:   {declared.get('label')}")
        print(f"capability flags:   {json.dumps(declared, ensure_ascii=False)}")
        print(f"state directory:    {state_dir}\n")

        session = "demo-session"
        cell = client.write_memory(
            session_id=session,
            memory_system=MemorySystem.EPISODIC,
            content="The deployment signature and health checks are valid.",
            summary="deployment verification evidence",
        )
        print(f"memory written:     {cell.get('cell_id')} (episodic)")

        prediction = client.predict(
            CivilizationRequest(
                text="The deployment signature and health checks are valid. Choose the supported operation.",
                answer_options=("approve", "reject"),
                session_id=session,
                task_name="demo_decision",
                memory_items=("The deployment signature and health checks are valid.",),
                rule_items=("Reject only when a critical verification is unresolved.",),
                state_values=(0.9, 0.1, 0.8),
            )
        )
        print(f"decision:           option {prediction.option_id} = {prediction.option_text!r}")
        print(f"runtime used:       {prediction.trace.get('runtime')}")
        print(f"memory trace:       {json.dumps(dict(prediction.memory_trace), ensure_ascii=False)}")

        retrieved = client.read_memory(session_id=session, query="deployment verification evidence")
        print(f"memory read back:   {len(retrieved)} cell(s)")

    print(
        "\nWhat just happened: the service validated a request, retrieved a memory cell, "
        "injected it into the decision context, and returned an auditable trace.\n"
        "Next steps:\n"
        "  civilization doctor                                  # what can this machine run\n"
        "  civilization serve --provider-base-url <url> \\\n"
        "      --provider-model <model>                         # host with a real backend\n"
        "  civilization predict --text 'Choose.' --options approve,reject"
    )
    return 0


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def _command_doctor(args: argparse.Namespace) -> int:
    from .engine import model_paths

    report: dict[str, Any] = {
        "civilization_version": __version__,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "requested_runtime": args.runtime,
        "client_dependencies": "standard library only",
    }

    optional = {}
    for name in ("torch", "transformers", "numpy", "psutil"):
        try:
            module = __import__(name)
            optional[name] = getattr(module, "__version__", "installed")
        except ImportError:
            optional[name] = None
    report["engine_dependencies"] = optional

    report["local_checkpoint"] = {
        "environment_variable": model_paths.MODEL_PATH_ENV,
        "resolved_path": str(model_paths.DEFAULT_MODEL_PATH),
        "available": model_paths.model_available(),
        "note": "only needed by the local_transformers and local_adapter runtimes",
    }

    if args.runtime == "provider":
        key_present = bool(os.environ.get(args.api_key_env))
        report["provider"] = {
            "base_url": args.provider_base_url or None,
            "model": args.provider_model or None,
            "api_key_env": args.api_key_env,
            "api_key_present": key_present,
        }

    report["declared_capabilities"] = capabilities_for(args.runtime).to_dict()

    ready = True
    notes: list[str] = []
    if args.runtime == "provider":
        if not args.provider_base_url or not args.provider_model:
            ready = False
            notes.append("provider runtime selected but --provider-base-url/--provider-model are not set")
        elif not os.environ.get(args.api_key_env):
            notes.append(f"{args.api_key_env} is not set in this shell")
    if args.runtime in {"local_transformers", "local_adapter"} and not model_paths.model_available():
        ready = False
        notes.append(
            f"local checkpoint not found at {model_paths.DEFAULT_MODEL_PATH}; "
            f"place it there or set {model_paths.MODEL_PATH_ENV}"
        )
    if optional["torch"] is None:
        notes.append("torch is not installed: `pip install -e '.[embedded]'` enables hosting")

    report["verdict"] = {"can_serve": ready, "notes": notes}

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"{PROGRAM} {report['civilization_version']} | python {report['python']} | {report['platform']}")
        print(f"client dependencies: {report['client_dependencies']}")
        engine_deps = ", ".join(f"{k}={v}" for k, v in optional.items() if v) or "none installed"
        print(f"engine dependencies: {engine_deps}")
        for name, value in optional.items():
            if value is None:
                print(f"  missing: {name}")
        checkpoint = report["local_checkpoint"]
        state = "found" if checkpoint["available"] else "not found"
        print(f"local checkpoint:    {state} at {checkpoint['resolved_path']} ({checkpoint['environment_variable']})")
        if args.runtime == "provider":
            provider = report["provider"]
            key = "present" if provider["api_key_present"] else "missing"
            print(f"provider:            {provider['base_url'] or '<unset>'} model={provider['model'] or '<unset>'}")
            print(f"provider API key:    {provider['api_key_env']} ({key})")
        print(f"runtime:             {args.runtime}")
        print(f"declared capabilities: {json.dumps(report['declared_capabilities'], ensure_ascii=False)}")
        print(f"can serve now:       {report['verdict']['can_serve']}")
        for note in notes:
            print(f"  note: {note}")
    return 0 if not args.strict or report["verdict"]["can_serve"] else 1


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #
def _command_serve(args: argparse.Namespace) -> int:
    _require_engine()
    from .embedded import EmbeddedCivilization

    config = _embedded_config(args)
    service = EmbeddedCivilization(config)
    client = service.start()
    base_url = client.base_url
    (Path(config.state_dir).expanduser()).mkdir(parents=True, exist_ok=True)
    (Path(config.state_dir).expanduser() / "base_url.txt").write_text(base_url + "\n", encoding="utf-8")

    capabilities = capabilities_for(config.runtime).to_dict()
    print(f"{PROGRAM} {__version__} serving on {base_url}")
    print(f"runtime:    {capabilities['kind']} ({capabilities['label']})")
    print(f"state dir:  {config.state_dir}")
    print(f"capabilities: {json.dumps(capabilities, ensure_ascii=False)}")
    print("endpoints:  /health /ready /metrics /v1/capabilities /v1/predict /v1/batch /v1/jobs ...")
    print("press Ctrl-C to stop")

    stop = threading.Event()

    def _handle_signal(*_: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        while not stop.wait(1.0):
            pass
    finally:
        service.shutdown()
    print("\nstopped")
    return 0


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #
def _command_predict(args: argparse.Namespace) -> int:
    options = _split_csv(args.options)
    if len(options) < 2:
        raise SystemExit("--options needs at least two comma-separated choices, for example --options approve,reject")

    state_values: Sequence[float] | None = None
    if args.state_values:
        try:
            state_values = tuple(float(value) for value in _split_csv(args.state_values))
        except ValueError as error:
            raise SystemExit(f"--state-values must be numeric: {error}") from error

    client = CivilizationClient(
        args.base_url,
        token_env=args.token_env if os.environ.get(args.token_env) else None,
        timeout=args.timeout,
        allow_insecure_http=args.allow_insecure_http,
    )
    request_kwargs: dict[str, Any] = {
        "text": args.text,
        "answer_options": options,
        "session_id": args.session_id,
        "task_name": args.task_name,
        "memory_items": args.memory or (),
        "rule_items": args.rule or (),
        "read_memory": not args.no_memory,
    }
    if state_values is not None:
        request_kwargs["state_values"] = state_values
    request = CivilizationRequest(**request_kwargs)

    prediction = client.predict(request)
    if args.json:
        print(
            json.dumps(
                {
                    "request_id": prediction.request_id,
                    "option_id": prediction.option_id,
                    "option_text": prediction.option_text,
                    "scores": {name: list(values) for name, values in prediction.scores.items()},
                    "trace": dict(prediction.trace),
                    "memory_trace": dict(prediction.memory_trace),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"{prediction.option_id}: {prediction.option_text}")
        runtime = prediction.trace.get("runtime")
        if runtime:
            print(f"runtime: {runtime}")
        if prediction.memory_trace:
            print(f"memory:  {json.dumps(dict(prediction.memory_trace), ensure_ascii=False)}")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="AstreusN Civilization — decision, memory, job, and export service.",
        epilog="Run `civilization demo` for a complete cycle with no configuration.",
    )
    parser.add_argument("--version", action="version", version=f"{PROGRAM} {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run a full decision and memory cycle offline")
    demo.add_argument("--state-dir", default="var/demo-state", help="where the demo service keeps its state")
    demo.set_defaults(func=_command_demo)

    doctor = subparsers.add_parser("doctor", help="report what this machine can run")
    _add_runtime_arguments(doctor)
    doctor.add_argument("--json", action="store_true", help="machine-readable output")
    doctor.add_argument("--strict", action="store_true", help="exit non-zero when the runtime is not ready")
    doctor.set_defaults(func=_command_doctor)

    serve = subparsers.add_parser("serve", help="host the service")
    _add_runtime_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1", help="bind address (loopback only)")
    serve.add_argument("--port", type=int, default=8765, help="bind port (0 picks a free port)")
    serve.add_argument("--state-dir", default="var/state", help="where the service keeps jobs, exports, and memory")
    serve.add_argument("--token-env", default="CIVILIZATION_API_TOKEN", help="environment variable holding the bearer token")
    serve.add_argument("--timeout", type=float, default=120.0, help="provider request timeout in seconds")
    serve.add_argument("--max-tokens", type=int, default=512, help="provider completion budget")
    serve.set_defaults(func=_command_serve)

    predict = subparsers.add_parser("predict", help="send one decision to a running service")
    predict.add_argument("--text", required=True, help="the decision question")
    predict.add_argument("--options", required=True, help="comma-separated answer options")
    predict.add_argument("--base-url", default=DEFAULT_BASE_URL, help="service base URL")
    predict.add_argument("--session-id", default="default", help="memory session")
    predict.add_argument("--task-name", default="cli", help="task label recorded in the trace")
    predict.add_argument("--memory", action="append", default=[], help="memory evidence item, repeatable")
    predict.add_argument("--rule", action="append", default=[], help="rule item, repeatable")
    predict.add_argument("--state-values", default=None, help="three comma-separated numbers")
    predict.add_argument("--no-memory", action="store_true", help="do not read session memory for this request")
    predict.add_argument("--token-env", default="CIVILIZATION_API_TOKEN", help="environment variable holding the bearer token")
    predict.add_argument("--timeout", type=float, default=120.0, help="request timeout in seconds")
    predict.add_argument("--allow-insecure-http", action="store_true", help="allow plain HTTP to a non-loopback host")
    predict.add_argument("--json", action="store_true", help="machine-readable output")
    predict.set_defaults(func=_command_predict)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
