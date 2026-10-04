from __future__ import annotations

import argparse
import json

from .pipeline import DEFAULT_MODEL_PATH, run_qwen3_hidden_state_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the read-only Qwen3 hidden-state baseline")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", default="artifacts/civilization/hidden_state_baseline")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--samples-per-label", type=int, default=30)
    parser.add_argument("--train-per-label", type=int, default=20)
    parser.add_argument("--seeds", default="202,303,404")
    parser.add_argument("--max-lengths", default="64,128,256")
    parser.add_argument("--layers", default="")
    parser.add_argument("--skip-chat-sanity", action="store_true")
    args = parser.parse_args()

    summary = run_qwen3_hidden_state_baseline(
        output_dir=args.output_dir,
        model_path=args.model_path,
        seeds=tuple(int(value) for value in args.seeds.split(",")),
        samples_per_label=args.samples_per_label,
        train_per_label=args.train_per_label,
        max_lengths=tuple(int(value) for value in args.max_lengths.split(",")),
        batch_size=args.batch_size,
        preferred_device=None if args.device == "auto" else args.device,
        selected_layers=None if not args.layers else tuple(int(value) for value in args.layers.split(",")),
        run_chat_sanity=not args.skip_chat_sanity,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
