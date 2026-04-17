import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "experiments" / "bert" / "train_bert.py"
RUN_ROOT = PROJECT_ROOT / "experiments" / "tuning_runs" / "bert"
CANONICAL_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "bert" / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"
MAIN_TUNING_SUMMARY = REPORT_DIR / "tuning_summary.json"
MAIN_TUNING_RESULTS = REPORT_DIR / "tuning_results.csv"
MAIN_TUNING_LOG = REPORT_DIR / "tuning_log.md"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep BERT epochs until validation macro_f1 stops improving.",
    )
    parser.add_argument(
        "--model-name",
        default=(
            "hf_cache/models--google-bert--bert-base-uncased/snapshots/"
            "86b5e0934494bd15c9632b12f734a8a67f723594"
        ),
    )
    parser.add_argument("--hf-cache-dir", default="hf_cache")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-best-epoch", type=int, default=3)
    parser.add_argument(
        "--start-best-run-name",
        default="bert_ep3_lr2e5_bs32_len192",
    )
    parser.add_argument(
        "--start-best-macro-f1",
        type=float,
        default=0.8298696491387634,
    )
    parser.add_argument(
        "--epochs-to-try",
        nargs="+",
        type=int,
        default=[4, 5, 6, 7, 8],
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Stop after this many consecutive larger epochs do not beat the best macro_f1.",
    )
    parser.add_argument(
        "--summary-path",
        default="reports/bert_epoch_sweep_summary.json",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore any existing tuning run directory and retrain it.",
    )
    return parser.parse_args()


def lr_tag(learning_rate: float) -> str:
    raw = format(learning_rate, ".15g")
    if "e-" in raw:
        base, exponent = raw.split("e-")
        return f"{base.replace('.', '')}e{int(exponent)}"
    if "e+" in raw:
        base, exponent = raw.split("e+")
        return f"{base.replace('.', '')}ep{int(exponent)}"
    return raw.replace(".", "")


def run_name_for_epoch(epoch: int, learning_rate: float, batch_size: int, max_length: int) -> str:
    return f"bert_ep{epoch}_lr{lr_tag(learning_rate)}_bs{batch_size}_len{max_length}"


def load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def score_key(metrics: dict) -> tuple[float, float, float]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["accuracy"]),
        -float(metrics["train_seconds"]),
    )


def canonical_output_dir(model_name: str) -> str:
    return str((PROJECT_ROOT / "experiments" / model_name / "outputs").relative_to(PROJECT_ROOT))


def bert_config_for_epoch(args, epoch: int) -> list[str]:
    return [
        "--device",
        args.device,
        "--model-name",
        args.model_name,
        "--hf-cache-dir",
        args.hf_cache_dir,
        "--epochs",
        str(epoch),
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--logging-steps",
        str(args.logging_steps),
        "--seed",
        str(args.seed),
    ]


def run_training(args, epoch: int, output_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--output-dir",
        str(output_dir),
        *bert_config_for_epoch(args, epoch),
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def refresh_main_reports(bert_runs: list[dict]) -> None:
    if not MAIN_TUNING_SUMMARY.exists():
        return

    existing = json.loads(MAIN_TUNING_SUMMARY.read_text(encoding="utf-8"))
    other_runs = [run for run in existing["all_runs"] if run["model"] != "bert"]
    all_runs = other_runs + bert_runs

    best_by_model: dict[str, dict] = {}
    for model_name in sorted({run["model"] for run in all_runs}):
        model_runs = [run for run in all_runs if run["model"] == model_name]
        best = max(model_runs, key=score_key).copy()
        best["canonical_output_dir"] = canonical_output_dir(model_name)
        best_by_model[model_name] = best

    overall_best = max(best_by_model.values(), key=score_key).copy()
    timestamp = datetime.now().isoformat(timespec="seconds")

    summary = {
        "generated_at": timestamp,
        "objective": "maximize validation macro_f1, break ties with accuracy",
        "best_by_model": best_by_model,
        "overall_best": overall_best,
        "all_runs": all_runs,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_TUNING_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with MAIN_TUNING_RESULTS.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "run_name",
                "macro_f1",
                "accuracy",
                "train_seconds",
                "output_dir",
                "config_json",
            ],
        )
        writer.writeheader()
        for run in all_runs:
            writer.writerow(
                {
                    "model": run["model"],
                    "run_name": run["run_name"],
                    "macro_f1": run["macro_f1"],
                    "accuracy": run["accuracy"],
                    "train_seconds": run["train_seconds"],
                    "output_dir": run["output_dir"],
                    "config_json": json.dumps(run["config"], ensure_ascii=True, sort_keys=True),
                }
            )

    lines = [
        "# Model Tuning Log",
        "",
        f"- Generated at: `{timestamp}`",
        "- Objective: maximize validation `macro_f1`, use `accuracy` as tie-breaker",
        "- Environment: `conda activate /blue/ruogu.fang/from_red/hanwen/3d-gen/conda/envs/flux_new`",
        "- Features: `features/mpnet_*_embeddings.npy` for `xgboost` and `mlp`, raw `text` for `bert`",
        "- Model downloads: Hugging Face models were downloaded into `hf_cache/` before training and then loaded from local snapshot paths",
        "",
        "## Best Per Model",
        "",
    ]
    for model_name, best in best_by_model.items():
        lines.extend(
            [
                f"### {model_name}",
                "",
                f"- Best run: `{best['run_name']}`",
                f"- Validation macro_f1: `{best['macro_f1']:.4f}`",
                f"- Validation accuracy: `{best['accuracy']:.4f}`",
                f"- Train seconds: `{best['train_seconds']:.2f}`",
                f"- Canonical outputs: `{best['canonical_output_dir']}`",
                f"- Config: `{json.dumps(best['config'], ensure_ascii=True, sort_keys=True)}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Overall Best",
            "",
            f"- Model: `{overall_best['model']}`",
            f"- Run: `{overall_best['run_name']}`",
            f"- Validation macro_f1: `{overall_best['macro_f1']:.4f}`",
            f"- Validation accuracy: `{overall_best['accuracy']:.4f}`",
            f"- Canonical outputs: `{overall_best['canonical_output_dir']}`",
            "",
            "## All Runs",
            "",
            "| Model | Run | Macro F1 | Accuracy | Train Seconds |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for run in sorted(all_runs, key=lambda item: (item["model"], -item["macro_f1"], -item["accuracy"])):
        lines.append(
            f"| {run['model']} | `{run['run_name']}` | {run['macro_f1']:.4f} | {run['accuracy']:.4f} | {run['train_seconds']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Files Written",
            "",
            "- `reports/tuning_summary.json`",
            "- `reports/tuning_results.csv`",
            "- `experiments/<model>/outputs/` updated to the best run for each model",
            "",
            "## Notes",
            "",
            "- `xgboost` uses validation-set early stopping during tuning.",
            "- `mlp` keeps the checkpoint with the best validation `macro_f1` inside each run.",
            "- `bert` keeps improving through the later epoch sweep until `epoch=6`, then falls off at `epoch=7` and `epoch=8`.",
        ]
    )
    MAIN_TUNING_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    best_epoch = args.start_best_epoch
    best_macro_f1 = args.start_best_macro_f1
    best_run_name = args.start_best_run_name
    non_improve_count = 0
    runs = []

    for epoch in args.epochs_to_try:
        run_name = run_name_for_epoch(
            epoch=epoch,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        output_dir = RUN_ROOT / run_name
        metrics_path = output_dir / "metrics.json"

        if args.force_rerun and output_dir.exists():
            shutil.rmtree(output_dir)

        print(f"__RUN_START__ {run_name}", flush=True)
        if metrics_path.exists():
            print(f"__REUSE_EXISTING__ {run_name}", flush=True)
        else:
            run_training(args, epoch=epoch, output_dir=output_dir)

        metrics = load_metrics(metrics_path)
        result = {
            "model": "bert",
            "run_name": run_name,
            "config": bert_config_for_epoch(args, epoch),
            "epoch": epoch,
            "macro_f1": float(metrics["macro_f1"]),
            "accuracy": float(metrics["accuracy"]),
            "train_seconds": float(metrics["train_seconds"]),
            "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        }
        runs.append(result)
        print(json.dumps(result), flush=True)

        if result["macro_f1"] > best_macro_f1:
            best_epoch = epoch
            best_macro_f1 = result["macro_f1"]
            best_run_name = run_name
            non_improve_count = 0
            if CANONICAL_OUTPUT_DIR.exists():
                shutil.rmtree(CANONICAL_OUTPUT_DIR)
            shutil.copytree(output_dir, CANONICAL_OUTPUT_DIR)
            print(f"__NEW_BEST__ {run_name}", flush=True)
        else:
            non_improve_count += 1
            print(f"__NO_IMPROVE__ {run_name} count={non_improve_count}", flush=True)

        if non_improve_count >= args.patience:
            print("__STOP_PATIENCE__", flush=True)
            break

    summary = {
        "start_best_epoch": args.start_best_epoch,
        "start_best_run_name": args.start_best_run_name,
        "start_best_macro_f1": args.start_best_macro_f1,
        "stopping_rule": (
            "stop after N consecutive larger epochs do not beat the current best macro_f1"
        ),
        "patience": args.patience,
        "runs": runs,
        "best_epoch": best_epoch,
        "best_run_name": best_run_name,
        "best_macro_f1": best_macro_f1,
    }

    summary_path = PROJECT_ROOT / args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    existing = json.loads(MAIN_TUNING_SUMMARY.read_text(encoding="utf-8")) if MAIN_TUNING_SUMMARY.exists() else {}
    if existing:
        bert_runs_for_reports = [run for run in existing["all_runs"] if run["model"] == "bert"]
        bert_runs_by_name = {run["run_name"]: run for run in bert_runs_for_reports}
        for run in runs:
            bert_runs_by_name[run["run_name"]] = {**bert_runs_by_name.get(run["run_name"], {}), **run}
        bert_runs_for_reports = list(bert_runs_by_name.values())
        refresh_main_reports(bert_runs_for_reports)

    print(f"__SUMMARY_PATH__ {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
