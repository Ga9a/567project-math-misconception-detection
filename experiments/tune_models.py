import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
REPORT_DIR = PROJECT_ROOT / "reports"
TUNING_ROOT = PROJECT_ROOT / "experiments" / "tuning_runs"


def run_command(cmd, env):
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score_key(metrics: dict):
    return (
        float(metrics["macro_f1"]),
        float(metrics["accuracy"]),
        -float(metrics["train_seconds"]),
    )


def prepare_env() -> dict:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(env["MPLCONFIGDIR"], exist_ok=True)
    return env


def candidate_spaces():
    bert_snapshot = (
        "hf_cache/models--google-bert--bert-base-uncased/snapshots/"
        "86b5e0934494bd15c9632b12f734a8a67f723594"
    )
    return {
        "xgboost": [
            {
                "name": "xgb_lr005_d8_est500_ss09_cs09_l1",
                "args": [
                    "--device", "cuda",
                    "--n-estimators", "500",
                    "--max-depth", "8",
                    "--learning-rate", "0.05",
                    "--subsample", "0.9",
                    "--colsample-bytree", "0.9",
                    "--reg-lambda", "1.0",
                    "--early-stopping-rounds", "30",
                    "--n-jobs", "8",
                ],
            },
            {
                "name": "xgb_lr003_d10_est900_ss09_cs09_l1",
                "args": [
                    "--device", "cuda",
                    "--n-estimators", "900",
                    "--max-depth", "10",
                    "--learning-rate", "0.03",
                    "--subsample", "0.9",
                    "--colsample-bytree", "0.9",
                    "--reg-lambda", "1.0",
                    "--early-stopping-rounds", "40",
                    "--n-jobs", "8",
                ],
            },
            {
                "name": "xgb_lr007_d8_est600_ss08_cs09_l1",
                "args": [
                    "--device", "cuda",
                    "--n-estimators", "600",
                    "--max-depth", "8",
                    "--learning-rate", "0.07",
                    "--subsample", "0.8",
                    "--colsample-bytree", "0.9",
                    "--reg-lambda", "1.0",
                    "--early-stopping-rounds", "30",
                    "--n-jobs", "8",
                ],
            },
            {
                "name": "xgb_lr005_d12_est700_ss09_cs08_l2",
                "args": [
                    "--device", "cuda",
                    "--n-estimators", "700",
                    "--max-depth", "12",
                    "--learning-rate", "0.05",
                    "--subsample", "0.9",
                    "--colsample-bytree", "0.8",
                    "--reg-lambda", "2.0",
                    "--early-stopping-rounds", "40",
                    "--n-jobs", "8",
                ],
            },
            {
                "name": "xgb_lr004_d10_est1200_ss085_cs085_l15",
                "args": [
                    "--device", "cuda",
                    "--n-estimators", "1200",
                    "--max-depth", "10",
                    "--learning-rate", "0.04",
                    "--subsample", "0.85",
                    "--colsample-bytree", "0.85",
                    "--reg-lambda", "1.5",
                    "--early-stopping-rounds", "50",
                    "--n-jobs", "8",
                ],
            },
        ],
        "mlp": [
            {
                "name": "mlp_h768_do02_lr1e3_bs1024",
                "args": [
                    "--device", "cuda",
                    "--epochs", "20",
                    "--batch-size", "1024",
                    "--hidden-dim", "768",
                    "--dropout", "0.2",
                    "--learning-rate", "0.001",
                    "--weight-decay", "0.0001",
                    "--patience", "5",
                ],
            },
            {
                "name": "mlp_h1024_do02_lr1e3_bs1024",
                "args": [
                    "--device", "cuda",
                    "--epochs", "20",
                    "--batch-size", "1024",
                    "--hidden-dim", "1024",
                    "--dropout", "0.2",
                    "--learning-rate", "0.001",
                    "--weight-decay", "0.0001",
                    "--patience", "5",
                ],
            },
            {
                "name": "mlp_h1024_do03_lr8e4_bs1024",
                "args": [
                    "--device", "cuda",
                    "--epochs", "24",
                    "--batch-size", "1024",
                    "--hidden-dim", "1024",
                    "--dropout", "0.3",
                    "--learning-rate", "0.0008",
                    "--weight-decay", "0.0002",
                    "--patience", "6",
                ],
            },
            {
                "name": "mlp_h768_do01_lr15e3_bs512",
                "args": [
                    "--device", "cuda",
                    "--epochs", "24",
                    "--batch-size", "512",
                    "--hidden-dim", "768",
                    "--dropout", "0.1",
                    "--learning-rate", "0.0015",
                    "--weight-decay", "0.00005",
                    "--patience", "6",
                ],
            },
            {
                "name": "mlp_h512_do02_lr2e3_bs1024",
                "args": [
                    "--device", "cuda",
                    "--epochs", "20",
                    "--batch-size", "1024",
                    "--hidden-dim", "512",
                    "--dropout", "0.2",
                    "--learning-rate", "0.002",
                    "--weight-decay", "0.0001",
                    "--patience", "5",
                ],
            },
        ],
        "bert": [
            {
                "name": "bert_ep2_lr2e5_bs32_len192",
                "args": [
                    "--device", "cuda",
                    "--model-name", bert_snapshot,
                    "--hf-cache-dir", "hf_cache",
                    "--epochs", "2",
                    "--batch-size", "32",
                    "--max-length", "192",
                    "--learning-rate", "2e-5",
                    "--weight-decay", "0.01",
                    "--warmup-ratio", "0.1",
                    "--logging-steps", "100",
                    "--seed", "42",
                ],
            },
            {
                "name": "bert_ep2_lr3e5_bs32_len192",
                "args": [
                    "--device", "cuda",
                    "--model-name", bert_snapshot,
                    "--hf-cache-dir", "hf_cache",
                    "--epochs", "2",
                    "--batch-size", "32",
                    "--max-length", "192",
                    "--learning-rate", "3e-5",
                    "--weight-decay", "0.01",
                    "--warmup-ratio", "0.1",
                    "--logging-steps", "100",
                    "--seed", "42",
                ],
            },
            {
                "name": "bert_ep2_lr15e5_bs32_len256",
                "args": [
                    "--device", "cuda",
                    "--model-name", bert_snapshot,
                    "--hf-cache-dir", "hf_cache",
                    "--epochs", "2",
                    "--batch-size", "32",
                    "--max-length", "256",
                    "--learning-rate", "1.5e-5",
                    "--weight-decay", "0.01",
                    "--warmup-ratio", "0.05",
                    "--logging-steps", "100",
                    "--seed", "42",
                ],
            },
            {
                "name": "bert_ep3_lr2e5_bs32_len192",
                "args": [
                    "--device", "cuda",
                    "--model-name", bert_snapshot,
                    "--hf-cache-dir", "hf_cache",
                    "--epochs", "3",
                    "--batch-size", "32",
                    "--max-length", "192",
                    "--learning-rate", "2e-5",
                    "--weight-decay", "0.01",
                    "--warmup-ratio", "0.1",
                    "--logging-steps", "100",
                    "--seed", "42",
                ],
            },
        ],
    }


def script_path(model_name: str) -> Path:
    return PROJECT_ROOT / "experiments" / model_name / f"train_{model_name}.py"


def canonical_output_dir(model_name: str) -> Path:
    return PROJECT_ROOT / "experiments" / model_name / "outputs"


def copy_best_run(best_run_dir: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(best_run_dir, destination)


def write_summary(all_runs: list[dict], best_by_model: dict[str, dict], overall_best: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")

    summary = {
        "generated_at": timestamp,
        "objective": "maximize validation macro_f1, break ties with accuracy",
        "best_by_model": best_by_model,
        "overall_best": overall_best,
        "all_runs": all_runs,
    }
    (REPORT_DIR / "tuning_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    with (REPORT_DIR / "tuning_results.csv").open("w", encoding="utf-8", newline="") as file:
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
            "- `bert` now loads the best validation epoch automatically when model saving is enabled.",
        ]
    )
    (REPORT_DIR / "tuning_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    env = prepare_env()
    spaces = candidate_spaces()
    all_runs = []
    best_by_model = {}

    for model_name, candidates in spaces.items():
        model_runs = []
        tuning_dir = TUNING_ROOT / model_name
        tuning_dir.mkdir(parents=True, exist_ok=True)
        script = script_path(model_name)
        for candidate in candidates:
            output_dir = tuning_dir / candidate["name"]
            if output_dir.exists():
                shutil.rmtree(output_dir)
            cmd = [PYTHON, str(script), "--output-dir", str(output_dir), *candidate["args"]]
            run_command(cmd, env)
            metrics = load_metrics(output_dir / "metrics.json")
            run_info = {
                "model": model_name,
                "run_name": candidate["name"],
                "config": candidate["args"],
                "macro_f1": float(metrics["macro_f1"]),
                "accuracy": float(metrics["accuracy"]),
                "train_seconds": float(metrics["train_seconds"]),
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
            }
            all_runs.append(run_info)
            model_runs.append(run_info)

        best = max(model_runs, key=lambda item: (item["macro_f1"], item["accuracy"], -item["train_seconds"]))
        destination = canonical_output_dir(model_name)
        copy_best_run(PROJECT_ROOT / best["output_dir"], destination)
        best["canonical_output_dir"] = str(destination.relative_to(PROJECT_ROOT))
        best_by_model[model_name] = best

    overall_best = max(
        best_by_model.values(),
        key=lambda item: (item["macro_f1"], item["accuracy"], -item["train_seconds"]),
    )
    write_summary(all_runs, best_by_model, overall_best)


if __name__ == "__main__":
    main()
