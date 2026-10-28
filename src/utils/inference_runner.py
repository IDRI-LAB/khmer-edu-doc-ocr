import argparse
import json
from pathlib import Path

from .config import CONFIG_PATH, load_config, resolve_path
from .evaluation import (
    aggregate_micro,
    evaluate_sample,
    print_micro_average,
    print_sample_result,
)
from .test_dataset import (
    get_ground_truth,
    get_image,
    get_row_id,
    load_test_split,
)


def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="Test-set row index. Omit to run every row.",
    )
    parser.add_argument(
        "--model",
        choices=("baseline", "lora", "both"),
        default="both",
        help="Which model(s) to run.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Evaluation split, e.g. test_setA or test_setB.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write predictions/report. Defaults to config inference.output_dir.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to the YAML config.",
    )
    return parser


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_inference(args, config_section, engine_cls, predict_kwargs_fn):
    config = load_config(args.config, section=config_section)
    data_config = config["data"]
    inference_config = config["inference"]
    model_config = config["model"]

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else resolve_path(inference_config["output_dir"])
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = args.split or data_config["test_split"]
    dataset = load_test_split(
        dataset_name=data_config["dataset_name"],
        subset=data_config["subset"],
        split=split,
    )
    print(
        f"Loaded {split}: {len(dataset)} rows from "
        f"{data_config['dataset_name']} ({data_config['subset']})"
    )

    if args.row is not None:
        if not -len(dataset) <= args.row < len(dataset):
            raise IndexError(
                f"Row {args.row} is out of range for a split of size {len(dataset)}."
            )
        indices = [args.row]
    else:
        indices = list(range(len(dataset)))

    local_dir = model_config.get("local_dir")
    local_model = resolve_path(local_dir) if local_dir else None
    if local_model is not None and local_model.exists():
        baseline_source = str(local_model)
    else:
        baseline_source = inference_config.get("baseline_model", model_config["name"])

    adapter_path = resolve_path(inference_config["adapter_path"])
    predict_kwargs = predict_kwargs_fn(inference_config)

    models = ["baseline", "lora"] if args.model == "both" else [args.model]
    if "lora" in models and not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

    report = {
        "config_section": config_section,
        "dataset": data_config["dataset_name"],
        "subset": data_config["subset"],
        "split": split,
        "row_indices": indices,
        "models": {},
    }

    for model_name in models:
        engine = engine_cls(
            baseline_source,
            adapter_path if model_name == "lora" else None,
        )
        records = []
        stats = []
        try:
            for index in indices:
                row = dataset[index]
                prediction = engine.predict(get_image(row), **predict_kwargs)
                record, sample_stats = evaluate_sample(
                    get_ground_truth(row), prediction
                )
                record = {
                    "index": index,
                    "id": get_row_id(row, index),
                    **record,
                }
                records.append(record)
                stats.append(sample_stats)
                print_sample_result(model_name, record)
        finally:
            engine.close()

        summary = aggregate_micro(stats)
        print_micro_average(model_name, summary)

        predictions_path = output_dir / f"{model_name}_predictions.jsonl"
        write_jsonl(predictions_path, records)
        report["models"][model_name] = {
            "model_source": baseline_source,
            "adapter_path": str(adapter_path) if model_name == "lora" else None,
            "micro_average": summary,
            "predictions": records,
        }
        with open(output_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"Predictions saved to: {predictions_path}")

    print(f"Report saved to:      {output_dir / 'report.json'}")
