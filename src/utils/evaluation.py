from .cer_wer_calculation import (
    cer_normalized_stats,
    wer_normalized_stats,
    ratio_from_stats,
)
from .preprocessing import clean_text_pipeline

METRIC_NAMES = ("cer", "wer")


def evaluate_sample(ground_truth, prediction):
    """Return a per-sample record and its (distance, length) stats."""
    ground_truth = clean_text_pipeline(ground_truth)
    prediction = clean_text_pipeline(prediction)

    cer_distance, cer_length, _ = cer_normalized_stats(ground_truth, prediction)
    wer_distance, wer_length, _ = wer_normalized_stats(ground_truth, prediction)

    stats = {
        "cer_distance": cer_distance,
        "cer_length": cer_length,
        "wer_distance": wer_distance,
        "wer_length": wer_length,
    }

    metrics = {
        "cer": ratio_from_stats(cer_distance, cer_length),
        "wer": ratio_from_stats(wer_distance, wer_length),
    }

    record = {
        "ground_truth": ground_truth,
        "prediction": prediction,
        "metrics": metrics,
    }
    return record, stats


def aggregate_micro(stats_list):
    """Corpus micro-average: sum distances/lengths, then divide once."""
    totals = {
        key: sum(stats[key] for stats in stats_list)
        for key in (
            "cer_distance",
            "cer_length",
            "wer_distance",
            "wer_length",
        )
    }
    return {
        "samples": len(stats_list),
        "cer": ratio_from_stats(totals["cer_distance"], totals["cer_length"]),
        "wer": ratio_from_stats(totals["wer_distance"], totals["wer_length"]),
    }


def print_sample_result(model_name, record):
    print(f"\n[{model_name}] index={record['index']} id={record['id']}")
    print(f"GT        : {record['ground_truth']}")
    print(f"Prediction: {record['prediction']}")
    print(
        "CER: {cer:.4f}   WER: {wer:.4f}".format(**record["metrics"])
    )


def print_micro_average(model_name, summary):
    print(
        f"\n[{model_name}] MICRO-AVERAGE over {summary['samples']} sample(s): "
        f"CER={summary['cer']:.4f} | WER={summary['wer']:.4f}"
    )
