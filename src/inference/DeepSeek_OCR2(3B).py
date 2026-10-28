"""Run DeepSeek-OCR-2 baseline and/or LoRA inference on the KhmerEduDoc test set.

Examples:
    python "src/inference/DeepSeek_OCR2(3B).py" --row 0
    python "src/inference/DeepSeek_OCR2(3B).py" --row 10 --model lora --split test_setA
    python "src/inference/DeepSeek_OCR2(3B).py" --model both --split test_setB

If --row is omitted, every row in the selected split is processed. Predictions,
per-sample metrics, and the micro-average report are written to the output
directory configured under inference.output_dir.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.inference_engine import DeepSeekOCREngine
from utils.inference_runner import build_parser, run_inference


def build_predict_kwargs(inference_config):
    return {
        "prompt": inference_config.get("prompt", "<image>\nFree OCR."),
        "image_size": inference_config.get("image_size", 768),
        "base_size": inference_config.get("base_size", 1024),
        "crop_mode": inference_config.get("crop_mode", True),
    }


def main():
    args = build_parser(__doc__).parse_args()
    run_inference(
        args,
        config_section="deepseek",
        engine_cls=DeepSeekOCREngine,
        predict_kwargs_fn=build_predict_kwargs,
    )


if __name__ == "__main__":
    main()
