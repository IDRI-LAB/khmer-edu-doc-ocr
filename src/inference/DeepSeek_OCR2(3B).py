"""Run DeepSeek-OCR 2 baseline or LoRA inference on one image.

Examples:
    python deepseek_ocr_inference.py /path/to/image.png --model baseline
    python deepseek_ocr_inference.py /path/to/image.png --model adapter

The public functions are ``infer_baseline(image_path)`` and
``infer_adapter(image_path)``. Both return the OCR/Markdown text.
"""

import argparse
import gc
import os
import sys
import tempfile
from pathlib import Path

sys.modules["flash_attn"] = None
sys.modules["torchao"] = None

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

DEEPSEEK_BASELINE = (
    "/home/ltu_team/workspace/leangchhay/models/"
    "deepseek_ocr2/deepseek_ocr2_baseline"
)
DEEPSEEK_ADAPTER = (
    "/home/ltu_team/workspace/leangchhay/models/"
    "deepseek_ocr2/v2_deepseek_ocr_lora_v2"
)


def _check_image_path(image_path: str | os.PathLike[str]) -> Path:
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    return path


def _run(image_path: str | os.PathLike[str], adapter_path: str | None) -> str:
    image = _check_image_path(image_path)
    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek-OCR inference requires a CUDA device.")

    tokenizer = AutoTokenizer.from_pretrained(
        DEEPSEEK_BASELINE, trust_remote_code=True
    )
    base_model = AutoModel.from_pretrained(
        DEEPSEEK_BASELINE,
        _attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
    ).eval().cuda().to(torch.bfloat16)

    model = base_model
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            base_model, adapter_path
        ).merge_and_unload()
    model = model.eval().cuda().to(torch.bfloat16)

    with tempfile.TemporaryDirectory(prefix="deepseek_ocr_") as output_dir:
        model.infer(
            tokenizer,
            prompt="<image>\nFree OCR.",
            image_file=str(image),
            output_path=output_dir,
            image_size=768,
            base_size=1024,
            crop_mode=True,
            save_results=True,
            test_compress=False,
        )
        result_path = Path(output_dir) / "result.mmd"
        if not result_path.is_file():
            raise RuntimeError(
                f"DeepSeek-OCR did not create an output file: {result_path}"
            )
        result = result_path.read_text(encoding="utf-8").strip()

    del model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def infer_baseline(image_path: str | os.PathLike[str]) -> str:
    """Run DeepSeek-OCR 2 baseline inference for ``image_path``."""
    return _run(image_path, adapter_path=None)


def infer_adapter(image_path: str | os.PathLike[str]) -> str:
    """Run DeepSeek-OCR 2 with the configured LoRA adapter."""
    return _run(image_path, adapter_path=DEEPSEEK_ADAPTER)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path)
    parser.add_argument(
        "--model",
        choices=("baseline", "adapter"),
        default="baseline",
        help="Use the base model or the configured LoRA adapter.",
    )
    args = parser.parse_args()
    infer = infer_baseline if args.model == "baseline" else infer_adapter
    print(infer(args.image_path))


if __name__ == "__main__":
    main()
