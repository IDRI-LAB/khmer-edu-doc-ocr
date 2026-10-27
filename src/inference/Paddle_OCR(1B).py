"""Run PaddleOCR-VL baseline or LoRA inference on one image.

Examples:
    python paddle_ocr_inference.py /path/to/image.png --model baseline
    python paddle_ocr_inference.py /path/to/image.png --model adapter

The public functions are ``infer_baseline(image_path)`` and
``infer_adapter(image_path)``. Both return the OCR text.
"""

import argparse
import gc
import os
import sys
from pathlib import Path

sys.modules["flash_attn"] = None
sys.modules["torchao"] = None

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor
import peft.tuners.lora.model as lora_model

lora_model.is_bnb_available = lambda: False
lora_model.is_bnb_4bit_available = lambda: False
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

DEEPSEEK_BASELINE = (
    "" #path to your baseline model
)
DEEPSEEK_ADAPTER = (
    "" #path to your LoRA adapter model
)
TASK_PROMPT = "OCR:"


def _check_image_path(image_path: str | os.PathLike[str]) -> Path:
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    return path


def _run(image_path: str | os.PathLike[str], adapter_path: str | None) -> str:
    image_path = _check_image_path(image_path)
    if not torch.cuda.is_available():
        raise RuntimeError("PaddleOCR-VL inference requires a CUDA device.")

    processor = AutoProcessor.from_pretrained(
        PADDLE_BASELINE, trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        PADDLE_BASELINE,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().cuda()

    model = base_model
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            base_model, adapter_path
        ).merge_and_unload()
    model = model.eval().cuda()

    with Image.open(image_path) as loaded_image:
        image = loaded_image.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TASK_PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            use_cache=True,
        )
    result = processor.batch_decode(
        output_ids, skip_special_tokens=True
    )[0].strip()

    del model, base_model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return result


def infer_baseline(image_path: str | os.PathLike[str]) -> str:
    """Run PaddleOCR-VL baseline inference for ``image_path``."""
    return _run(image_path, adapter_path=None)


def infer_adapter(image_path: str | os.PathLike[str]) -> str:
    """Run PaddleOCR-VL with the configured LoRA adapter."""
    return _run(image_path, adapter_path=PADDLE_ADAPTER)


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
