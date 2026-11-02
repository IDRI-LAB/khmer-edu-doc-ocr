# KhmerEduDoc OCR: DeepSeek-OCR-2 and PaddleOCR-VL LoRA

Training, inference, and evaluation code for the Khmer OCR models trained on
the **KhmerEduDoc** dataset.

- Dataset: [CADT-IDRI/KhmerEduDoc](https://huggingface.co/datasets/CADT-IDRI/KhmerEduDoc)
- DeepSeek-OCR-2 adapter: [CADT-IDRI/deepseek-ocr-2-khmer-lora-KhmerEduDoc](https://huggingface.co/CADT-IDRI/deepseek-ocr-2-khmer-lora-KhmerEduDoc)
- PaddleOCR-VL adapter: [CADT-IDRI/paddleocr-vl-khmer-lora-KhmerEduDoc](https://huggingface.co/CADT-IDRI/paddleocr-vl-khmer-lora-KhmerEduDoc)

This repository does not host the dataset or model weights.

## Repository layout

```text
.
├── configs/
│   └── lora.yaml                       # DeepSeek and Paddle configuration
├── requirements.txt
└── src/
    ├── data.py                         # Dataset loading and collators
    ├── fine-tuning/
    │   ├── Deepseek_OCR2(3B).py        # DeepSeek-OCR-2 LoRA training
    │   └── Paddle_OCR(1B).py           # PaddleOCR-VL LoRA training
    ├── inference/
    │   ├── DeepSeek_OCR2(3B).py        # DeepSeek baseline/LoRA inference
    │   └── Paddle_OCR(1B).py           # Paddle baseline/LoRA inference
    └── utils/
        ├── config.py                   # Config loading and path resolution
        ├── inference_runner.py         # Shared inference CLI
        ├── inference_engine.py         # DeepSeek and Paddle model loaders
        ├── training_utils.py           # Logs, metrics JSON, loss curves
        ├── test_dataset.py             # Hugging Face split loader
        ├── evaluation.py               # CER/WER and micro-average
        ├── preprocessing.py            # Shared text cleaning
        ├── khmer_norm.py               # Khmer normalization
        └── cer_wer_calculation.py      # CER/WER calculation
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/IDRI-LAB/khmer-edu-doc-ocr.git
cd khmer-edu-doc-ocr
```

### 2. Create the environment

```bash
conda create -n khmer-edu-ocr python=3.11 -y
conda activate khmer-edu-ocr
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

The pinned versions target CUDA 12.4, PyTorch 2.6.0, Transformers 4.56.2,
and Unsloth 2026.6.1.

### 4. Download released adapters (optional)

Only needed if you want to run inference without training first:

```bash
huggingface-cli download CADT-IDRI/deepseek-ocr-2-khmer-lora-KhmerEduDoc \
  --local-dir src/models/deepseek_ocr2_khmer_lora

huggingface-cli download CADT-IDRI/paddleocr-vl-khmer-lora-KhmerEduDoc \
  --local-dir src/models/paddleocr_vl_khmer_lora
```

### 5. Run

```bash
# Fine-tune DeepSeek-OCR-2
python "src/fine-tuning/Deepseek_OCR2(3B).py"

# Fine-tune PaddleOCR-VL
python "src/fine-tuning/Paddle_OCR(1B).py"

# Evaluate DeepSeek on one test row
python "src/inference/DeepSeek_OCR2(3B).py" --row 0 --split test_setA

# Evaluate Paddle on one test row
python "src/inference/Paddle_OCR(1B).py" --row 0 --split test_setB
```

## Configuration

All model settings live in `configs/lora.yaml`, split into `deepseek:` and
`paddle:` sections. Relative paths are resolved against `src/`.

Shared training settings: BF16, LoRA dropout `0`, AdamW 8-bit, weight decay
`0.001`, linear scheduler, `5` warmup steps, batch `2` with gradient
accumulation `5`, max sequence length `4096`, and `2` epochs over the 61,750
training pages (`12,350` steps).

| Parameter         | DeepSeek-OCR-2       | PaddleOCR-VL                                    |
| ----------------- | -------------------- | ----------------------------------------------- |
| LoRA rank / alpha | 16 / 16              | 64 / 64                                         |
| Learning rate     | 1e-4                 | 5e-5                                            |
| Image resolution  | 768 min / 1024 base  | dynamic                                         |
| LoRA targets      | 7 projection modules | same + `out_proj, fc1, fc2, linear_1, linear_2` |

## Training

Run from the repository root:

```bash
python "src/fine-tuning/Deepseek_OCR2(3B).py"
python "src/fine-tuning/Paddle_OCR(1B).py"
```

Training writes:

- Adapter: `src/models/deepseek_ocr2_khmer_lora/` or
  `src/models/paddleocr_vl_khmer_lora/`
- Logs: `deepseek_training_summary.log`, `paddle_training_summary.log`
- Metrics: `deepseek_training_results.json`, `paddle_training_results.json`
- Loss curves and trainer state in the adapter directory

## Inference and evaluation

Run from the repository root:

```bash
# DeepSeek: one row, both baseline and LoRA
python "src/inference/DeepSeek_OCR2(3B).py" --row 0 --split test_setA

# DeepSeek: LoRA only
python "src/inference/DeepSeek_OCR2(3B).py" --row 10 --model lora --split test_setB

# Paddle: both models on every row of a split
python "src/inference/Paddle_OCR(1B).py" --model both --split test_setA
```

CLI options:

| Option                         | Description                                  |
| ------------------------------ | -------------------------------------------- |
| `--row N`                      | Test row index. Omit to run the whole split. |
| `--model baseline\|lora\|both` | Model(s) to run. Default: `both`.            |
| `--split test_setA\|test_setB` | Evaluation split. Default from config.       |
| `--output-dir PATH`            | Output directory.                            |
| `--config PATH`                | Alternative YAML config.                     |

Outputs are written to `src/output/deepseek/` and `src/output/paddle/`:

- `baseline_predictions.jsonl` / `lora_predictions.jsonl`
- `report.json` with predictions and micro-averaged CER/WER

## Metrics

- Ground truth comes from the selected test split.
- The same text cleaning is applied to ground truth and prediction.
- CER uses Khmer-normalized character Levenshtein distance.
- WER uses Khmer word segmentation, then normalized word-level Levenshtein
  distance.
- Only Khmer-normalized CER/WER are reported.
- Corpus scores are always micro-averages.

## Dataset

`CADT-IDRI/KhmerEduDoc` (`default` subset):

- `train`: 61,750 pages
- `validation`: 3,250 pages
- `test_setA`: 60 pages
- `test_setB`: 100 pages

Columns: `image`, `text`, `book_id`, `page`, `label_source`, `content_type`.

## Contribution

- [Kimleangchhay SONG](https://github.com/leangchhay1523)
