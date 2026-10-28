import json
import time
from datetime import datetime
from pathlib import Path

import torch


def resolve_precision(training_args, bf16_supported):
    """Resolve fp16/bf16='auto' into concrete booleans."""
    training_args = dict(training_args)
    if training_args.get("fp16") == "auto":
        training_args["fp16"] = not bf16_supported
    if training_args.get("bf16") == "auto":
        training_args["bf16"] = bf16_supported
    return training_args


def write_log(text, path):
    """Print text and append it (with a timestamp) to a .log file."""
    print(text)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}]\n{text}\n\n")


def plot_loss_curves(log_history, output_dir, title):
    """Save combined, train-only, and eval-only loss curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_steps = [entry["step"] for entry in log_history if "loss" in entry]
        train_losses = [entry["loss"] for entry in log_history if "loss" in entry]
        eval_steps = [entry["step"] for entry in log_history if "eval_loss" in entry]
        eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]

        if not train_losses and not eval_losses:
            return

        plt.figure(figsize=(10, 6))
        if train_losses:
            plt.plot(train_steps, train_losses, label="Train loss", linewidth=1.5)
        if eval_losses:
            plt.plot(eval_steps, eval_losses, label="Eval loss", linewidth=2, marker="o")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"{title} Loss Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "training_loss_curve.png", dpi=150)
        plt.close()

        if train_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(train_steps, train_losses, label="Train loss", linewidth=1.5, color="tab:blue")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title(f"{title} - Train Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "train_loss_curve.png", dpi=150)
            plt.close()

        if eval_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(eval_steps, eval_losses, label="Eval loss", linewidth=2, marker="o", color="tab:orange")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title(f"{title} - Eval Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "eval_loss_curve.png", dpi=150)
            plt.close()
    except Exception as exc:
        print(f"Warning: could not save loss curves: {type(exc).__name__}: {exc}")


def save_training_artifacts(
    trainer,
    trainer_stats,
    start_time,
    start_gpu_memory,
    max_memory,
    log_file="training_summary.log",
    json_file="training_results.json",
    loss_curve_dir=None,
    title="Fine-tuning",
):
    """Write the shared training log, metrics JSON, loss curves, and trainer state."""
    duration_seconds = int(time.time() - start_time)
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, _ = divmod(remainder, 60)

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    summary = "\n".join([
        f"Total training time: {hours}h {minutes}m",
        f"{trainer_stats.metrics['train_runtime']} seconds used for training.",
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.",
        f"Peak reserved memory = {used_memory} GB.",
        f"Peak reserved memory for training = {used_memory_for_lora} GB.",
        f"Peak reserved memory % of max memory = {used_percentage} %.",
        f"Peak reserved memory for training % of max memory = {lora_percentage} %.",
    ])
    write_log(summary, log_file)

    final_eval = trainer.evaluate()
    log_history = trainer.state.log_history
    train_logs = [entry for entry in log_history if "loss" in entry]

    best_ckpt = trainer.state.best_model_checkpoint
    best_step = int(best_ckpt.split("-")[-1]) if best_ckpt else trainer.state.global_step
    train_loss_at_best_step = next(
        (entry["loss"] for entry in reversed(train_logs) if entry["step"] <= best_step),
        None,
    )

    results = {
        "final_model": {
            "best_checkpoint": best_ckpt,
            "best_step": best_step,
            "validation_loss": final_eval["eval_loss"],
            "train_loss_at_best_step": train_loss_at_best_step,
        },
        "training_run": {
            "average_train_loss": trainer_stats.metrics["train_loss"],
            "last_logged_train_loss": train_logs[-1]["loss"] if train_logs else None,
            "total_steps": trainer.state.global_step,
            "epochs_completed": trainer.state.epoch,
            "train_runtime_seconds": trainer_stats.metrics["train_runtime"],
            "peak_reserved_memory_gb": used_memory,
        },
        "log_history": log_history,
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    write_log(
        f"Final model (step {best_step}): "
        f"train loss = {train_loss_at_best_step}, validation loss = {final_eval['eval_loss']}",
        log_file,
    )

    if loss_curve_dir is not None:
        plot_loss_curves(log_history, loss_curve_dir, title)

    trainer.save_state()
    return results
