"""
Model Training Script for Multi-Task Dementia & Cognitive Decline Screening.
Trains the WavLM + LLaMA-3 LoRA Cross-Modal Attention Pipeline with Multi-Task Loss.
"""
import torch
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import PROJECT_ROOT, MODELS_DIR, SAMPLES_DIR, TRAIN_CONFIG
from src.data.synthetic_generator import ClinicalSyntheticGenerator
from src.data.dataset_loader import DementiaDataset, create_dataloaders
from src.models.multitask_model import DementiaScreeningNet
from src.models.loss import MultiTaskCognitiveLoss
from src.evaluation.metrics import compute_classification_metrics, compute_regression_metrics


def train_model(
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 3e-4,
    device: str = "cpu"
):
    print("=" * 70)
    print("  EARLY DEMENTIA & COGNITIVE DECLINE SCREENING AI - TRAINING PIPELINE")
    print("=" * 70)

    # 1. Ensure sample dataset exists
    gen = ClinicalSyntheticGenerator(output_dir=SAMPLES_DIR)
    samples = gen.generate_all_samples()
    print(f"[Data] Loaded {len(samples)} clinical speech samples.")

    # Create augmented manifest
    augmented_samples = []
    for _ in range(4):
        for s in samples:
            aug = dict(s)
            aug["mmse"] = float(np.clip(s["mmse"] + np.random.normal(0, 0.3), 0, 30))
            augmented_samples.append(aug)

    train_loader, val_loader, test_loader = create_dataloaders(augmented_samples, batch_size=batch_size)
    print(f"[Data] Split into Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # 2. Instantiate Model and Loss
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"[Device] Running on: {device_obj}")

    model = DementiaScreeningNet().to(device_obj)
    criterion = MultiTaskCognitiveLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_ckpt_path = MODELS_DIR / "best_dementia_model.pt"

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            audio = batch["audio"].to(device_obj)
            text_emb = batch["text_emb"].to(device_obj)
            text_mask = batch["text_mask"].to(device_obj)
            biomarkers = batch["biomarkers"].to(device_obj)
            labels = batch["label"].to(device_obj)
            mmse_targets = batch["mmse"].to(device_obj)

            optimizer.zero_grad()
            outputs = model(
                audio_tensors=audio,
                text_features=text_emb,
                biomarkers=biomarkers,
                text_mask=text_mask
            )

            loss_dict = criterion(
                cls_logits=outputs["cls_logits"],
                reg_preds=outputs["mmse_score"],
                cls_targets=labels,
                reg_targets=mmse_targets
            )

            loss = loss_dict["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_probs = []
        all_mmse_true = []
        all_mmse_pred = []

        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device_obj)
                text_emb = batch["text_emb"].to(device_obj)
                text_mask = batch["text_mask"].to(device_obj)
                biomarkers = batch["biomarkers"].to(device_obj)
                labels = batch["label"].to(device_obj)
                mmse_targets = batch["mmse"].to(device_obj)

                outputs = model(
                    audio_tensors=audio,
                    text_features=text_emb,
                    biomarkers=biomarkers,
                    text_mask=text_mask
                )

                loss_dict = criterion(
                    cls_logits=outputs["cls_logits"],
                    reg_preds=outputs["mmse_score"],
                    cls_targets=labels,
                    reg_targets=mmse_targets
                )
                val_loss += loss_dict["total_loss"].item()

                all_labels.extend(labels.cpu().numpy().flatten())
                all_probs.extend(outputs["dementia_prob"].cpu().numpy().flatten())
                all_mmse_true.extend(mmse_targets.cpu().numpy().flatten())
                all_mmse_pred.extend(outputs["mmse_score"].cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_loader)
        cls_metrics = compute_classification_metrics(np.array(all_labels), np.array(all_probs))
        reg_metrics = compute_regression_metrics(np.array(all_mmse_true), np.array(all_mmse_pred))

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {cls_metrics['accuracy']*100:.1f}% | F1: {cls_metrics['f1_score']:.3f} | MMSE RMSE: {reg_metrics['rmse']:.2f} pts")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), str(best_ckpt_path))
            print(f"  --> Saved new best checkpoint to {best_ckpt_path.name}")

    print("\n[Complete] Training finished successfully.")
    return str(best_ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
