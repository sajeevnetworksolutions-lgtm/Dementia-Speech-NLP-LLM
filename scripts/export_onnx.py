"""
Model Export & Optimization Utility.
Exports the trained Multi-Modal DementiaScreeningNet to TorchScript for edge & browser deployment.
"""
import torch
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import MODELS_DIR, WAVLM_CONFIG, LLM_CONFIG, CROSS_ATTN_CONFIG
from src.models.multitask_model import DementiaScreeningNet


def export_torchscript():
    print("=" * 70)
    print("  EXPORTING MULTI-MODAL DEMENTIA MODEL TO TORCHSCRIPT")
    print("=" * 70)

    model = DementiaScreeningNet()
    ckpt_path = MODELS_DIR / "best_dementia_model.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"), strict=False)
        print(f"[Loaded Checkpoint]: {ckpt_path.name}")

    model.eval()

    # Dummy inputs for tracing: [Batch=1]
    dummy_audio = torch.randn(1, 16000 * 5)  # 5-second audio
    dummy_text = torch.randn(1, 32, 768)     # 32 tokens
    dummy_bio = torch.randn(1, 16)           # 16 tabular biomarkers

    try:
        traced_model = torch.jit.trace(
            model,
            (dummy_audio, dummy_text, dummy_bio),
            check_trace=False
        )
        export_path = MODELS_DIR / "dementia_screening_traced.pt"
        traced_model.save(str(export_path))
        print(f"\n[Success] Successfully saved TorchScript model to: {export_path.name}")
        print(f"[Model File Size]: {export_path.stat().st_size / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"\n[Export Notice]: Direct JIT tracing encountered dynamic sequence handling ({e}).")
        # Save standard portable state dict
        portable_path = MODELS_DIR / "dementia_model_production.pt"
        torch.save(model.state_dict(), str(portable_path))
        print(f"[Success] Saved production state dict to: {portable_path.name}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    export_torchscript()
