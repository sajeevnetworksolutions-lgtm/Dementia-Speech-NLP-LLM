"""
Interactive CLI Demo for Early Dementia & Cognitive Decline Screening.
Runs end-to-end multi-modal screening directly from the command line.
"""
import argparse
import json
from pathlib import Path
from typing import Optional
import sys
import os

# Set UTF-8 encoding for Windows terminal
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import SAMPLES_DIR, MODELS_DIR
from src.pipeline import DementiaScreeningPipeline


def run_cli_demo(audio_path: Optional[str] = None, transcript: Optional[str] = None):
    print("\n" + "=" * 75)
    print("  [AI SCREENING] EARLY DEMENTIA & COGNITIVE DECLINE SCREENING - CLI DEMO")
    print("=" * 75)

    ckpt = MODELS_DIR / "best_dementia_model.pt"
    ckpt_path = str(ckpt) if ckpt.exists() else None
    pipeline = DementiaScreeningPipeline(model_checkpoint_path=ckpt_path)

    if not audio_path:
        manifest_path = SAMPLES_DIR / "samples_manifest.json"
        if not manifest_path.exists():
            from src.data.synthetic_generator import ClinicalSyntheticGenerator
            ClinicalSyntheticGenerator().generate_all_samples()

        with open(manifest_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        print("\nAvailable Preloaded Test Profiles:")
        for i, s in enumerate(samples):
            print(f" [{i+1}] {s['id']} ({s['category']} | Age {s['age']} | True MMSE: {s['mmse']})")

        idx = 2
        selected = samples[idx]

        audio_path = selected["audio_path"]
        transcript = selected["transcript"]
        print(f"\n[Selected Profile]: {selected['id']} ({selected['category']})")

    print(f"[Audio Path]: {audio_path}")
    print(f"[Processing Audio & Transcripts]...")

    res = pipeline.process_screening(audio_input=audio_path, transcript_input=transcript)

    print("\n" + "-" * 75)
    print(f"  DIAGNOSTIC PREDICTION:")
    print(f"  • Cognitive Risk %:      {res['dementia_risk_percentage']} (Severity: {res['severity_level']})")
    print(f"  • Predicted MMSE Score:  {res['predicted_mmse_score']:.1f} / 30.0 ({res['clinical_stage']})")
    print(f"  • Audio Duration:        {res['audio_duration_sec']:.1f} seconds")
    print(f"  • Silence / Pause Ratio: {res['vad_summary']['pause_ratio']*100:.1f}% ({res['vad_summary']['pause_count']} micro-pauses)")
    print(f"  • Mean Pitch (F0):       {res['paralinguistic_summary']['mean_f0_hz']:.1f} Hz (Std: {res['paralinguistic_summary']['std_f0_hz']:.1f} Hz)")
    print(f"  • Type-Token Ratio:      {res['lexical_summary']['type_token_ratio']:.3f} (Richness: Brunet's W {res['lexical_summary']['brunet_w']:.2f})")
    print(f"  • Pronoun-to-Noun Ratio: {res['lexical_summary']['pronoun_to_noun_ratio']:.2f}")
    print("-" * 75)
    print(f"\nCLINICIAN CHAIN-OF-THOUGHT (CoT) REASONING:")
    print(f"  {res['clinical_chain_of_thought']['multimodal_synthesis']}")
    print(f"\n  Acoustic Observations:   {res['clinical_chain_of_thought']['acoustic_analysis']}")
    print(f"  Linguistic Observations: {res['clinical_chain_of_thought']['linguistic_analysis']}")
    print(f"\n  Clinical Recommendations:")
    for rec in res['clinical_chain_of_thought']['clinical_recommendations']:
        print(f"    - {rec}")
    print("\n" + "=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default=None, help="Path to WAV audio file")
    parser.add_argument("--transcript", type=str, default=None, help="Transcript text")
    args = parser.parse_args()

    run_cli_demo(audio_path=args.audio, transcript=args.transcript)
