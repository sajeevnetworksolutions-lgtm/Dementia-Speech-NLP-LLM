"""
Comprehensive Synopsis-to-Code Double-Testing Validation Suite.
Cross-verifies every specification, equation, objective, and benchmark from the M.Tech Dissertation Synopsis.
"""
import unittest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json

from config import (
    AUDIO_CONFIG, WAVLM_CONFIG, LLM_CONFIG, CROSS_ATTN_CONFIG,
    MULTI_TASK_CONFIG, MMSE_THRESHOLDS, SAMPLES_DIR, MODELS_DIR
)
from src.audio.vad_preprocessor import AudioVADPreprocessor
from src.audio.paralinguistics import ParalinguisticExtractor
from src.audio.acoustic_encoder import WavLMAcousticEncoder
from src.audio.asr_transcriber import ClinicalASRTranscriber
from src.nlp.lexical_analyzer import LexicalAnalyzer
from src.nlp.llama_reasoner import ClinicalLlamaReasoner
from src.models.cross_attention import TemporalCrossModalAttention
from src.models.multitask_model import DementiaScreeningNet
from src.models.loss import MultiTaskCognitiveLoss
from src.data.synthetic_generator import ClinicalSyntheticGenerator, SAMPLE_PROFILES
from src.data.dataset_loader import DementiaDataset, create_dataloaders
from src.evaluation.metrics import compute_classification_metrics, compute_regression_metrics
from src.evaluation.benchmark_eval import BenchmarkEvaluator, BENCHMARK_RESULTS
from src.evaluation.pdf_report_generator import ClinicalReportGenerator
from src.pipeline import DementiaScreeningPipeline


class TestSynopsisCompliance(unittest.TestCase):
    """
    Direct verification of the 12 Synopsis Sections against implementation.
    """

    @classmethod
    def setUpClass(cls):
        cls.sr = AUDIO_CONFIG["sample_rate"]
        cls.generator = ClinicalSyntheticGenerator()
        cls.samples = cls.generator.generate_all_samples()
        cls.pipeline = DementiaScreeningPipeline()

    def test_section_01_abstract_and_audio_specs(self):
        """Verify 16kHz audio preprocessing, 2-minute screening window, and RMS normalization."""
        self.assertEqual(self.sr, 16000, "Audio sampling rate must be strictly 16,000 Hz.")
        self.assertEqual(AUDIO_CONFIG["max_duration_sec"], 120.0, "Screening audio window must support 2-minute recording.")

        vad = AudioVADPreprocessor()
        raw_sig = np.random.randn(16000 * 3).astype(np.float32) * 5.0
        norm_sig, sr = vad.load_and_normalize(raw_sig, 16000)
        self.assertTrue(np.max(np.abs(norm_sig)) <= 1.0, "Audio must be normalized within [-1.0, 1.0].")

    def test_section_02_paralinguistic_biomarkers(self):
        """Verify pitch (F0), jitter, shimmer, HNR, and micro-pause extraction."""
        para = ParalinguisticExtractor(sample_rate=16000)
        t = np.linspace(0, 2.0, 16000 * 2, endpoint=False)
        f0_inst = 180.0 + np.sin(2 * np.pi * 5 * t) * 5.0
        phase = 2 * np.pi * np.cumsum(f0_inst) / 16000
        test_speech = (0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase)).astype(np.float32)

        feats = para.extract_features(test_speech)
        self.assertIn("mean_f0_hz", feats)
        self.assertIn("jitter_percent", feats)
        self.assertIn("shimmer_percent", feats)
        self.assertIn("hnr_db", feats)
        self.assertIn("articulation_rate_syl_sec", feats)
        self.assertGreater(feats["mean_f0_hz"], 100.0)

    def test_section_03_wavlm_acoustic_encoder(self):
        """Verify WavLM 1024-dimensional acoustic embedding output."""
        encoder = WavLMAcousticEncoder()
        self.assertEqual(encoder.embedding_dim, 1024, "WavLM embedding dimension must be exactly 1024-d.")

        dummy_audio = torch.randn(2, 16000 * 2)
        out = encoder(dummy_audio)
        self.assertIn("frame_embeddings", out)
        self.assertIn("pooled_embedding", out)
        self.assertEqual(out["frame_embeddings"].shape[-1], 1024, "Acoustic frame features must be 1024-dimensional.")
        self.assertEqual(out["pooled_embedding"].shape, (2, 1024))

    def test_section_04_linguistic_lexical_analysis(self):
        """Verify TTR, Brunet's W, Honore's R, and pronoun-to-noun anomic ratios on clinical profiles."""
        lex = LexicalAnalyzer()
        # Full Cookie Theft description monologues
        hc_transcript = SAMPLE_PROFILES[0]["transcript"]
        ad_transcript = SAMPLE_PROFILES[3]["transcript"]

        h_feats = lex.analyze_transcript(hc_transcript)
        d_feats = lex.analyze_transcript(ad_transcript)

        self.assertGreater(h_feats["type_token_ratio"], d_feats["type_token_ratio"], "Control TTR must be higher than Dementia TTR.")
        self.assertGreater(d_feats["hesitation_count"], h_feats["hesitation_count"], "Dementia hesitation count must exceed Control.")
        self.assertGreater(d_feats["pronoun_to_noun_ratio"], h_feats["pronoun_to_noun_ratio"], "Dementia pronoun substitution ratio must exceed Control.")

    def test_section_05_llama3_qlora_and_cot(self):
        """Verify LLaMA-3 QLoRA configuration (r=16, alpha=32) and CoT rationale generator."""
        self.assertEqual(LLM_CONFIG["lora_r"], 16, "LoRA rank must be 16.")
        self.assertEqual(LLM_CONFIG["lora_alpha"], 32, "LoRA alpha must be 32.")
        self.assertEqual(LLM_CONFIG["projected_dim"], 768, "Text projection dimension must be 768-d.")

        reasoner = ClinicalLlamaReasoner()
        cot = reasoner.generate_clinical_rationale(
            transcript="The lady is washing plates...",
            risk_probability=0.62,
            predicted_mmse=18.5,
            acoustic_biomarkers={"pause_ratio": 0.55, "pause_rate_per_min": 24.0, "std_f0_hz": 11.2, "jitter_percent": 3.4},
            lexical_biomarkers={"type_token_ratio": 0.44, "pronoun_to_noun_ratio": 3.2, "hesitation_count": 8, "brunet_w": 8.5}
        )
        self.assertIn("clinical_stage", cot)
        self.assertIn("multimodal_synthesis", cot)
        self.assertIn("acoustic_analysis", cot)
        self.assertIn("linguistic_analysis", cot)
        self.assertIn("clinical_recommendations", cot)
        self.assertEqual(cot["clinical_stage"], "Mild Alzheimer's Dementia")

    def test_section_06_temporal_cross_modal_attention(self):
        """Verify Cross-Modal Attention: Audio (1024d) + Text (768d) -> Fused Cognitive Vector (512d)."""
        cross_attn = TemporalCrossModalAttention()
        self.assertEqual(cross_attn.audio_dim, 1024)
        self.assertEqual(cross_attn.text_dim, 768)
        self.assertEqual(cross_attn.fused_dim, 512)

        batch_size = 3
        audio_feats = torch.randn(batch_size, 128, 1024)
        text_feats = torch.randn(batch_size, 32, 768)

        fused, weights = cross_attn(audio_feats, text_feats)
        self.assertEqual(fused.shape, (batch_size, 512), "Fused representation must be [Batch, 512].")
        self.assertEqual(weights.shape, (batch_size, 32, 128), "Attention weights must map text tokens to audio frames.")

    def test_section_07_multitask_loss_and_scoring_head(self):
        """Verify Multi-Task loss (BCE + Smooth L1/Huber) and dual prediction heads."""
        self.assertEqual(MULTI_TASK_CONFIG["cls_weight"], 1.0)
        self.assertEqual(MULTI_TASK_CONFIG["reg_weight"], 0.5)

        loss_fn = MultiTaskCognitiveLoss()
        logits = torch.tensor([[1.2], [-0.8]])
        mmse_pred = torch.tensor([[18.5], [27.0]])
        cls_target = torch.tensor([[1.0], [0.0]])
        mmse_target = torch.tensor([[17.0], [28.0]])

        loss_dict = loss_fn(logits, mmse_pred, cls_target, mmse_target)
        self.assertIn("total_loss", loss_dict)
        self.assertIn("cls_loss", loss_dict)
        self.assertIn("reg_loss", loss_dict)
        self.assertGreater(loss_dict["total_loss"].item(), 0.0)

    def test_section_08_comparative_benchmark_targets(self):
        """Verify baseline benchmark targets from Synopsis Section 8."""
        evaluator = BenchmarkEvaluator()
        benchmarks = evaluator.benchmarks
        self.assertEqual(len(benchmarks), 4, "Must contain all 4 synopsis benchmark configurations.")

        proposed = benchmarks[3]
        self.assertEqual(proposed["model"], "Proposed WavLM + LLaMA-3 LoRA (Ours)")
        self.assertAlmostEqual(proposed["accuracy"], 0.926, places=3, msg="Target accuracy must be 92.6%.")
        self.assertAlmostEqual(proposed["recall_f1"], 0.924, places=3, msg="Target F1 must be 0.924.")
        self.assertAlmostEqual(proposed["mmse_rmse"], 2.18, places=2, msg="Target MMSE RMSE must be 2.18 pts.")

    def test_section_09_mmse_clinical_stratification(self):
        """Verify standard MMSE score cutoff stratification bands."""
        self.assertEqual(MMSE_THRESHOLDS["Normal"], (26.0, 30.0))
        self.assertEqual(MMSE_THRESHOLDS["Mild Cognitive Impairment (MCI)"], (21.0, 25.9))
        self.assertEqual(MMSE_THRESHOLDS["Mild Dementia"], (15.0, 20.9))
        self.assertEqual(MMSE_THRESHOLDS["Moderate to Severe Dementia"], (0.0, 14.9))

    def test_section_10_end_to_end_pipeline_screening(self):
        """Verify full pipeline on healthy vs dementia sample audio."""
        for sample in self.samples:
            res = self.pipeline.process_screening(
                audio_input=sample["audio_path"],
                transcript_input=sample["transcript"]
            )
            self.assertEqual(res["status"], "success")
            self.assertIn("dementia_risk_probability", res)
            self.assertIn("predicted_mmse_score", res)
            self.assertIn("clinical_chain_of_thought", res)
            self.assertIn("html_report", res)

            self.assertGreaterEqual(res["dementia_risk_probability"], 0.0)
            self.assertLessEqual(res["dementia_risk_probability"], 1.0)
            self.assertGreaterEqual(res["predicted_mmse_score"], 0.0)
            self.assertLessEqual(res["predicted_mmse_score"], 30.0)


if __name__ == "__main__":
    unittest.main()
