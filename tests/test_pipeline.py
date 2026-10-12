"""
Unit Tests for Early Dementia Screening Pipeline Modules.
"""
import unittest
import numpy as np
import torch
from pathlib import Path

from config import AUDIO_CONFIG, SAMPLES_DIR
from src.audio.vad_preprocessor import AudioVADPreprocessor
from src.audio.paralinguistics import ParalinguisticExtractor
from src.nlp.lexical_analyzer import LexicalAnalyzer
from src.nlp.llama_reasoner import ClinicalLlamaReasoner
from src.models.cross_attention import TemporalCrossModalAttention
from src.models.multitask_model import DementiaScreeningNet
from src.pipeline import DementiaScreeningPipeline
from src.data.synthetic_generator import ClinicalSyntheticGenerator


class TestDementiaScreeningPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sr = 16000
        cls.duration = 4.0
        # Generate clean synthetic test tone
        t = np.linspace(0, cls.duration, int(cls.duration * cls.sr), endpoint=False)
        cls.test_audio = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        cls.test_transcript = "The boy is taking cookies from the jar and the mother is washing dishes."

    def test_01_vad_preprocessor(self):
        vad = AudioVADPreprocessor()
        norm_audio, sr = vad.load_and_normalize(self.test_audio, self.sr)
        self.assertEqual(sr, self.sr)
        self.assertTrue(np.max(np.abs(norm_audio)) <= 1.0)

        vad_info = vad.detect_vad_segments(norm_audio, sr)
        self.assertIn("total_duration", vad_info)
        self.assertIn("speech_ratio", vad_info)
        self.assertIn("pause_count", vad_info)

    def test_02_paralinguistics(self):
        para = ParalinguisticExtractor(sample_rate=self.sr)
        feats = para.extract_features(self.test_audio)
        self.assertIn("mean_f0_hz", feats)
        self.assertIn("jitter_percent", feats)
        self.assertIn("shimmer_percent", feats)
        self.assertIn("hnr_db", feats)
        self.assertGreater(feats["mean_f0_hz"], 50.0)

    def test_03_lexical_analyzer(self):
        lex = LexicalAnalyzer()
        feats = lex.analyze_transcript(self.test_transcript)
        self.assertIn("type_token_ratio", feats)
        self.assertIn("brunet_w", feats)
        self.assertIn("honore_r", feats)
        self.assertIn("hesitation_count", feats)
        self.assertGreater(feats["type_token_ratio"], 0.0)

    def test_04_cross_modal_attention(self):
        attn = TemporalCrossModalAttention()
        audio_feats = torch.randn(2, 50, 1024)
        text_feats = torch.randn(2, 20, 768)
        fused, weights = attn(audio_feats, text_feats)
        self.assertEqual(fused.shape, (2, 512))
        self.assertEqual(weights.shape, (2, 20, 50))

    def test_05_multitask_network(self):
        net = DementiaScreeningNet()
        audio_tensors = torch.randn(2, 16000 * 2)
        text_emb = torch.randn(2, 15, 768)
        bio = torch.randn(2, 16)
        out = net(audio_tensors, text_emb, bio)
        self.assertIn("cls_logits", out)
        self.assertIn("dementia_prob", out)
        self.assertIn("mmse_score", out)
        self.assertEqual(out["dementia_prob"].shape, (2, 1))
        self.assertEqual(out["mmse_score"].shape, (2, 1))

    def test_06_end_to_end_pipeline(self):
        pipe = DementiaScreeningPipeline()
        res = pipe.process_screening(self.test_audio, self.test_transcript, self.sr)
        self.assertEqual(res["status"], "success")
        self.assertIn("dementia_risk_percentage", res)
        self.assertIn("predicted_mmse_score", res)
        self.assertIn("clinical_chain_of_thought", res)
        self.assertGreaterEqual(res["predicted_mmse_score"], 0.0)
        self.assertLessEqual(res["predicted_mmse_score"], 30.0)


if __name__ == "__main__":
    unittest.main()
