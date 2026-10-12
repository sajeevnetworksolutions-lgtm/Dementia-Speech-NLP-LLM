"""
End-to-End Dementia & Cognitive Decline Screening Pipeline.
Unifies Audio VAD, Paralinguistics, ASR Transcription, Lexical NLP, WavLM-LLaMA Cross-Attention, and CoT Reasoning.
"""
import torch
import numpy as np
import scipy.io.wavfile
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

from .audio.vad_preprocessor import AudioVADPreprocessor
from .audio.paralinguistics import ParalinguisticExtractor
from .audio.asr_transcriber import ClinicalASRTranscriber
from .nlp.lexical_analyzer import LexicalAnalyzer
from .nlp.llama_reasoner import ClinicalLlamaReasoner
from .models.multitask_model import DementiaScreeningNet
from .evaluation.pdf_report_generator import ClinicalReportGenerator

try:
    from config import AUDIO_CONFIG, WAVLM_CONFIG, LLM_CONFIG, CROSS_ATTN_CONFIG, MMSE_THRESHOLDS
except ImportError:
    from ..config import AUDIO_CONFIG, WAVLM_CONFIG, LLM_CONFIG, CROSS_ATTN_CONFIG, MMSE_THRESHOLDS


class DementiaScreeningPipeline:
    def __init__(self, model_checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.vad_processor = AudioVADPreprocessor()
        self.paralinguistics = ParalinguisticExtractor(sample_rate=16000)
        self.asr_transcriber = ClinicalASRTranscriber()
        self.lexical_analyzer = LexicalAnalyzer()
        self.llama_reasoner = ClinicalLlamaReasoner()
        self.report_generator = ClinicalReportGenerator()

        self.model = DementiaScreeningNet(
            audio_config=WAVLM_CONFIG,
            llm_config=LLM_CONFIG,
            cross_attn_config=CROSS_ATTN_CONFIG,
            num_biomarkers=16
        ).to(self.device)

        if model_checkpoint_path and Path(model_checkpoint_path).exists():
            state = torch.load(model_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
            print(f"[Pipeline] Loaded model checkpoint from: {model_checkpoint_path}")

        self.model.eval()

    def process_screening(
        self,
        audio_input: Union[str, Path, np.ndarray],
        transcript_input: Optional[str] = None,
        sample_rate: int = 16000
    ) -> Dict:
        # 1. Load and normalize audio
        if isinstance(audio_input, (str, Path)):
            sr, raw_audio = scipy.io.wavfile.read(str(audio_input))
            audio, sr = self.vad_processor.load_and_normalize(raw_audio, sr)
        else:
            audio, sr = self.vad_processor.load_and_normalize(audio_input, sample_rate)

        # 2. VAD and Silence / Pause Analysis
        vad_info = self.vad_processor.detect_vad_segments(audio, sr)

        # 3. Acoustic Paralinguistic Biomarkers
        acoustic_biomarkers = self.paralinguistics.extract_features(audio, vad_info)

        # 4. Transcription (Automatic if None provided)
        if transcript_input and transcript_input.strip():
            transcript = transcript_input.strip()
            asr_source = "User / Ground Truth"
        else:
            asr_res = self.asr_transcriber.transcribe(audio, sample_rate=sr, vad_info=vad_info)
            transcript = asr_res["text"]
            asr_source = asr_res.get("source", "ASR")

        # 5. Lexical Analysis
        lexical_biomarkers = self.lexical_analyzer.analyze_transcript(transcript)

        # 6. Extract 16-d Biomarker Vector
        biomarker_vec = [
            vad_info.get("pause_ratio", 0.0),
            vad_info.get("pause_rate_per_min", 0.0) / 40.0,
            vad_info.get("mean_pause_duration", 0.0) / 2.0,
            vad_info.get("speech_ratio", 0.0),
            acoustic_biomarkers.get("mean_f0_hz", 150.0) / 300.0,
            acoustic_biomarkers.get("std_f0_hz", 15.0) / 50.0,
            acoustic_biomarkers.get("jitter_percent", 1.0) / 5.0,
            acoustic_biomarkers.get("shimmer_percent", 3.0) / 10.0,
            acoustic_biomarkers.get("hnr_db", 15.0) / 30.0,
            acoustic_biomarkers.get("articulation_rate_syl_sec", 3.0) / 8.0,
            lexical_biomarkers.get("type_token_ratio", 0.5),
            lexical_biomarkers.get("brunet_w", 10.0) / 20.0,
            lexical_biomarkers.get("honore_r", 100.0) / 200.0,
            lexical_biomarkers.get("yules_k", 50.0) / 200.0,
            lexical_biomarkers.get("hesitation_rate", 0.05) / 0.3,
            lexical_biomarkers.get("pronoun_to_noun_ratio", 1.0) / 5.0,
        ]
        bio_tensor = torch.tensor(biomarker_vec, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 7. Extract Text Embeddings
        text_emb, text_mask = self.llama_reasoner.extract_text_embeddings(transcript)
        text_emb = text_emb.to(self.device)
        text_mask = text_mask.to(self.device)

        max_samples = min(len(audio), int(AUDIO_CONFIG["max_duration_sec"] * sr))
        audio_tensor = torch.from_numpy(audio[:max_samples]).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                audio_tensors=audio_tensor,
                text_features=text_emb,
                biomarkers=bio_tensor,
                text_mask=text_mask
            )

        # 8. Calibration
        pause_penalty = min(0.35, max(0.0, (vad_info["pause_ratio"] - 0.25) * 0.8))
        hesitation_penalty = min(0.25, max(0.0, (lexical_biomarkers["hesitation_rate"] - 0.04) * 1.5))
        ttr_penalty = min(0.20, max(0.0, (0.55 - lexical_biomarkers["type_token_ratio"]) * 0.6))

        raw_prob = float(outputs["dementia_prob"].cpu().numpy().item())
        calibrated_prob = np.clip(raw_prob * 0.4 + (pause_penalty + hesitation_penalty + ttr_penalty) * 0.6, 0.02, 0.98)

        model_mmse = float(outputs["mmse_score"].cpu().numpy().item())
        calibrated_mmse = float(np.clip(30.0 - (calibrated_prob * 18.0) + (model_mmse - 15.0) * 0.2, 5.0, 30.0))

        # 9. Clinical Chain-of-Thought Rationale
        clinical_report = self.llama_reasoner.generate_clinical_rationale(
            transcript=transcript,
            risk_probability=calibrated_prob,
            predicted_mmse=calibrated_mmse,
            acoustic_biomarkers={**vad_info, **acoustic_biomarkers},
            lexical_biomarkers=lexical_biomarkers
        )

        result_dict = {
            "status": "success",
            "audio_duration_sec": len(audio) / sr,
            "dementia_risk_probability": float(calibrated_prob),
            "dementia_risk_percentage": f"{calibrated_prob * 100.0:.1f}%",
            "predicted_mmse_score": float(calibrated_mmse),
            "clinical_stage": clinical_report["clinical_stage"],
            "severity_level": clinical_report["severity"],
            "vad_summary": {
                "total_speech_sec": float(vad_info["total_speech_time"]),
                "total_pause_sec": float(vad_info["total_pause_time"]),
                "pause_ratio": float(vad_info["pause_ratio"]),
                "pause_count": int(vad_info["pause_count"]),
                "mean_pause_sec": float(vad_info["mean_pause_duration"]),
            },
            "paralinguistic_summary": acoustic_biomarkers,
            "lexical_summary": lexical_biomarkers,
            "clinical_chain_of_thought": clinical_report,
            "raw_transcript": transcript,
            "transcript_source": asr_source,
        }

        # Generate HTML report
        result_dict["html_report"] = self.report_generator.generate_html_report(result_dict)
        return result_dict
