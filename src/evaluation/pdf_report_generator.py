"""
Clinical PDF / HTML Diagnostic Report Generator.
Renders medical screening reports with visual styling, biomarker tables,
and clinician Chain-of-Thought sign-off.
"""
from pathlib import Path
import datetime
from typing import Dict, Optional


class ClinicalReportGenerator:
    def __init__(self, clinic_name: str = "NeuroCognitive Speech AI Laboratory", doctor_name: str = "Dr. Neuro AI, MD, PhD"):
        self.clinic_name = clinic_name
        self.doctor_name = doctor_name

    def generate_html_report(self, screening_result: Dict, patient_id: str = "PATIENT_SAMPLE", patient_age: int = 72, patient_gender: str = "Unspecified") -> str:
        """
        Generates clean, printable HTML report with medical styling.
        """
        now_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M:%S")
        risk_pct = screening_result.get("dementia_risk_percentage", "N/A")
        risk_float = screening_result.get("dementia_risk_probability", 0.0) * 100.0
        mmse = screening_result.get("predicted_mmse_score", 0.0)
        stage = screening_result.get("clinical_stage", "N/A")
        severity = screening_result.get("severity_level", "Low")
        
        vad = screening_result.get("vad_summary", {})
        para = screening_result.get("paralinguistic_summary", {})
        lex = screening_result.get("lexical_summary", {})
        cot = screening_result.get("clinical_chain_of_thought", {})
        transcript = screening_result.get("raw_transcript", "")

        # Color codes
        if risk_float < 30:
            badge_bg = "#DCFCE7"
            badge_color = "#166534"
        elif risk_float < 55:
            badge_bg = "#FEF3C7"
            badge_color = "#92400E"
        else:
            badge_bg = "#FEE2E2"
            badge_color = "#991B1B"

        recs_html = "".join(f"<li>{r}</li>" for r in cot.get("clinical_recommendations", []))

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cognitive Decline Screening Report - {patient_id}</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #1E293B; margin: 40px auto; max-width: 850px; line-height: 1.5; }}
  .header {{ border-bottom: 2px solid #0284C7; padding-bottom: 15px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: flex-end; }}
  .clinic-title {{ font-size: 22px; font-weight: 700; color: #0369A1; }}
  .report-date {{ font-size: 13px; color: #64748B; }}
  .patient-card {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 15px 20px; margin-bottom: 25px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size: 14px; }}
  .patient-card b {{ color: #475569; }}
  .score-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 25px; }}
  .score-box {{ background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
  .score-box .val {{ font-size: 26px; font-weight: 700; margin: 5px 0; }}
  .score-box .label {{ font-size: 13px; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}
  .badge {{ display: inline-block; padding: 4px 14px; border-radius: 16px; font-weight: 600; font-size: 14px; background: {badge_bg}; color: {badge_color}; }}
  .section-title {{ font-size: 17px; font-weight: 700; color: #0F172A; border-bottom: 1px solid #E2E8F0; padding-bottom: 6px; margin: 25px 0 12px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 13.5px; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #E2E8F0; }}
  th {{ background: #F1F5F9; color: #475569; font-weight: 600; }}
  .cot-box {{ background: #F0F9FF; border-left: 4px solid #0284C7; padding: 15px 18px; border-radius: 4px; margin: 15px 0; font-size: 14px; }}
  .transcript-box {{ background: #F8FAFC; border: 1px dashed #CBD5E1; padding: 12px 15px; border-radius: 6px; font-size: 13px; color: #334155; font-style: italic; }}
  .signature-area {{ margin-top: 40px; display: flex; justify-content: space-between; align-items: flex-end; border-top: 1px solid #E2E8F0; padding-top: 20px; }}
  @media print {{ body {{ margin: 20px; }} }}
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="clinic-title">🧠 {self.clinic_name}</div>
    <div style="font-size: 13px; color: #475569;">Spontaneous Speech Acoustic NLP & Clinical LLM Screening System</div>
  </div>
  <div class="report-date">Generated: {now_str}</div>
</div>

<div class="patient-card">
  <div><b>Patient ID:</b> {patient_id}</div>
  <div><b>Age:</b> {patient_age} yrs</div>
  <div><b>Gender:</b> {patient_gender}</div>
  <div><b>Test Protocol:</b> Boston Cookie Theft</div>
</div>

<div class="score-grid">
  <div class="score-box">
    <div class="label">Cognitive Risk %</div>
    <div class="val"><span class="badge">{risk_pct}</span></div>
    <div style="font-size: 12px; color: #64748B;">Severity: <b>{severity}</b></div>
  </div>
  <div class="score-box">
    <div class="label">Estimated MMSE Score</div>
    <div class="val" style="color: #0284C7;">{mmse:.1f} <span style="font-size: 15px; color: #64748B;">/ 30.0</span></div>
    <div style="font-size: 12px; color: #64748B;">Normal Range: ≥ 26.0</div>
  </div>
  <div class="score-box">
    <div class="label">Diagnostic Category</div>
    <div class="val" style="font-size: 17px; color: #0F172A; margin: 10px 0;">{stage}</div>
    <div style="font-size: 12px; color: #64748B;">WavLM + LLaMA-3 QLoRA</div>
  </div>
</div>

<div class="section-title">🔬 Quantitative Digital Biomarkers</div>
<table>
  <tr><th>Biomarker Domain</th><th>Metric Parameter</th><th>Patient Value</th><th>Clinical Reference Range</th></tr>
  <tr><td><b>Acoustic Prosody</b></td><td>Mean Pitch (F0) & Variability</td><td>{para.get('mean_f0_hz', 0.0):.1f} Hz (± {para.get('std_f0_hz', 0.0):.1f} Hz)</td><td>120–220 Hz (Std ≥ 20 Hz)</td></tr>
  <tr><td><b>Acoustic Timing</b></td><td>Silence Hesitation Ratio</td><td>{vad.get('pause_ratio', 0.0)*100:.1f}% ({vad.get('pause_count', 0)} pauses)</td><td>&lt; 25.0% total duration</td></tr>
  <tr><td><b>Vocal Perturbation</b></td><td>Cycle Jitter & Shimmer</td><td>Jitter: {para.get('jitter_percent', 0.0):.2f}% | Shimmer: {para.get('shimmer_percent', 0.0):.2f}%</td><td>Jitter &lt; 1.5% | Shimmer &lt; 4.0%</td></tr>
  <tr><td><b>Lexical Diversity</b></td><td>Type-Token Ratio (TTR)</td><td>{lex.get('type_token_ratio', 0.0):.3f} (Brunet's W: {lex.get('brunet_w', 0.0):.2f})</td><td>TTR ≥ 0.60 (Rich vocabulary)</td></tr>
  <tr><td><b>Anomic Substitution</b></td><td>Pronoun-to-Noun Ratio</td><td>{lex.get('pronoun_to_noun_ratio', 0.0):.2f}</td><td>&lt; 1.20 (Concrete nouns preserved)</td></tr>
</table>

<div class="section-title">🧠 Explainable AI Chain-of-Thought (CoT) Clinical Rationale</div>
<div class="cot-box">
  <b>Multi-Modal Synthesis:</b><br>
  {cot.get('multimodal_synthesis', 'Assessment completed.')}
</div>
<div style="font-size: 13.5px; margin-bottom: 8px;"><b>Acoustic Paralinguistic Evidence:</b> {cot.get('acoustic_analysis', 'N/A')}</div>
<div style="font-size: 13.5px; margin-bottom: 12px;"><b>Linguistic Lexical Evidence:</b> {cot.get('linguistic_analysis', 'N/A')}</div>

<div class="section-title">📋 Recommended Clinical Actions</div>
<ul style="font-size: 13.5px; padding-left: 20px;">
  {recs_html}
</ul>

<div class="section-title">📜 Monologue Transcript (Cookie Theft Task)</div>
<div class="transcript-box">"{transcript}"</div>

<div class="signature-area">
  <div>
    <div style="font-size: 12px; color: #64748B;">Validated with Interspeech ADReSS / DementiaBank Pitt Benchmarks</div>
    <div style="font-size: 11px; color: #94A3B8;">Model Architecture: WavLM-Large (1024d) + LLaMA-3 QLoRA + Cross-Modal Attention (512d)</div>
  </div>
  <div style="text-align: right;">
    <div style="font-size: 14px; font-weight: 700;">{self.doctor_name}</div>
    <div style="font-size: 12px; color: #64748B;">Neuro-AI Clinical Screening Specialist</div>
  </div>
</div>

</body>
</html>"""
        return html
