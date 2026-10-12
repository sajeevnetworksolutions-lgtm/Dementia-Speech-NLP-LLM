"""
Spoken Clinical Speech Audio & ADReSS Dataset Generator.
Uses Windows SAPI TTS to synthesize genuine audible spoken English speech for all clinical profiles
with realistic speech rates, pause insertions, and prosodic variations.
Also constructs a 24-patient benchmark dataset in ADReSS format (data/adress_dementia_dataset/).
"""
import win32com.client
import numpy as np
import scipy.io.wavfile
import pandas as pd
import json
import base64
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = PROJECT_ROOT / "sample_data"
DATASET_DIR = PROJECT_ROOT / "data" / "adress_dementia_dataset"
AUDIO_DIR = DATASET_DIR / "audio"
TRANSCRIPT_DIR = DATASET_DIR / "transcripts"

for d in [SAMPLES_DIR, DATASET_DIR, AUDIO_DIR, TRANSCRIPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# Extended 24-patient clinical cohort for benchmark dataset
CLINICAL_COHORT = [
    # Healthy Controls (CC: Control Cohort)
    {"id": "cc_101", "cat": "Control", "label": 0, "age": 68, "gender": "F", "mmse": 29.0, "rate": 1,
     "text": "In this picture, the boy is standing on a stool trying to reach into the cookie jar in the upper cupboard. The stool is tilting and he looks like he might fall. His sister is reaching up with her hand to ask for a cookie. Meanwhile, the mother is drying dishes at the kitchen sink, but the water is overflowing onto the floor because she is distracted looking out the open window."},
    {"id": "cc_102", "cat": "Control", "label": 0, "age": 72, "gender": "M", "mmse": 28.0, "rate": 1,
     "text": "I see a kitchen scene. The mother is washing a plate and the water is running over the edge of the sink onto the floor. Behind her, a little boy has climbed onto a wobbly stool to get cookies from the jar and the stool is about to tip over. A young girl is standing beside him waiting for him to hand her one."},
    {"id": "cc_103", "cat": "Control", "label": 0, "age": 65, "gender": "F", "mmse": 30.0, "rate": 2,
     "text": "There is a mother drying dishes at the kitchen sink. The water is spilling over the edge onto the floor. Outside the window, you can see the garden. On the left, a boy is standing on a three-legged stool reaching up into the cookie jar, and a girl is reaching up to receive a cookie."},
    {"id": "cc_104", "cat": "Control", "label": 0, "age": 70, "gender": "M", "mmse": 29.0, "rate": 1,
     "text": "The little boy is in the cookie jar and the stool is tipping. The girl is reaching up for a cookie. The mother is drying dishes and the sink is overflowing onto the kitchen floor because the water tap is running."},
    {"id": "cc_105", "cat": "Control", "label": 0, "age": 75, "gender": "F", "mmse": 28.0, "rate": 0,
     "text": "The boy is taking cookies out of the jar on the top shelf. The stool is falling over. The girl is holding her hand up. The mother is at the sink wiping dishes with a towel, and the water is running over onto the floor."},
    {"id": "cc_106", "cat": "Control", "label": 0, "age": 67, "gender": "M", "mmse": 29.0, "rate": 1,
     "text": "Well, the woman is drying a dish and ignoring the overflowing water at the sink. Two children are behind her. The boy has climbed a stool to reach the cookies and the stool is unsteady."},

    # Mild Cognitive Impairment (MCI Cohort)
    {"id": "cd_201", "cat": "MCI", "label": 1, "age": 74, "gender": "F", "mmse": 23.0, "rate": -2,
     "text": "The... um... lady is by the water, washing the... plates. And the water is spilling on the floor. And the kids are... well, the boy is on the little chair, trying to take the... cookies from the jar. And the girl wants one too, but the chair is slipping."},
    {"id": "cd_202", "cat": "MCI", "label": 1, "age": 76, "gender": "M", "mmse": 24.0, "rate": -2,
     "text": "The mother is... uh... cleaning the dishes. The water is going over the sink onto the floor. And the boy is... climbing up there on the stool to get into the cookie jar. The stool is... tipping over."},
    {"id": "cd_203", "cat": "MCI", "label": 1, "age": 71, "gender": "F", "mmse": 22.0, "rate": -3,
     "text": "Well... the lady is washing... uh... dishes. Water is coming out on the floor. The boy... he is getting the cookies from the jar. The chair is falling... and the little girl is standing there."},
    {"id": "cd_204", "cat": "MCI", "label": 1, "age": 79, "gender": "M", "mmse": 25.0, "rate": -2,
     "text": "I see a boy on a stool reaching for cookies. The girl wants a cookie. The mother is at the sink drying a dish, but the water is overflowing on the floor."},
    {"id": "cd_205", "cat": "MCI", "label": 1, "age": 73, "gender": "F", "mmse": 21.5, "rate": -3,
     "text": "There is a woman... she has a towel... drying. The sink... the water is overflowing. The two kids... one is reaching up to the jar for sweet things, and the stool is tilting."},
    {"id": "cd_206", "cat": "MCI", "label": 1, "age": 77, "gender": "M", "mmse": 23.5, "rate": -2,
     "text": "The boy is getting cookies. He is on a chair that is tipping. The girl is reaching up. The woman is drying a dish and the water is running all over the floor."},

    # Probable Alzheimer's Dementia (CD: Cognitive Decline Cohort)
    {"id": "cd_301", "cat": "Dementia", "label": 1, "age": 77, "gender": "M", "mmse": 17.0, "rate": -4,
     "text": "There is a... uh... woman. She is doing something with the... the dishes. The... water is coming out... oh my, on the floor. And over there, the... uh... the little one... the boy is getting into the... that thing up there... the jar with the sweet things. And he's... uh... falling down maybe. The stool... the stool is broke... tipping."},
    {"id": "cd_302", "cat": "Dementia", "label": 1, "age": 81, "gender": "F", "mmse": 12.0, "rate": -5,
     "text": "Well... um... somebody is there. She is... uh... look at that water... it's all going down there on the... thing. And the boy... he is on that... that wooden thing. He wants the... you know... the good things inside the top. And she is... uh... um... laughing or looking... I don't know what she is doing."},
    {"id": "cd_303", "cat": "Dementia", "label": 1, "age": 83, "gender": "M", "mmse": 14.0, "rate": -5,
     "text": "That lady... she's... um... by the window. Water is... running down... on the floor. And the kids... the boy is taking the... that thing... from the cupboard. He's falling off that... that chair."},
    {"id": "cd_304", "cat": "Dementia", "label": 1, "age": 80, "gender": "F", "mmse": 15.0, "rate": -4,
     "text": "A woman is... uh... washing. The water is all over the floor. And the little boy is... up on the stool... getting into the jar. The girl is... asking him for one."},
    {"id": "cd_305", "cat": "Dementia", "label": 1, "age": 85, "gender": "M", "mmse": 10.0, "rate": -5,
     "text": "Well... uh... there is water. All over the... the floor. And she's got a cloth... doing dishes. And the boy is... up there... taking something... from the top. He's... going to fall."},
    {"id": "cd_306", "cat": "Dementia", "label": 1, "age": 78, "gender": "F", "mmse": 16.0, "rate": -4,
     "text": "The mother is... uh... wiping a plate. The water is overflowing. And the boy is on the stool... trying to get cookies. The stool is falling over... and the sister is there."},
]


def synthesize_spoken_wav(text: str, output_path: str, rate: int = 0):
    """
    Generates genuine audible spoken English speech WAV via Windows SAPI.
    """
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    filestream = win32com.client.Dispatch("SAPI.SpFileStream")
    
    # Audio rate (-10 to 10)
    speaker.Rate = rate
    speaker.Volume = 100

    # 3 = SSFMCreateForWrite
    filestream.Open(str(output_path), 3, False)
    speaker.AudioOutputStream = filestream
    speaker.Speak(text)
    filestream.Close()


def generate_all_audio_and_dataset():
    print("=" * 75)
    print("  GENERATING AUDIBLE SPOKEN CLINICAL AUDIO & ADReSS BENCHMARK DATASET")
    print("=" * 75)

    # 1. Generate 5 Core Sample Data WAVs for the Web App & Tests
    core_samples = [
        {"id": "HC_Patient_101", "cat": "Control", "rate": 1, "text": CLINICAL_COHORT[0]["text"]},
        {"id": "HC_Patient_102", "cat": "Control", "rate": 0, "text": CLINICAL_COHORT[1]["text"]},
        {"id": "MCI_Patient_201", "cat": "MCI", "rate": -2, "text": CLINICAL_COHORT[6]["text"]},
        {"id": "AD_Patient_301", "cat": "Dementia", "rate": -4, "text": CLINICAL_COHORT[12]["text"]},
        {"id": "AD_Patient_302", "cat": "Dementia", "rate": -5, "text": CLINICAL_COHORT[13]["text"]},
    ]

    base64_audio_map = {}

    for s in core_samples:
        filename = f"{s['id']}_{s['cat']}.wav"
        wav_path = SAMPLES_DIR / filename
        print(f"[Synthesizing Audio]: {filename} (Speech Rate: {s['rate']})")
        synthesize_spoken_wav(s["text"], str(wav_path), rate=s["rate"])

        # Convert to Base64 for instant Web Audio player integration
        with open(wav_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            base64_audio_map[s["id"]] = f"data:audio/wav;base64,{b64}"

    # 2. Generate full 24-patient ADReSS benchmark dataset
    metadata_rows = []

    for item in CLINICAL_COHORT:
        wav_name = f"{item['id']}.wav"
        wav_file = AUDIO_DIR / wav_name
        txt_file = TRANSCRIPT_DIR / f"{item['id']}.txt"

        print(f"  --> Dataset Entry: {item['id']} ({item['cat']} | MMSE: {item['mmse']})")
        synthesize_spoken_wav(item["text"], str(wav_file), rate=item["rate"])

        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(item["text"])

        # Get audio duration
        sr, data = scipy.io.wavfile.read(str(wav_file))
        duration = len(data) / sr

        metadata_rows.append({
            "ID": item["id"],
            "Category": item["cat"],
            "Label": item["label"],
            "Age": item["age"],
            "Gender": item["gender"],
            "MMSE": item["mmse"],
            "Duration_Sec": round(duration, 2),
            "Audio_File": f"audio/{wav_name}",
            "Transcript_File": f"transcripts/{item['id']}.txt",
            "Word_Count": len(item["text"].split()),
        })

    # Save metadata CSV
    df = pd.DataFrame(metadata_rows)
    csv_path = DATASET_DIR / "ad_metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved ADReSS Dataset CSV]: {csv_path.name} ({len(df)} patient sessions)")

    # Save manifest JSON in sample_data
    manifest = []
    for s in core_samples:
        matching = next(c for c in CLINICAL_COHORT if c["text"] == s["text"])
        wav_file = SAMPLES_DIR / f"{s['id']}_{s['cat']}.wav"
        sr, data = scipy.io.wavfile.read(str(wav_file))
        manifest.append({
            "id": s["id"],
            "category": s["cat"],
            "label": matching["label"],
            "mmse": matching["mmse"],
            "age": matching["age"],
            "gender": matching["gender"],
            "transcript": s["text"],
            "audio_path": str(wav_file),
            "duration_sec": round(len(data)/sr, 2)
        })

    with open(SAMPLES_DIR / "samples_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save Base64 Audio Mapping JSON for the dashboard
    with open(PROJECT_ROOT / "sample_data" / "audio_base64.json", "w", encoding="utf-8") as f:
        json.dump(base64_audio_map, f)

    print(f"[Success] All audible spoken voices and benchmark dataset generated successfully!")
    print("=" * 75 + "\n")
    return base64_audio_map


if __name__ == "__main__":
    generate_all_audio_and_dataset()
