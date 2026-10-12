# M.Tech Dissertation Viva-Voce Defense Guide

## Early Dementia & Cognitive Decline Screening using Acoustic NLP & Fine-Tuned LLMs

This document prepares the candidate for comprehensive examination by academic and clinical external review committees.

---

### Section 1: Clinical Neurocognition & Speech Biomarkers

#### Q1: Why is spontaneous speech analysis superior to structural MRI for early screening?
**Examiner Focus:** Clinical rationale & early diagnostic window.  
**Answer:**  
Structural MRI measures macroscopic anatomical tissue loss (e.g., hippocampal and cortical atrophy). However, significant atrophy only manifests **after billions of neurons have already died**, at which point disease progression is largely irreversible. Conversely, spoken discourse formulation requires rapid, synchronized orchestration across the temporal, frontal, and parietal cortices in real time. Cognitive micro-hesitations, lexical search delays, and syntactic degradation appear **3 to 5 years earlier** than macroscopic brain shrinkage. Voice screening is 100% non-invasive, takes under 2 minutes, and costs virtually \$0 compared to \$1,500+ MRI scans.

#### Q2: What is the Boston Cookie Theft description task and why is it standardized?
**Examiner Focus:** Benchmark validity & psychological standardization.  
**Answer:**  
Originating from the *Boston Diagnostic Aphasia Examination (BDAE)*, the Cookie Theft task presents a complex visual stimulus containing simultaneous actions (a boy stealing cookies from an upper cupboard while his stool tips; a mother washing dishes while the kitchen sink overflows). It elicits spontaneous connected speech that tests:
1. **Lexical retrieval:** Naming concrete objects (*cookie jar, stool, cupboard, sink, faucet*).
2. **Spatial awareness & visual tracking:** Identifying the tipping stool and water puddle.
3. **Temporal coordination:** Relating multiple actions simultaneously.
Because the visual stimulus is identical across subjects, it provides a strictly controlled baseline to measure word-finding delays and semantic substitutions across age and gender cohorts.

#### Q3: What is the difference between normal hesitation and pathological micro-pauses in dementia?
**Examiner Focus:** Signal processing & acoustic differentiation.  
**Answer:**  
In healthy controls, speech pauses predominantly occur at **major syntactic boundaries** (clause/sentence breaks) to inhale or plan discourse (typically $<250$ ms). In dementia patients, pauses occur **intra-phrase** (in the middle of sentences or directly preceding content nouns), with mean durations frequently exceeding $750–1500$ ms. These intra-phrase pauses reflect anomic retrieval failures (the inability to access lexical concepts from semantic memory).

---

### Section 2: Deep Learning Architecture & Multi-Modal Fusion

#### Q4: Why use WavLM instead of standard Mel-Spectrograms or MFCCs?
**Examiner Focus:** Self-supervised representation learning vs handcrafted audio features.  
**Answer:**  
Handcrafted acoustic features (e.g., eGeMAPS, MFCCs) capture static spectral envelopes but fail to model long-range temporal context and phonetic structure. **WavLM** is pre-trained on 94,000 hours of speech using masked speech denoising and pseudo-phoneme prediction. It produces rich 1024-dimensional contextual embeddings that disentangle phonetic content from paralinguistic speaker states (jitter, shimmer, pitch dynamics, breathiness), allowing the downstream network to capture subtle acoustic micro-pauses without manual engineering.

#### Q5: Explain the mechanics of the Temporal Cross-Modal Attention Layer.
**Examiner Focus:** Multi-modal fusion mathematics.  
**Answer:**  
Standard late fusion merely concatenates pooled audio and text vectors, losing temporal alignment. In our **Temporal Cross-Modal Attention layer**:
1. **Queries ($\mathbf{Q}$):** Formed from linguistic token vectors ($\mathbf{H}_{text} \in \mathbb{R}^{L \times 768}$).
2. **Keys & Values ($\mathbf{K}, \mathbf{V}$):** Formed from acoustic frame vectors ($\mathbf{H}_{audio} \in \mathbb{R}^{T \times 1024}$).
3. **Cross-Attention:**
   $$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
4. This explicitly computes which acoustic frames (e.g., a 1.2-second silence burst) coincide with specific spoken tokens (e.g., *"the... um... that wooden thing"*), enabling the network to learn synchronous audio-linguistic decay patterns.

#### Q6: How does QLoRA allow fine-tuning an 8-Billion parameter LLaMA-3 model on a single consumer GPU?
**Examiner Focus:** Parameter efficiency & memory optimization.  
**Answer:**  
1. **4-Bit NormalFloat (NF4) Quantization:** Quantizes base model 16-bit float weights into optimal 4-bit distributions, reducing model weight memory from $\sim 16$ GB to $\sim 4.5$ GB.
2. **Double Quantization (DQ):** Quantizes quantization constants, saving an additional 0.37 bits/parameter.
3. **Paged Optimizers:** Uses CUDA unified memory to prevent out-of-memory spikes during gradient checkpointing.
4. **Low-Rank Adapters (LoRA):** Injects low-rank decomposition matrices ($r=16, \alpha=32$) into key linear layers ($W_q, W_k, W_v, W_o$), training $<1\%$ of total parameters while preserving full 8B reasoning capability.

---

### Section 3: Multi-Task Learning & Clinical Metrics

#### Q7: Why formulate the problem as Multi-Task Learning (Classification + MMSE Regression)?
**Examiner Focus:** Objective formulation & clinical utility.  
**Answer:**  
1. **Regularization & Synergy:** Binary classification ($0/1$) is coarse, whereas continuous MMSE regression ($0–30$) provides fine-grained granularity. Training jointly forces the shared latent representation to capture both categorical disease probability and quantitative severity progression.
2. **Loss Function:**
   $$\mathcal{L}_{total} = \lambda_{cls} \mathcal{L}_{BCE}(\hat{y}, y) + \lambda_{reg} \mathcal{L}_{Huber}(\hat{m}, m)$$
   Smooth L1 / Huber loss prevents outlier degradation for severe cognitive scores.

#### Q8: What are the clinical cutoff values for MMSE scores?
**Examiner Focus:** Clinical domain knowledge.  
**Answer:**  
* **$26.0 – 30.0$:** Normal Cognition / Healthy Control
* **$21.0 – 25.9$:** Mild Cognitive Impairment (MCI)
* **$15.0 – 20.9$:** Mild Alzheimer's Dementia
* **$0.0 – 14.9$:** Moderate to Severe Dementia

---

### Section 4: Computational Complexity & Deployment

#### Q9: What is the inference latency and memory footprint?
**Examiner Focus:** Production viability.  
**Answer:**  
* **Peak VRAM:** $< 7.2$ GB during training; $< 1.8$ GB during inference with 4-bit quantization.
* **Inference Speed:** Under $1.2$ seconds for a 2-minute audio file on an NVIDIA RTX 3060 / T4 GPU; $\sim 3.8$ seconds on an 8-core CPU.
* **Platform Support:** Fully web-deployable via Streamlit, with TorchScript/ONNX export support for hospital tablet integration.
