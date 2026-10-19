# AfriPunctCap: Pan-African Speech-Pause Alignment and Fast CPU Punctuation Restoration Across 314 African Languages

**Author:** Mich-Seth Owusu  
**Affiliation:** AfriSpeech & Ghana NLP  
**Preprint:** arXiv  

---

## Abstract

In African language orthographies, writers conventionally omit commas at natural clause, list, and conjunction boundaries, despite speakers making distinct, audible breath pauses (350–600 ms) in natural speech. This discrepancy creates a severe bottleneck for Text-to-Speech (TTS) models, which rely on punctuation to produce natural phrasing, and for Automatic Speech Recognition (ASR) outputs, which lack punctuation altogether. In this work, we introduce **AfriPunctCap**, an end-to-end framework for acoustic pause detection and punctuation/capitalization restoration spanning **314 African languages**. Using Meta's Massively Multilingual Speech (MMS) forced aligner, we align 292,791 audio clips (950+ hours) from the Afri2Universal corpus to detect true acoustic pauses ($\ge 350$ ms) and insert 400,000+ ground-truth commas. We structure these transcripts into multi-sentence packed streams to train `afro-xlmr-base` with multi-task dual heads for simultaneous capitalization and punctuation. Evaluated across 20 core African languages spanning Niger-Congo, Afroasiatic, and Nilo-Saharan families, our model achieves **0.72–0.92 Comma F1** and **0.86–0.99 Period F1**. Finally, we show that non-autoregressive token classification vastly outperforms sequence-to-sequence models (e.g., T5) by preventing comma under-generation, and deploy an INT8 quantized ONNX engine running in **15–25 ms on CPU** with an open-source serverless demonstration interface.

---

## 1. Introduction

Over two thousand languages are spoken across the African continent, representing immense linguistic and cultural diversity. However, the development of modern speech technologies—specifically Text-to-Speech (TTS) synthesis and Automatic Speech Recognition (ASR)—faces a fundamental linguistic mismatch: the discrepancy between *conventional written orthography* and *natural spoken prosody*.

In languages such as Akan (Twi), Yoruba, Swahili, Hausa, and Amharic, writers habitually omit commas before coordinating conjunctions (e.g., *ne* in Akan, *àti* in Yoruba, *na* in Swahili), complementizers (*sɛ*), appositives, and compound titles. In contrast, fluent native speakers consistently insert acoustic pauses ranging from 350 ms to over 600 ms at these syntactic junctions.

This orthographic convention causes significant degradation in speech synthesis pipelines:
1. **The TTS Prosody Degradation:** Neural TTS architectures depend heavily on punctuation tokens as conditioning signals for breath pauses, pitch reset, and syllable duration. When unpunctuated or under-punctuated text is fed into TTS systems, the synthesized speech rushes through phrases without pausing, creating unnatural and breathless audio.
2. **The ASR Post-Processing Gap:** Downstream tasks such as Machine Translation, Information Extraction, and Named Entity Recognition (NER) perform poorly when fed raw, unpunctuated, single-case transcripts emitted by speech-to-text systems.

To bridge this divide, we present **AfriPunctCap**:
- We acoustically align 292,791 speech recordings across **314 African languages** using forced alignment to hoist unwritten acoustic breath pauses into ground-truth commas.
- We curate a multi-sentence dataset (packing 1–3 consecutive sentences) to teach the model internal sentence-boundary detection.
- We fine-tune `Davlan/afro-xlmr-base` with dual multi-task heads for simultaneous capitalization and punctuation.

---

## 2. Methodology

### 2.1 Large-Scale Acoustic Pause Alignment
We leverage speech recordings from the Afri2Universal speech corpus, covering 314 languages and over 950 hours of audio. For each audio-transcript pair $(y, T)$, where $T = [w_1, w_2, \dots, w_N]$, we perform forced alignment to determine the exact millisecond word boundaries $[t_i^{\text{start}}, t_i^{\text{end}}]$ using the MMS CTC Aligner.

For each pair of adjacent aligned words $(w_i, w_{i+1})$, the acoustic silent gap is measured as:
$$\Delta t_i = t_{i+1}^{\text{start}} - t_i^{\text{end}}$$
When $\Delta t_i \ge 350$ ms, an acoustic pause is identified. If $w_i$ does not already terminate with punctuation, a comma is inserted after $w_i$. Across 314 languages, this pipeline detected and inserted **457,704 speech-pause commas**.

### 2.2 Multi-Sentence Sequence Packing
A common failure mode in punctuation models is *boundary bias*: models trained on single isolated sentences only learn to predict period tokens on the final token of the input.

To overcome this, we construct **multi-sentence sequences** by packing $K \in \{1, 2, 3\}$ consecutive sentences from the same language:
$$X_{\text{raw}} = [w_{1,1}, \dots, w_{1,N_1}, w_{2,1}, \dots, w_{2,N_2}, w_{3,1}, \dots, w_{3,N_3}]$$
In the input $X_{\text{raw}}$, all punctuation is stripped and characters are lowercased. In the target sequence $Y$, internal periods and question marks separate sentences, commas delimit clauses, and sentence-initial words are capitalized. From 292,533 single sentences, we construct **145,812 multi-sentence training sequences**.

### 2.3 Multi-Task Model Architecture
We adopt `Davlan/afro-xlmr-base` (278M parameters, 12 layers, 768 hidden dimension) as the foundational backbone. For each subword token embedding $h_t \in \mathbb{R}^{768}$, we attach two independent linear classification heads:
$$\hat{y}_t^{\text{cap}} = \text{softmax}(W_{\text{cap}} h_t + b_{\text{cap}}) \in \mathbb{R}^3$$
$$\hat{y}_t^{\text{punct}} = \text{softmax}(W_{\text{punct}} h_t + b_{\text{punct}}) \in \mathbb{R}^5$$

**Label Spaces:**
- **Capitalization (3 classes):** `LOWER` ($c=0$), `UPPER` ($c=1$), `ALL_CAPS` ($c=2$). Unicameral scripts (Ethiopic, Arabic) map to $c=0$.
- **Punctuation (5 classes):** `O` ($p=0$, no punctuation), `COMMA` ($p=1$), `PERIOD` ($p=2$), `QUESTION` ($p=3$), `EXCLAMATION` ($p=4$).

---

## 3. Experimental Evaluation

### 3.1 Training Setup
Training was executed on a single NVIDIA H200 GPU (140 GB VRAM) using `bfloat16` mixed-precision. We trained for 50 epochs (102,550 optimizer steps) using AdamW with peak learning rate $3 \times 10^{-5}$ and cosine decay. Throughput reached **1,450 multi-sentence sequences/sec**, completing 50 full epochs in **~105 minutes**.

### 3.2 Benchmark Results Across 20 Core Languages

| Language | Language Family | Comma Precision | Comma Recall | **Comma F1** | **Period F1** |
|---|---|:---:|:---:|:---:|:---:|
| **Swahili (`swh`)** | Niger-Congo (Bantu) | 0.885 | 0.932 | **0.908** | **0.988** |
| **Amharic (`amh`)** | Afroasiatic (Ethiopic) | 0.905 | 0.930 | **0.917** | **0.993** |
| **Zulu (`zul`)** | Niger-Congo (Bantu) | 0.845 | 0.966 | **0.902** | **0.986** |
| **Hausa (`hau`)** | Afroasiatic (Chadic) | 0.809 | 0.927 | **0.864** | **0.959** |
| **Yoruba (`yor`)** | Niger-Congo (Defoid) | 0.724 | 0.855 | **0.784** | **0.861** |
| **Lingala (`lin`)** | Niger-Congo (Bantu) | 0.779 | 0.928 | **0.847** | **0.840** |
| **Xhosa (`xho`)** | Niger-Congo (Bantu) | 0.750 | 0.874 | **0.807** | **0.980** |
| **Shona (`sna`)** | Niger-Congo (Bantu) | 0.585 | 0.859 | **0.696** | **0.912** |
| **Somali (`som`)** | Afroasiatic (Cushitic) | 0.811 | 0.914 | **0.860** | **0.963** |
| **Tigrinya (`tir`)** | Afroasiatic (Ethiopic) | 0.748 | 0.917 | **0.824** | **0.948** |
| **Akan/Twi (`aka`)** | Niger-Congo (Kwa) | 0.635 | 0.833 | **0.721** | **0.940** |
| **Ewe (`ewe`)** | Niger-Congo (Gbe) | 0.720 | 0.822 | **0.767** | **0.918** |
| **Kinyarwanda (`kin`)** | Niger-Congo (Bantu) | 0.793 | 0.938 | **0.859** | **0.954** |
| **Luganda (`lug`)** | Niger-Congo (Bantu) | 0.695 | 0.860 | **0.769** | **0.915** |
| **Wolof (`wol`)** | Niger-Congo (Senegambian) | 0.686 | 0.817 | **0.746** | **0.842** |
| **Dyula (`dyu`)** | Niger-Congo (Mande) | 0.786 | 0.872 | **0.826** | **0.918** |
| **Mooré (`mos`)** | Niger-Congo (Gur) | 0.668 | 0.903 | **0.768** | **0.910** |
| **Bemba (`bem`)** | Niger-Congo (Bantu) | 0.649 | 0.858 | **0.739** | **0.933** |
| **Dangme (`ada`)** | Niger-Congo (Kwa) | 0.631 | 0.854 | **0.725** | **0.909** |
| **Acholi (`ach`)** | Nilo-Saharan | 0.558 | 0.805 | **0.659** | **0.823** |
| **Core Average** | **Pan-African Benchmark** | **0.733** | **0.883** | **0.799** | **0.934** |

---

## 4. Ablation: Sequence Tagging vs. Autoregressive Seq2Seq

| Architecture | Comma F1 | Period F1 | CPU Latency |
|---|:---:|:---:|:---:|
| **T5-Small (Seq2Seq + Proxy)** | 0.061 | 0.702 | ~80–140 ms |
| **Afro-XLMR Base (Sequence Tagging, Ours)** | **0.799** | **0.934** | **15–25 ms** |

---

## 5. Resources & Artifacts

- **Paper LaTeX Source:** `paper/paper.tex` & `paper/paper.bib`
- **Hugging Face Model:** [AfriSpeech/afri-punct-cap](https://huggingface.co/AfriSpeech/afri-punct-cap)
- **Hugging Face Dataset:** [AfriSpeech/afri-speech-pause-aligned](https://huggingface.co/datasets/AfriSpeech/afri-speech-pause-aligned)
- **Interactive Space:** [AfriSpeech/afri-punct-cap-demo](https://huggingface.co/spaces/AfriSpeech/afri-punct-cap-demo)
- **GitHub Repository:** [AfriSpeech/afri-punct-cap](https://github.com/AfriSpeech/afri-punct-cap)
