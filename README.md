# AfriPunctCap: Pan-African Punctuation & Capitalization Restoration

End-to-end pipeline for **acoustic speech-pause alignment** and **fast CPU-based punctuation & capitalization restoration** across **314 African languages**.

Part of the **[AfriSpeech](https://afrispeech.org)** open-source speech initiative.

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-AfriSpeech%2Fafri2universal-blue)](https://huggingface.co/datasets/AfriSpeech/afri2universal)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## Overview

When writing African languages, writers conventionally omit commas at natural clause boundaries, list items, appositives, and conjunctions (e.g. `ne` in Akan, `àti` in Yoruba, `na` in Swahili). However, native speakers consistently make distinct acoustic pauses (350ms–600ms) at these junctions.

For **Text-to-Speech (TTS)** and **Automatic Speech Recognition (ASR)** pipelines:
* **The TTS Bottleneck:** Without commas, TTS models synthesize audio without breath pauses, producing rushed and unnatural phrasing.
* **Acoustic Pause Alignment:** By running forced alignment on the audio from [AfriSpeech/afri2universal](https://huggingface.co/datasets/AfriSpeech/afri2universal) (spanning 314 African languages and 950+ hours of speech), speech pauses ($\ge 350$ms) are detected and restored as punctuation commas.
* **Multi-Task Restoration:** A fast multilingual transformer (`Davlan/afro-xlmr-mini`) is fine-tuned with dual heads to simultaneously restore **Capitalization** (3 classes) and **Punctuation** (5 classes: commas, periods, question marks, exclamations) across all 314 languages.
* **Ultra-Fast CPU Inference:** Quantized into an INT8 ONNX engine running in **~15–25 ms** per sentence on standard CPUs.

---

## Languages Covered

Spans **314 African languages and dialects** across all 4 major African language families (Niger-Congo, Afroasiatic, Nilo-Saharan, Khoisan), including:
* **West Africa:** Akan/Twi (`aka`), Yoruba (`yor`), Hausa (`hau`), Igbo (`ibo`), Ewe (`ewe`), Ga (`gaa`), Wolof (`wol`), Fula (`ful`), Mende (`men`), etc.
* **East Africa:** Swahili (`swh`), Amharic (`amh`), Somali (`som`), Oromo (`orm`), Tigrinya (`tir`), Luganda (`lug`), Kinyarwanda (`kin`), Acholi (`ach`), etc.
* **Central & Southern Africa:** Lingala (`lin`), Kongo (`kon`), Shona (`sna`), Zulu (`zul`), Xhosa (`xho`), Sotho (`sot`), Tswana (`tsn`), etc.
* **North Africa:** Egyptian Arabic (`arz`), Moroccan Arabic (`ary`), Algerian Arabic (`arq`), Tunisian Arabic (`aeb`), etc.

---

## Repository Structure

```
├── afri_restorer.py            # Standalone CPU inference library and CLI tool (ONNX INT8 & PyTorch)
├── requirements.txt           # Project dependencies
├── align_afri2universal.py    # Multi-language acoustic pause alignment using Meta MMS Forced Aligner
├── train_afri_multilingual.py # Fine-tuning Afro-XLMR-mini, ONNX export, and INT8 dynamic quantization
├── orchestrate_pan_african.py # Autonomous pipeline monitor and execution controller
├── benchmark_h200_speed.py    # Throughput benchmarking on NVIDIA H200 GPU
└── test_align_multilingual.py # Cross-language alignment verification test
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/AfriSpeech/afri-punct-cap.git
cd afri-punct-cap
pip install -r requirements.txt
```

### 2. Python Usage

```python
from afri_restorer import AfriRestorer

# Automatically uses ultra-fast quantized INT8 ONNX if available
restorer = AfriRestorer(backend="auto")

# Yoruba example (raw unpunctuated ASR transcript)
yor_raw = "a maa n je ounje asaraloore ka ma baa saisan tori ta o ba se bee se la maa fi iwonba owo to n wole fun wa tojú aisan"
print(restorer.restore(yor_raw))

# Swahili example
swh_raw = "dokezo tumia dakika kumi kabla ya kumaliza kusoma kupitia mambo uliyosoma"
print(restorer.restore(swh_raw))

# Twi example
twi_raw = "ahene nanahene nana kwame mensah bonsu aka se ese se wobebo dawuro ama chief of staff julius debrah"
print(restorer.restore(twi_raw))
```

### 3. CLI Usage

```bash
# Run on a single sentence
python3 afri_restorer.py "dokezo tumia dakika kumi kabla ya kumaliza kusoma kupitia mambo uliyosoma"

# Interactive REPL mode
python3 afri_restorer.py
```

---

## Pipelines

### 1. Acoustic Pause Alignment (`align_afri2universal.py`)

Aligns all 314 language partitions from `AfriSpeech/afri2universal` on GPU:

```bash
python3 align_afri2universal.py
```

* Uses **Meta MMS Forced Aligner** (`torchaudio.pipelines.MMS_FA`) with `uroman` universal Romanization.
* Identifies acoustic silent intervals $\ge 350$ ms.
* Inserts speech-pause commas into parallel columns: `original_transcript` and `comma_fixed_transcript`.
* Idempotent and resumable per-language parquet saves.

### 2. Model Training (`train_afri_multilingual.py`)

Fine-tunes `Davlan/afro-xlmr-mini` with multi-task dual heads:

```bash
python3 train_afri_multilingual.py \
    --aligned_dir data/aligned/split=train \
    --output_dir models/afri-punct-cap-mini \
    --epochs 4 \
    --batch_size 64 \
    --lr 5e-5
```

* **Zero Hallucination Guarantee:** Non-autoregressive sequence tagging preserves all input words in their original order.
* **Export:** Saves PyTorch weights, safetensors, standard ONNX, and **INT8 quantized ONNX** (~115 MB).

---

## Benchmarks & Performance

| Metric | PyTorch (FP32) | ONNX (INT8 Quantized) |
|---|---|---|
| **Model Size** | 460 MB | **115 MB** (4x smaller) |
| **CPU Latency (2-core)** | ~80 ms | **~15–25 ms** (3x–5x faster) |
| **Throughput (H200 GPU)** | ~35 sent/sec | — |
| **Supported Languages** | **314 African Languages** | **314 African Languages** |

---

## Citation & Credits

* Base Corpus: [AfriSpeech/afri2universal](https://huggingface.co/datasets/AfriSpeech/afri2universal)
* Multilingual Backbone: [Davlan/afro-xlmr-mini](https://huggingface.co/Davlan/afro-xlmr-mini) (David Adelani et al.)
* Alignment Model: [Meta MMS Forced Aligner](https://pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html)
* Organization: **[AfriSpeech](https://afrispeech.org)** & **[Ghana NLP](https://ghananlp.org)**
