# AfriPunctCap: Pan-African Punctuation & Capitalization Restoration

End-to-end pipeline for **acoustic speech-pause alignment** and **fast CPU-based punctuation & capitalization restoration** across **314 African languages**.

Part of the **[AfriSpeech](https://afrispeech.org)** open-source speech initiative.

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-AfriSpeech%2Fafri--punct--cap-yellow)](https://huggingface.co/AfriSpeech/afri-punct-cap)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-AfriSpeech%2Fafri--speech--pause--aligned-blue)](https://huggingface.co/datasets/AfriSpeech/afri-speech-pause-aligned)
[![Base Corpus](https://img.shields.io/badge/%F0%9F%A4%97%20Base%20Corpus-AfriSpeech%2Fafri2universal-green)](https://huggingface.co/datasets/AfriSpeech/afri2universal)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## Overview

When writing African languages, writers conventionally omit commas at natural clause boundaries, list items, appositives, and conjunctions (e.g. `ne` in Akan, `àti` in Yoruba, `na` in Swahili). However, native speakers consistently make distinct acoustic pauses (350ms–600ms) at these junctions.

For **Text-to-Speech (TTS)** and **Automatic Speech Recognition (ASR)** pipelines:
* **The TTS Bottleneck:** Without commas, TTS models synthesize audio without breath pauses, producing rushed and unnatural phrasing.
* **Acoustic Pause Alignment & Multi-Corpus Aggregation:** Combines 3,305,255 multi-sentence passages across **751 African languages** from `AfriSpeech/africa-corpus`, `AfriSpeech/african-corpus-jw`, and `AfriSpeech/african-transcribed-speech-monolingual`.
* **Multi-Task Restoration:** Fine-tunes `Davlan/afro-xlmr-base` (278M params) with dual heads to simultaneously restore **Capitalization** (3 classes) and **Punctuation** (5 classes: commas, periods, question marks, exclamations) across all 751 languages with **zero language tags**.
* **Ultra-Fast CPU Inference:** Quantized into an INT8 ONNX engine running in **~15–25 ms** per sentence on standard CPUs.

### 🌍 751 Languages & Pan-African Generalizability

The model is trained on **751 African languages and dialects** representing all four major African language families (Niger-Congo, Afroasiatic, Nilo-Saharan, and Khoisan):
* **Shared Prosodic Typology:** Grammatical clause boundaries, appositives, list elements, conjunctions, and breath pauses follow shared syntactic and acoustic rhythms across related African language subfamilies (e.g. Bantu, Kwa, Chadic, Cushitic, Senegambian, Gur, Mande).
* **Pan-African Representation:** Built on `Davlan/afro-xlmr-base` with SentencePiece tokenization covering African orthographies (Ethiopic, Arabic, Latin with African phonetic extensions), the model learns universal cross-lingual punctuation cues rather than language-memorized tokens.
* **Broad Dialect Coverage:** Unwritten, low-resource, or regional language varieties that share phonetic or syntactic phrasing with these 751 languages benefit directly from zero-shot punctuation and capitalization transfer.

---

## Languages Covered

Spans **751 African languages and dialects** across all 4 major African language families (Niger-Congo, Afroasiatic, Nilo-Saharan, Khoisan), including:
* **West Africa:** Akan/Twi (`aka`/`twi`), Yoruba (`yor`), Hausa (`hau`), Igbo (`ibo`), Ewe (`ewe`), Ga (`gaa`), Wolof (`wol`), Fula (`ful`), Mende (`men`), Fanti (`fat`), Dagbani (`dag`), Dangme (`ada`), etc.
* **East Africa:** Swahili (`swh`), Amharic (`amh`), Somali (`som`), Oromo (`gax`), Tigrinya (`tir`), Luganda (`lug`), Kinyarwanda (`kin`), Acholi (`ach`), etc.
* **Central & Southern Africa:** Lingala (`lin`), Kongo (`kon`), Shona (`sna`), Zulu (`zul`), Xhosa (`xho`), Sotho (`sot`), Tswana (`tsn`), Bemba (`bem`), etc.
* **North Africa:** Egyptian Arabic (`arz`), Moroccan Arabic (`ary`), Algerian Arabic (`arq`), Tunisian Arabic (`aeb`), etc.

---

## Repository Structure

```
├── afri_restorer.py                          # Standalone CPU inference library and CLI tool (FP16 & INT8 ONNX)
├── requirements.txt                         # Project dependencies
├── build_pan_african_mega_corpus.py         # Assembles 3.3M passages across 751 languages from 3 corpora
├── train_mega_corpus_afro_xlmr.py           # 20-epoch training on Davlan/afro-xlmr-base (zero language tags)
├── push_mega_dataset_to_hf.py               # Uploads unified 3.3M dataset to AfriSpeech/afri-speech-pause-aligned
├── modal_server.py                          # Serverless Modal API (FP16 GPU / INT8 ONNX, auto-sleep on idle)
└── paper/                                   # 5-page arXiv preprint LaTeX source and compiled PDF
```

---

## Datasets on Hugging Face

### [AfriSpeech/afri-speech-pause-aligned](https://huggingface.co/datasets/AfriSpeech/afri-speech-pause-aligned)
* **`train` Split:** **3,263,160 multi-sentence passages** (701 MB)
* **`eval` Split:** **42,095 balanced passages** (7.2 MB, stratified across all 751 languages)
* **Strict Narrative Context (Not Randomly Joined):**
  * Grouped strictly within the same book and chapter into 2–3 sentence paragraphs.
  * Filtered of English contamination.
  * **Zero Audio Overhead:** Exclusively text-only parallel data.

```python
from datasets import load_dataset

ds = load_dataset("AfriSpeech/afri-speech-pause-aligned")
print(ds["train"])  # 3,263,160 rows
print(ds["eval"])   # 42,095 rows
```

---

## Non-Latin Scripts & Multi-Script Coverage

The pipeline natively supports non-Latin African writing systems with **0 out-of-vocabulary (`<unk>`) tokens**:
* **Ethiopic / Ge'ez (`ፊደል`)**: Amharic (`amh`), Tigrinya (`tir`) — restores Ethiopic full-stops (`።`), semicolons (`፤`), and commas (`፣`).
* **Arabic Script**: Egyptian (`arz`), Moroccan (`ary`), Algerian (`arq`), Tunisian (`aeb`), Hausa Ajami — restores Arabic commas (`،`) and question marks (`؟`).
* **Latin with African Phonetic Extensions**: Akan/Twi (`aka`), Ewe (`ewe`), Yoruba (`yor`), Hausa (`hau`), etc. — handles open vowels (`ɛ`, `ɔ`), tone marks, and hooked consonants (`ɗ`, `ƙ`, `ɓ`).

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
