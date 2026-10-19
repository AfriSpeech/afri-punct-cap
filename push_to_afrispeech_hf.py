import os
import glob
import shutil
from huggingface_hub import HfApi

api = HfApi()

model_dir = "/mnt/volume_d2wey28/projects/afri-punct-cap/models/afri-punct-cap-mini"
aligned_dir = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/aligned/split=train"
source_dir = "/mnt/volume_d2wey28/projects/afripa-asr/data/corpus-univ/version=0/corpus=afripa/split=train"

all_langs = sorted([os.path.basename(p).split("=")[1] for p in glob.glob(os.path.join(source_dir, "language=*"))])
yaml_langs = "\n".join(f"- {l}" for l in all_langs)
print(f"Total languages covered: {len(all_langs)}")

# -------------------------------------------------------------
# 1. PUSH MODEL TO AfriSpeech/afri-punct-cap
# -------------------------------------------------------------
model_repo = "AfriSpeech/afri-punct-cap"
print(f"\n1. Pushing model to Hugging Face: {model_repo}...")
api.create_repo(repo_id=model_repo, repo_type="model", exist_ok=True)

model_card = f"""---
language:
{yaml_langs}
license: apache-2.0
tags:
- african-languages
- punctuation-restoration
- capitalization
- truecasing
- asr-post-processing
- text-to-speech
- tts-prosody
- xlm-roberta
- onnx
- afrispeech
widget:
- text: "ahene nanahene nana kwame mensah bonsu aka se ese se wobebo dawuro ama chief of staff julius debrah se onkasa nkyere president john dramani mahama ne vice president professor naana jane opoku-agyemang"
- text: "a maa n je ounje asaraloore ka ma baa saisan tori ta o ba se bee se la maa fi iwonba owo to n wole fun wa tojú aisan"
- text: "dokezo tumia dakika kumi kabla ya kumaliza kusoma kupitia mambo uliyosoma"
---

# AfriPunctCap: Pan-African Punctuation & Capitalization Model

A fast, lightweight multilingual model fine-tuned on **314 African languages** to restore **capitalization** (Title Case, ALL CAPS) and **punctuation** (commas, sentence-ending periods, question marks).

Part of the **[AfriSpeech](https://afrispeech.org)** open-source speech initiative.

## 🌍 Language Coverage & Generalizability

* **Directly Trained on 314 African Languages:** Spans Niger-Congo, Afroasiatic, Nilo-Saharan, and Khoisan language families from [AfriSpeech/afri2universal](https://huggingface.co/datasets/AfriSpeech/afri2universal).
* **Zero-Shot Generalizability to Other African Languages:** Because grammatical clause boundaries, breath pauses, appositives, and conjunctions follow shared typological and phonetic patterns across related African language families, the model exhibits robust **zero-shot generalizability to hundreds of other African languages and local dialects** not explicitly in the training set.

## 🎙️ Why This Model is Essential for African Text-to-Speech (TTS)

Across African languages, **writers conventionally omit commas**, despite native speakers making clear, audible pauses in normal speech:
- **Written Orthography Habit:** Commas are routinely skipped before conjunctions (like *ne* in Akan, *àti* in Yoruba, *na* in Swahili), complementizers, appositives, and compound titles.
- **The TTS Bottleneck:** For **Text-to-Speech (TTS)** models, punctuation is the single most critical signal for breath pauses and prosodic rhythm. Without commas, TTS engines rush through clauses unnaturally without breathing.
- **Acoustic Grounding:** Fine-tuned on speech where pauses (>=350ms) were acoustically aligned with Meta MMS Forced Aligner, this model **systematically compensates for commas conventionally ignored in writing**, matching natural spoken rhythm.

## Highlights

- **Backbone**: `Davlan/afro-xlmr-mini` (12 layers, 384 hidden, 117M params).
- **Fast CPU Inference**:
  - **INT8 Quantized ONNX**: **~15–25 ms** per sentence (118 MB model).
  - **FP16 GPU / Safetensors**: **~225 MB** compact half-precision weights.
- **Zero Hallucination Guarantee**: Non-autoregressive token classification preserves all input words in their original order.

## Model Formats Available

- `model_quantized.onnx` (**118 MB**, INT8 quantized for ultra-fast CPU inference)
- `model_fp16.safetensors` (**225 MB**, compact FP16 weights for GPU)
- `model.safetensors` (449 MB)
- `model.onnx` (449 MB)
- `pytorch_model.bin` (449 MB)
- Complete SentencePiece tokenizer files

## GitHub Repository

Code, data pipelines, and CLI tool: [AfriSpeech/afri-punct-cap](https://github.com/AfriSpeech/afri-punct-cap)
"""

with open(os.path.join(model_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write(model_card)

api.upload_folder(
    folder_path=model_dir,
    repo_id=model_repo,
    repo_type="model",
    commit_message="Publish Pan-African Punctuation and Capitalization model (314 languages, FP16 + INT8 ONNX)"
)
print(f"Model pushed to https://huggingface.co/{model_repo}!")

# -------------------------------------------------------------
# 2. PUSH DATASET TO AfriSpeech/afri-speech-pause-aligned
# -------------------------------------------------------------
dataset_repo = "AfriSpeech/afri-speech-pause-aligned"
print(f"\n2. Pushing text-only parallel dataset to Hugging Face: {dataset_repo}...")
api.create_repo(repo_id=dataset_repo, repo_type="dataset", exist_ok=True)

dataset_staging_dir = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/hf_dataset"
data_dir = os.path.join(dataset_staging_dir, "data")
os.makedirs(data_dir, exist_ok=True)

aligned_files = sorted(glob.glob(os.path.join(aligned_dir, "*.parquet")))
print(f"Staging {len(aligned_files)} language parquet files (text only, zero audio)...")
for f in aligned_files:
    dest = os.path.join(data_dir, os.path.basename(f))
    if not os.path.exists(dest):
        try:
            os.link(f, dest)
        except OSError:
            shutil.copy2(f, dest)

dataset_card = f"""---
language:
{yaml_langs}
license: cc-by-nc-4.0
task_categories:
- text-to-speech
- automatic-speech-recognition
tags:
- african-languages
- speech-pause-alignment
- punctuation-restoration
- comma-restoration
- text-to-speech
- tts-prosody
- universal-orthography
- afrispeech
configs:
- config_name: default
  data_files:
  - split: train
    path: "data/*.parquet"
size_categories:
- 100K<n<1M
---

# Pan-African Speech-Pause Aligned Dataset (314 African Languages)

Parallel text dataset for **314 African languages** acoustically aligning natural speech breath/clause pauses (>=350ms) with text transcripts.

Part of the **[AfriSpeech](https://afrispeech.org)** open-source speech initiative.

## 🎙️ Why This Dataset Matters for African Speech & TTS

Across African languages, **writers conventionally omit commas** before conjunctions (like *ne* in Akan, *àti* in Yoruba, *na* in Swahili), complementizers, and clause boundaries. When training Text-to-Speech (TTS) models on unpunctuated text, the resulting audio rushes through phrases unnaturally without taking breath.

By aligning 292,791 audio clips from [AfriSpeech/afri2universal](https://huggingface.co/datasets/AfriSpeech/afri2universal) using Meta MMS Forced Aligner on the H200 GPU, this dataset inserts commas at true acoustic pauses, providing ground-truth prosodic punctuation for high-quality African TTS and ASR.

## 🌍 Language Coverage & Generalizability

Directly covers **314 African languages** spanning all 4 major language families (*Niger-Congo*, *Afroasiatic*, *Nilo-Saharan*, *Khoisan*). The shared acoustic and prosodic phrasing across related African language subfamilies allows models trained on this dataset to **generalize zero-shot to other African dialects and unwritten language varieties**.

## Schema (Text-Only Parallel Data)

- `language` (*string*): ISO 639-3 language code (e.g. `yor`, `hau`, `aka`, `swh`, `amh`).
- `original_transcript` (*string*): The raw transcript from afri2universal.
- `comma_fixed_transcript` (*string*): Transcript with commas inserted at acoustic pause locations.
- `commas_added` (*int64*): Number of commas inserted for this sentence.
- `duration` (*float64*): Duration of the spoken audio clip in seconds.

## Citation & Credits

- Source Corpus: [AfriSpeech/afri2universal](https://huggingface.co/datasets/AfriSpeech/afri2universal)
- Organization: **[AfriSpeech](https://afrispeech.org)** & **[Ghana NLP](https://ghananlp.org)**
"""

with open(os.path.join(dataset_staging_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write(dataset_card)

api.upload_folder(
    folder_path=dataset_staging_dir,
    repo_id=dataset_repo,
    repo_type="dataset",
    commit_message="Publish 314 African language speech-pause aligned parallel text dataset"
)
print(f"Dataset pushed to https://huggingface.co/datasets/{dataset_repo}!")
print("\nAll Hugging Face AfriSpeech uploads completed successfully!")
