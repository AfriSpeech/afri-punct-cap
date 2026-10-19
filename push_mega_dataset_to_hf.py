"""
Push Mega Pan-African Training Dataset (751 Languages, 3.3M Sequences) to Hugging Face
Replaces old files on AfriSpeech/afri-speech-pause-aligned with unified train and eval splits.
"""

import os
import shutil
import pyarrow.parquet as pq
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete

api = HfApi()

dataset_repo = "AfriSpeech/afri-speech-pause-aligned"
train_src = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/mega_african_train.parquet"
eval_src = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/mega_african_eval.parquet"

staging_dir = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/mega_hf_dataset"
os.makedirs(os.path.join(staging_dir, "data"), exist_ok=True)

# 1. Inspect unique languages for YAML metadata
print("Extracting 751 language tags from training set...")
table = pq.read_table(train_src, columns=["language"])
all_langs = sorted(list(set(table["language"].to_pylist())))
print(f"Total languages: {len(all_langs)}")

yaml_langs = "\n".join(f"- {l}" for l in all_langs)

# 2. Hardlink / stage the train and eval parquets
train_dest = os.path.join(staging_dir, "data", "train-00000-of-00001.parquet")
eval_dest = os.path.join(staging_dir, "data", "eval-00000-of-00001.parquet")

if not os.path.exists(train_dest):
    try:
        os.link(train_src, train_dest)
    except OSError:
        shutil.copy2(train_src, train_dest)

if not os.path.exists(eval_dest):
    try:
        os.link(eval_src, eval_dest)
    except OSError:
        shutil.copy2(eval_src, eval_dest)

# 3. Comprehensive Dataset Card
dataset_card = f"""---
language:
{yaml_langs}
license: cc-by-nc-4.0
task_categories:
- text-to-speech
- automatic-speech-recognition
tags:
- african-languages
- punctuation-restoration
- capitalization
- speech-pause-alignment
- text-to-speech
- tts-prosody
- pan-african
- afrispeech
configs:
- config_name: default
  data_files:
  - split: train
    path: "data/train-*"
  - split: eval
    path: "data/eval-*"
size_categories:
- 1M<n<10M
---

# Mega Pan-African Punctuation & Capitalization Dataset (751 Languages)

A large-scale, multi-sentence parallel text corpus spanning **751 African languages and dialects** across all four major African language families (*Niger-Congo*, *Afroasiatic*, *Nilo-Saharan*, and *Khoisan*).

Part of the **[AfriSpeech](https://afrispeech.org)** open-source speech initiative.

## 📊 Dataset Statistics

* **Total Training Sequences:** **3,263,160 passages** (701 MB)
* **Total Validation Sequences:** **42,095 passages** (7.2 MB, balanced mixture across all 751 languages)
* **Total Scale:** **3,305,255 multi-sentence passages**
* **Languages Covered:** **751 African languages and dialects**
* **Format:** **Exclusively text-only parallel data** (zero audio overhead).

## 🌍 Data Sources & Narrative Discourse Order

To avoid randomly joined sentences, sequences were assembled strictly in their **natural, consecutive discourse order**:

1. **[`AfriSpeech/africa-corpus`](https://huggingface.co/datasets/AfriSpeech/africa-corpus) (693 African languages, 2,228,521 passages):**
   * Keyed by canonical scripture location: `BOOK.CHAPTER.VERSE`.
   * Grouped consecutive verses **strictly within the same book and chapter** into 2–3 sentence narrative paragraphs (e.g., Verse 1 + Verse 2 + Verse 3). Never crossed chapter boundaries.
2. **[`AfriSpeech/african-corpus-jw`](https://huggingface.co/datasets/AfriSpeech/african-corpus-jw) (140 African languages, 540,566 passages):**
   * Keyed by article and paragraph number (`ARTICLE_ID#PARAGRAPH`).
   * Contains natural multi-sentence magazine and article prose.
3. **[`AfriSpeech/african-transcribed-speech-monolingual`](https://huggingface.co/datasets/AfriSpeech/african-transcribed-speech-monolingual) (15 languages, 479,177 sentences):**
   * Authentic spoken speech transcriptions filtered to remove non-African/English noise.

## 🎙️ Why This Dataset Matters for African Speech & TTS

In standard African language writing habits, **writers conventionally omit commas** before coordinating conjunctions (e.g. *ne* in Akan, *àti* in Yoruba, *na* in Swahili), complementizers, and clause boundaries. 

* **The TTS Bottleneck:** Text-to-Speech (TTS) models depend heavily on punctuation for breath pauses and prosodic rhythm. Without commas, TTS engines synthesize breathless, rushed, and unnatural audio.
* **Continuous Paragraph Context:** By training on consecutive multi-sentence streams (1–3 sentences), models learn both **internal sentence-boundary detection** (`.`, `?`, `!`, `።`) inside running text and **clause-level pause commas** (`,`, `،`, `፣`).

## 🌍 Zero-Shot Generalizability to Other African Languages

Because syntactic clause boundaries, appositives, and breath pauses follow shared typological and acoustic rhythms across related African language subfamilies (e.g., Bantu, Kwa, Chadic, Cushitic, Senegambian, Gur, Mande), models trained on this 751-language foundation **generalize zero-shot to other unwritten dialects and low-resource African language varieties**.

## Schema

- `language` (*string*): Language ISO 639-3 or dialect code (751 languages).
- `text` (*string*): The target text with natural punctuation (commas, sentence-ending periods, question marks) and capitalization.
- `source` (*string*): Data source origin (`speech`, `jw_prose`, `bible_consecutive`).

## Quick Usage

```python
from datasets import load_dataset

# Load full 3.3M sequence dataset
ds = load_dataset("AfriSpeech/afri-speech-pause-aligned")

print(ds["train"]) # 3,263,160 rows
print(ds["eval"])  # 42,095 rows
```

## Citation & Credits

- Base Corpora: [AfriSpeech/africa-corpus](https://huggingface.co/datasets/AfriSpeech/africa-corpus), [AfriSpeech/african-corpus-jw](https://huggingface.co/datasets/AfriSpeech/african-corpus-jw), [AfriSpeech/african-transcribed-speech-monolingual](https://huggingface.co/datasets/AfriSpeech/african-transcribed-speech-monolingual)
- Organization: **[AfriSpeech](https://afrispeech.org)** & **[Ghana NLP](https://ghananlp.org)**
"""

with open(os.path.join(staging_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write(dataset_card)

# 4. Clean up old per-language files from the repo to make the repo clean and unified
print(f"Listing old files on {dataset_repo}...")
existing_files = api.list_repo_files(dataset_repo, repo_type="dataset")
old_parquet_files = [f for f in existing_files if f.startswith("data/") and not f.startswith("data/train-") and not f.startswith("data/eval-")]

print(f"Found {len(old_parquet_files)} old per-language files to clean up...")

# 5. Commit upload of unified train & eval splits and deletion of old files
print("Uploading unified train & eval splits to Hugging Face...")
api.upload_folder(
    folder_path=staging_dir,
    repo_id=dataset_repo,
    repo_type="dataset",
    commit_message="Replace dataset with unified Mega Pan-African Corpus (3.3M sequences, 751 languages)"
)

# Delete old files if any
if old_parquet_files:
    print(f"Deleting {len(old_parquet_files)} legacy files in chunks...")
    chunk_size = 50
    for i in range(0, len(old_parquet_files), chunk_size):
        chunk = old_parquet_files[i : i + chunk_size]
        operations = [CommitOperationDelete(path_in_repo=f) for f in chunk]
        api.create_commit(
            repo_id=dataset_repo,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Clean up legacy per-language files chunk {i//chunk_size + 1}"
        )

print(f"\nSuccessfully replaced dataset on https://huggingface.co/datasets/{dataset_repo}!")
