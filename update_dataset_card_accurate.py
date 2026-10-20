import os
import pyarrow.parquet as pq
from huggingface_hub import HfApi

api = HfApi()

train_src = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/mega_african_train.parquet"
dataset_repo = "AfriSpeech/afri-speech-pause-aligned"

table = pq.read_table(train_src, columns=["language"])
all_langs = sorted(list(set(table["language"].to_pylist())))
yaml_langs = "\n".join(f"- {l}" for l in all_langs)

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
- parallel-corpus
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
- 1M<n<1M
---

# Mega Pan-African Parallel Punctuation & Capitalization Dataset (751 Languages)

A massive parallel text corpus spanning **751 African languages and dialects** across all four major African language families (*Niger-Congo*, *Afroasiatic*, *Nilo-Saharan*, and *Khoisan*).

Part of the **[AfriSpeech](https://afrispeech.org)** open-source speech initiative.

## 📊 Dataset Parallel Columns

This dataset provides side-by-side parallel columns for evaluating and training punctuation and capitalization models:

- `language` (*string*): Language ISO 639-3 or dialect code (751 languages).
- `input_text` (*string*): **Raw unpunctuated, lowercased text** (as output by ASR speech-to-text systems or unformatted transcripts).
- `target_text` (*string*): **Ground-truth punctuated and capitalized text** (with natural commas, periods, question marks, and truecasing).
- `source` (*string*): Data origin (`speech`, `jw_prose`, `bible_consecutive`).
- `num_words` (*int64*): Word count of the passage.
- `num_commas` (*int64*): Total commas in target.
- `num_periods` (*int64*): Total sentence-ending periods / question marks in target.

## 📈 Dataset Statistics

* **Total Training Sequences:** **3,263,160 parallel pairs** (7 sharded Parquet files, ~200 MB each)
* **Total Validation Sequences:** **42,095 parallel pairs** (~14 MB, balanced mixture across all 751 languages)
* **Total Scale:** **3,305,255 parallel passages**
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
   * Authentic spoken speech transcriptions filtered of English contamination.

## 🎙️ Why This Dataset Matters for African Speech & TTS

In standard African language writing habits, **writers conventionally omit commas** before coordinating conjunctions (e.g. *ne* in Akan, *àti* in Yoruba, *na* in Swahili), complementizers, and clause boundaries. 

* **The TTS Bottleneck:** Text-to-Speech (TTS) models depend heavily on punctuation for breath pauses and prosodic rhythm. Without commas, TTS engines synthesize breathless, rushed, and unnatural audio.
* **Continuous Paragraph Context:** By training on consecutive multi-sentence streams (1–3 sentences), models learn both **internal sentence-boundary detection** (`.`, `?`, `!`, `።`) inside running text and **clause-level commas** (`,`, `،`, `፣`) based on generalized syntactic patterns.

## Quick Usage

```python
from datasets import load_dataset

# Load full 3.3M parallel sequence dataset
ds = load_dataset("AfriSpeech/afri-speech-pause-aligned")

# Access parallel columns
print(ds["train"][0]["input_text"])   # Raw unpunctuated lowercase
print(ds["train"][0]["target_text"])  # Fully punctuated and capitalized
```

## Citation & Credits

- Base Corpora: [AfriSpeech/africa-corpus](https://huggingface.co/datasets/AfriSpeech/africa-corpus), [AfriSpeech/african-corpus-jw](https://huggingface.co/datasets/AfriSpeech/african-corpus-jw), [AfriSpeech/african-transcribed-speech-monolingual](https://huggingface.co/datasets/AfriSpeech/african-transcribed-speech-monolingual)
- Organization: **[AfriSpeech](https://afrispeech.org)** & **[Ghana NLP](https://ghananlp.org)**
"""

api.upload_file(
    path_or_fileobj=dataset_card.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=dataset_repo,
    repo_type="dataset",
    commit_message="Update dataset card: clarify multi-sentence corpus assembly and TTS prosody motivation"
)
print("Dataset card successfully updated on Hugging Face!")
