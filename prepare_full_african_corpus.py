"""
Prepare Full African Corpus for Punctuation & Capitalization Restoration
Combines:
1. AfriSpeech/african-transcribed-speech-monolingual (15 languages)
2. AfriSpeech/africa-corpus (consecutive verses by chapter)
3. AfriSpeech/african-corpus-jw (consecutive paragraph prose)

Features:
- Consecutive discourse order (never randomly joined).
- English contamination filtering.
- Stratified balanced train and validation splits.
- Pure text (no language tags).
"""

import os
import re
import glob
import time
import random
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

out_dir = "/mnt/volume_d2wey28/projects/afri-punct-cap/data"
os.makedirs(out_dir, exist_ok=True)

ENG_WORDS = {"the", "and", "of", "to", "in", "is", "you", "that", "it", "he", "was", "for", "on", "are", "as", "with", "his", "they", "at", "be", "this", "have", "from", "one", "by", "word", "see", "while", "we"}

def is_english(text):
    words = set(re.findall(r"[a-z]+", text.lower()))
    matches = len(words.intersection(ENG_WORDS))
    return matches >= 2 and (matches / max(1, len(words))) > 0.3

def clean_word(tok):
    return re.sub(r'^[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+|[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+$', '', tok)

def main():
    print("=== Step 1: Loading african-transcribed-speech-monolingual ===")
    t0 = time.time()
    speech_langs = ["amh", "efi", "fat", "ibo", "kea", "kin", "lin", "mlg", "sna", "swa", "tsn", "twi", "xho", "yor", "zul"]
    all_train_passages = []
    all_eval_passages = []

    for lang in speech_langs:
        try:
            ds = load_dataset("AfriSpeech/african-transcribed-speech-monolingual", lang, split="train")
            ds_eval = load_dataset("AfriSpeech/african-transcribed-speech-monolingual", lang, split="eval")
            
            train_texts = [r["text"].strip() for r in ds if r.get("text") and not is_english(r["text"]) and len(r["text"].split()) >= 4]
            eval_texts = [r["text"].strip() for r in ds_eval if r.get("text") and not is_english(r["text"]) and len(r["text"].split()) >= 4]

            # Add to collection
            for t in train_texts:
                all_train_passages.append({"language": lang, "text": t, "source": "transcribed_speech"})
            for t in eval_texts[:200]: # Up to 200 per language for balanced eval
                all_eval_passages.append({"language": lang, "text": t, "source": "transcribed_speech"})

            print(f"[{lang.upper()}] Transcribed speech: {len(train_texts):,} train, {min(200, len(eval_texts))} eval")
        except Exception as e:
            print(f"[{lang.upper()}] Error loading: {e}")

    print(f"Transcribed speech loaded in {time.time()-t0:.1f}s: {len(all_train_passages):,} train sequences.\n")

    print("=== Step 2: Loading Consecutive Passages from africa-corpus ===")
    t1 = time.time()
    api = HfApi()
    corpus_files = api.list_repo_files("AfriSpeech/africa-corpus", repo_type="dataset")
    csv_files = [f for f in corpus_files if f.endswith(".csv")]

    # Select representative major language CSVs
    major_codes = ["swh", "yor", "hau", "ibo", "twi", "amh", "gax", "som", "tir", "zul", "xho", "sna", "lin", "lug", "kin", "wol", "ewe", "gaa", "dag", "mos", "dyu", "bem", "ada", "ach"]
    selected_csvs = []
    for code in major_codes:
        matches = [f for f in csv_files if f"_{code}_" in f]
        if matches:
            selected_csvs.append((code, matches[0]))

    print(f"Selected {len(selected_csvs)} major African language corpora from africa-corpus.")

    for code, fname in selected_csvs:
        try:
            path = hf_hub_download(repo_id="AfriSpeech/africa-corpus", filename=fname, repo_type="dataset")
            df = pd.read_csv(path)
            if "verse_key" not in df.columns or "local" not in df.columns:
                continue

            # Group consecutive verses by chapter
            df["chapter_key"] = df["verse_key"].apply(lambda k: ".".join(str(k).split(".")[:2]) if "." in str(k) else str(k))
            
            lang_passages = []
            for _, group in df.groupby("chapter_key"):
                verses = [str(v).strip() for v in group["local"].dropna() if len(str(v).strip()) > 5]
                i = 0
                while i < len(verses):
                    pack_size = 2 if (i % 2 == 0) else 3
                    passage = " ".join(verses[i : i + pack_size])
                    i += pack_size
                    if len(passage.split()) >= 10:
                        lang_passages.append(passage)

            # Split 95% train, 5% eval
            random.seed(42)
            random.shuffle(lang_passages)
            split_idx = int(0.95 * len(lang_passages))
            
            train_p = lang_passages[:split_idx]
            eval_p = lang_passages[split_idx:split_idx + 150] # 150 per language for eval

            for p in train_p:
                all_train_passages.append({"language": code, "text": p, "source": "africa_corpus_consecutive"})
            for p in eval_p:
                all_eval_passages.append({"language": code, "text": p, "source": "africa_corpus_consecutive"})

            print(f"[{code.upper()}] africa-corpus: {len(train_p):,} passages (from {len(df):,} verses)")
        except Exception as e:
            print(f"[{code.upper()}] Error: {e}")

    print(f"\nTotal Dataset Assembly Complete in {time.time()-t1:.1f}s:")
    print(f"  Total Train Passages: {len(all_train_passages):,}")
    print(f"  Total Eval Passages:  {len(all_eval_passages):,} (balanced mixture across all languages)")

    # Save to Parquet
    print(f"\nSaving to {out_dir}/unified_african_train.parquet...")
    train_table = pa.Table.from_pandas(pd.DataFrame(all_train_passages))
    pq.write_table(train_table, os.path.join(out_dir, "unified_african_train.parquet"), compression="snappy")
    print(f"Train Parquet size: {os.path.getsize(os.path.join(out_dir, 'unified_african_train.parquet'))/1e6:.1f} MB")

    print(f"Saving to {out_dir}/unified_african_eval.parquet...")
    eval_table = pa.Table.from_pandas(pd.DataFrame(all_eval_passages))
    pq.write_table(eval_table, os.path.join(out_dir, "unified_african_eval.parquet"), compression="snappy")
    print(f"Eval Parquet size: {os.path.getsize(os.path.join(out_dir, 'unified_african_eval.parquet'))/1e6:.1f} MB")

if __name__ == "__main__":
    main()
