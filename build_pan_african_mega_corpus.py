"""
Build Mega Pan-African Punctuation & Capitalization Corpus
Aggregates:
1. AfriSpeech/africa-corpus (693 African languages, chapter-consecutive verses)
2. AfriSpeech/african-corpus-jw (129 African languages, consecutive paragraph prose)
3. AfriSpeech/african-transcribed-speech-monolingual (15 languages, transcribed speech)

Features:
- Spans 700+ African languages and dialects.
- Strict consecutive narrative flow (grouped by book/chapter/article).
- English contamination filtering.
- Stratified balanced validation split across all languages.
- Zero language conditioning.
"""

import os
import re
import csv
import time
import random
import concurrent.futures
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

out_dir = "/mnt/volume_d2wey28/projects/afri-punct-cap/data"
os.makedirs(out_dir, exist_ok=True)

ENG_WORDS = {"the", "and", "of", "to", "in", "is", "you", "that", "it", "he", "was", "for", "on", "are", "as", "with", "his", "they", "at", "be", "this", "have", "from", "one", "by", "word", "see", "while", "we", "unto", "shall", "lord"}

def is_english(text):
    words = set(re.findall(r"[a-z]+", text.lower()))
    matches = len(words.intersection(ENG_WORDS))
    return matches >= 3 and (matches / max(1, len(words))) > 0.25

def clean_word(tok):
    return re.sub(r'^[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+|[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+$', '', tok)

def download_file_safe(repo_id, fname):
    try:
        return hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
    except Exception:
        return None

def main():
    api = HfApi()
    all_train_passages = []
    all_eval_passages = []

    # =========================================================================
    # 1. AfriSpeech/african-transcribed-speech-monolingual (15 languages)
    # =========================================================================
    print("=== Source 1: Loading african-transcribed-speech-monolingual ===")
    t0 = time.time()
    speech_langs = ["amh", "efi", "fat", "ibo", "kea", "kin", "lin", "mlg", "sna", "swa", "tsn", "twi", "xho", "yor", "zul"]

    for lang in speech_langs:
        try:
            ds = load_dataset("AfriSpeech/african-transcribed-speech-monolingual", lang, split="train")
            ds_eval = load_dataset("AfriSpeech/african-transcribed-speech-monolingual", lang, split="eval")

            train_texts = [r["text"].strip() for r in ds if r.get("text") and not is_english(r["text"]) and len(r["text"].split()) >= 4]
            eval_texts = [r["text"].strip() for r in ds_eval if r.get("text") and not is_english(r["text"]) and len(r["text"].split()) >= 4]

            for t in train_texts:
                all_train_passages.append({"language": lang, "text": t, "source": "speech"})
            for t in eval_texts[:100]:
                all_eval_passages.append({"language": lang, "text": t, "source": "speech"})

            print(f"[{lang.upper()}] Transcribed speech: {len(train_texts):,} train, {min(100, len(eval_texts))} eval")
        except Exception as e:
            print(f"[{lang.upper()}] Error: {e}")

    print(f"Source 1 done in {time.time()-t0:.1f}s: {len(all_train_passages):,} passages.\n")

    # =========================================================================
    # 2. AfriSpeech/african-corpus-jw (129 African Languages)
    # =========================================================================
    print("=== Source 2: Loading african-corpus-jw (Watchtower literature) ===")
    t1 = time.time()
    jw_files = [f for f in api.list_repo_files("AfriSpeech/african-corpus-jw", repo_type="dataset") if f.endswith(".csv")]
    print(f"Downloading {len(jw_files)} language files from african-corpus-jw in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(download_file_safe, "AfriSpeech/african-corpus-jw", f): f for f in jw_files}
        downloaded_jw = {}
        for fut in concurrent.futures.as_completed(futs):
            fname = futs[fut]
            path = fut.result()
            if path:
                downloaded_jw[fname] = path

    print(f"Downloaded {len(downloaded_jw)} JW files in {time.time()-t1:.1f}s. Parsing paragraphs...")

    jw_count = 0
    for fname, path in downloaded_jw.items():
        lang = fname.split("_")[-2] if len(fname.split("_")) >= 3 else fname.split(".")[0]
        try:
            df = pd.read_csv(path)
            col = "local" if "local" in df.columns else ("text" if "text" in df.columns else None)
            if not col:
                continue

            # Each row is already a paragraph of natural prose
            paragraphs = [str(t).strip() for t in df[col].dropna() if len(str(t).strip().split()) >= 8 and not is_english(str(t))]
            if not paragraphs:
                continue

            random.seed(42)
            random.shuffle(paragraphs)

            # Limit per language to 4,000 for balanced distribution
            paragraphs = paragraphs[:4000]
            val_n = min(50, int(len(paragraphs) * 0.05))

            for p in paragraphs[:-val_n]:
                all_train_passages.append({"language": lang, "text": p, "source": "jw_prose"})
                jw_count += 1
            for p in paragraphs[-val_n:]:
                all_eval_passages.append({"language": lang, "text": p, "source": "jw_prose"})

        except Exception:
            continue

    print(f"Source 2 done: added {jw_count:,} paragraphs across {len(downloaded_jw)} languages in {time.time()-t1:.1f}s.\n")

    # =========================================================================
    # 3. AfriSpeech/africa-corpus (693 African Languages)
    # =========================================================================
    print("=== Source 3: Loading africa-corpus (693 languages, chapter-consecutive verses) ===")
    t2 = time.time()
    bible_files = [f for f in api.list_repo_files("AfriSpeech/africa-corpus", repo_type="dataset") if f.endswith(".csv") and not f.startswith("reference_caches")]
    print(f"Downloading {len(bible_files)} language files from africa-corpus in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(download_file_safe, "AfriSpeech/africa-corpus", f): f for f in bible_files}
        downloaded_bible = {}
        for fut in concurrent.futures.as_completed(futs):
            fname = futs[fut]
            path = fut.result()
            if path:
                downloaded_bible[fname] = path

    print(f"Downloaded {len(downloaded_bible)} files in {time.time()-t2:.1f}s. Assembling consecutive passages...")

    bible_count = 0
    # Group by language code to deduplicate multiple translations per language
    lang_to_files = {}
    for fname, path in downloaded_bible.items():
        parts = fname.replace(".csv", "").split("_")
        lang_code = parts[-2] if len(parts) >= 3 else parts[0]
        lang_to_files.setdefault(lang_code, []).append(path)

    print(f"Processing {len(lang_to_files)} unique language codes from africa-corpus...")

    for lang_code, paths in lang_to_files.items():
        lang_passages = []
        for path in paths[:2]: # Max 2 versions per language
            try:
                df = pd.read_csv(path)
                if "verse_key" not in df.columns or "local" not in df.columns:
                    continue

                # Group strictly by book & chapter (e.g. GEN.8.1 -> GEN.8)
                df["chapter_key"] = df["verse_key"].apply(lambda k: ".".join(str(k).split(".")[:2]) if "." in str(k) else str(k))
                
                for _, group in df.groupby("chapter_key"):
                    verses = [str(v).strip() for v in group["local"].dropna() if len(str(v).strip().split()) >= 3 and not is_english(str(v))]
                    i = 0
                    while i < len(verses):
                        pack = 2 if (i % 2 == 0) else 3
                        passage = " ".join(verses[i : i + pack])
                        i += pack
                        if len(passage.split()) >= 8:
                            lang_passages.append(passage)
            except Exception:
                continue

        if not lang_passages:
            continue

        random.seed(42)
        random.shuffle(lang_passages)

        # Cap at 3,500 passages per language for balanced distribution
        lang_passages = lang_passages[:3500]
        val_n = min(50, int(len(lang_passages) * 0.05))

        for p in lang_passages[:-val_n]:
            all_train_passages.append({"language": lang_code, "text": p, "source": "bible_consecutive"})
            bible_count += 1
        for p in lang_passages[-val_n:]:
            all_eval_passages.append({"language": lang_code, "text": p, "source": "bible_consecutive"})

    print(f"Source 3 done: added {bible_count:,} passages across {len(lang_to_files)} languages in {time.time()-t2:.1f}s.\n")

    # =========================================================================
    # Final Dataset Compilation
    # =========================================================================
    print("=== Final Dataset Compilation & Parquet Export ===")
    print(f"Total Combined Training Passages:   {len(all_train_passages):,}")
    print(f"Total Balanced Validation Passages: {len(all_eval_passages):,}")

    unique_langs = len(set(r["language"] for r in all_train_passages))
    print(f"Unique African languages represented: {unique_langs}")

    # Shuffle training set
    random.seed(42)
    random.shuffle(all_train_passages)

    train_path = os.path.join(out_dir, "mega_african_train.parquet")
    print(f"Exporting train set to {train_path}...")
    t_train = pa.Table.from_pandas(pd.DataFrame(all_train_passages))
    pq.write_table(t_train, train_path, compression="snappy")
    print(f"Train Parquet size: {os.path.getsize(train_path)/1e6:.1f} MB")

    eval_path = os.path.join(out_dir, "mega_african_eval.parquet")
    print(f"Exporting eval set to {eval_path}...")
    t_eval = pa.Table.from_pandas(pd.DataFrame(all_eval_passages))
    pq.write_table(t_eval, eval_path, compression="snappy")
    print(f"Eval Parquet size: {os.path.getsize(eval_path)/1e6:.1f} MB")

    print(f"\nMega-Corpus Build Complete! Ready for Pan-African training across {unique_langs} languages.")

if __name__ == "__main__":
    main()
