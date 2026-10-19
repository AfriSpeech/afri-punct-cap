"""
Export and Push Multi-Sentence Training Dataset to Hugging Face
Builds the 138,119 multi-sentence training sequences (1-3 sentences packed)
and pushes data/multi_sentence_train.parquet to AfriSpeech/afri-speech-pause-aligned.
"""

import os
import re
import glob
import time
import random
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

aligned_dir = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/aligned/split=train"
out_parquet = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/multi_sentence_train.parquet"
dataset_repo = "AfriSpeech/afri-speech-pause-aligned"

api = HfApi()

def clean_word(tok):
    return re.sub(r'^[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+|[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+$', '', tok)

def main():
    print(f"Loading 314 language parquet files from {aligned_dir}...")
    parquet_files = sorted(glob.glob(os.path.join(aligned_dir, "*.parquet")))
    print(f"Found {len(parquet_files)} language files.")

    rows = []
    total_single = 0
    t0 = time.time()
    random.seed(42)

    for p_file in parquet_files:
        lang = os.path.basename(p_file).replace(".parquet", "")
        df = pq.read_table(p_file).to_pandas()
        sentences = [t.strip() for t in df["comma_fixed_transcript"].dropna() if len(t.strip()) > 5]
        total_single += len(sentences)

        i = 0
        while i < len(sentences):
            r = random.random()
            if r < 0.25:
                pack_size = 1
            elif r < 0.75:
                pack_size = 2
            else:
                pack_size = 3

            group = sentences[i : i + pack_size]
            i += pack_size

            target_text = " ".join(group)
            
            # Generate unpunctuated, lowercased input text with language prefix
            tokens = target_text.split()
            cleaned_tokens = [clean_word(t).lower() for t in tokens if clean_word(t)]
            if len(cleaned_tokens) < 3:
                continue

            input_text = f"[{lang}] " + " ".join(cleaned_tokens)
            
            # Count punctuation in target
            num_commas = target_text.count(",") + target_text.count("،") + target_text.count("፣")
            num_periods = target_text.count(".") + target_text.count("።") + target_text.count(";") + target_text.count("፤")

            rows.append({
                "language": lang,
                "input_text": input_text,
                "target_text": target_text,
                "num_sentences": len(group),
                "num_words": len(cleaned_tokens),
                "num_commas": num_commas,
                "num_periods": num_periods,
            })

    print(f"Built {len(rows):,} multi-sentence training rows from {total_single:,} sentences in {time.time()-t0:.1f}s.")
    
    # Save as parquet
    print(f"Saving to {out_parquet}...")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_parquet, compression="snappy")
    file_size_mb = os.path.getsize(out_parquet) / 1e6
    print(f"Saved {out_parquet} ({file_size_mb:.1f} MB)!")

    # Push to Hugging Face
    print(f"Pushing multi_sentence_train.parquet to {dataset_repo}...")
    api.upload_file(
        path_or_fileobj=out_parquet,
        path_in_repo="data/multi_sentence_train.parquet",
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message="Add multi-sentence (1-3 sentences packed) training dataset across 314 languages"
    )
    print(f"Successfully uploaded to https://huggingface.co/datasets/{dataset_repo}/blob/main/data/multi_sentence_train.parquet!")

if __name__ == "__main__":
    main()
