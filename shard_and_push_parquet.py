"""
Re-shard Parquet dataset for Hugging Face Dataset Viewer compliance:
1. Max shard size <= 200 MB (~500,000 rows per shard)
2. Small row-group size (row_group_size = 50,000)
3. Deletes single monolithic 1.4GB parquet file from Hugging Face
4. Resolves TooBigContentError
"""

import os
import glob
import math
import pyarrow.parquet as pq
from huggingface_hub import HfApi, CommitOperationDelete

api = HfApi()

dataset_repo = "AfriSpeech/afri-speech-pause-aligned"
staging_dir = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/mega_hf_sharded"
os.makedirs(os.path.join(staging_dir, "data"), exist_ok=True)

source_train_parquet = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/mega_hf_parallel/data/train-00000-of-00001.parquet"
source_eval_parquet = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/mega_hf_parallel/data/eval-00000-of-00001.parquet"

def shard_train_file():
    print(f"Reading {source_train_parquet}...")
    table = pq.read_table(source_train_parquet)
    total_rows = len(table)
    print(f"Total training rows: {total_rows:,}")

    # Aim for ~475,000 rows per shard (~170 MB per file) -> 7 shards
    rows_per_shard = 475_000
    num_shards = math.ceil(total_rows / rows_per_shard)
    print(f"Splitting into {num_shards} shards of ~{rows_per_shard:,} rows...")

    for i in range(num_shards):
        start_idx = i * rows_per_shard
        end_idx = min(total_rows, (i + 1) * rows_per_shard)
        shard_table = table.slice(start_idx, end_idx - start_idx)

        shard_filename = f"train-{i:05d}-of-{num_shards:05d}.parquet"
        shard_dest = os.path.join(staging_dir, "data", shard_filename)

        print(f"  Writing {shard_filename} ({len(shard_table):,} rows, row_group_size=50,000)...")
        pq.write_table(shard_table, shard_dest, compression="snappy", row_group_size=50_000)
        print(f"  Saved {shard_dest} ({os.path.getsize(shard_dest)/1e6:.1f} MB)")

def copy_eval_file():
    print(f"\nProcessing eval split with small row_group_size...")
    eval_table = pq.read_table(source_eval_parquet)
    eval_dest = os.path.join(staging_dir, "data", "eval-00000-of-00001.parquet")
    pq.write_table(eval_table, eval_dest, compression="snappy", row_group_size=10_000)
    print(f"Saved {eval_dest} ({os.path.getsize(eval_dest)/1e6:.1f} MB)")

def copy_readme():
    src_readme = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/mega_hf_parallel/README.md"
    dest_readme = os.path.join(staging_dir, "README.md")
    with open(src_readme, "r", encoding="utf-8") as f:
        content = f.read()
    with open(dest_readme, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Copied README.md to {dest_readme}")

def main():
    shard_train_file()
    copy_eval_file()
    copy_readme()

    # Delete monolithic train file from Hugging Face if present
    print(f"\nChecking for monolithic files to remove on {dataset_repo}...")
    existing = api.list_repo_files(dataset_repo, repo_type="dataset")
    if "data/train-00000-of-00001.parquet" in existing:
        print("Deleting monolithic data/train-00000-of-00001.parquet on Hugging Face...")
        api.delete_file(
            path_in_repo="data/train-00000-of-00001.parquet",
            repo_id=dataset_repo,
            repo_type="dataset",
            commit_message="Remove monolithic un-sharded train parquet"
        )

    # Upload sharded dataset
    print(f"\nUploading sharded dataset to https://huggingface.co/datasets/{dataset_repo}...")
    api.upload_folder(
        folder_path=staging_dir,
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message="Re-shard dataset into <=200MB files with 50k row-group size for HF viewer"
    )
    print(f"Upload complete! Hugging Face Dataset Viewer will now load without scan size limit errors.")

if __name__ == "__main__":
    main()
