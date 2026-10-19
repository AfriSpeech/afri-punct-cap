#!/bin/bash
set -e

cd /mnt/volume_d2wey28/projects/afri-punct-cap
echo "Waiting for build_pan_african_mega_corpus.py to finish..."

while pgrep -f "build_pan_african_mega_corpus.py" > /dev/null; do
    sleep 10
done

echo "Mega corpus build complete! Checking files..."
ls -lh data/mega_african_*.parquet

echo "Launching train_mega_corpus_afro_xlmr.py on H200 GPU..."
.venv/bin/python -u train_mega_corpus_afro_xlmr.py > train_mega.log 2>&1

echo "Mega pipeline finished successfully!"
