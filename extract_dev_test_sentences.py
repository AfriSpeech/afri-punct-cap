"""
Extract clean, held-out test sentences from Afri2Universal split=dev across all languages.
Produces demo_test_sentences.json mapping language codes/names to decapped/depunctuated input pairs.
"""

import os
import re
import glob
import json
import unicodedata
import pyarrow.parquet as pq

dev_dir = "/mnt/volume_d2wey28/projects/afripa-asr/data/corpus-univ/version=0/corpus=afripa/split=dev"
out_json = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/demo_test_sentences.json"

PUNCT_REGEX = re.compile(r'[,.?!;:—\-\"“”‘’،؟።፤፡\'\(\)\[\]«»]+')

def strip_punctuation_and_lower(text):
    text = unicodedata.normalize("NFC", text)
    clean = PUNCT_REGEX.sub(" ", text)
    return " ".join(clean.lower().split())

def main():
    lang_dirs = sorted(glob.glob(os.path.join(dev_dir, "language=*")))
    print(f"Scanning {len(lang_dirs)} dev language directories...")

    demo_data = {}

    for d in lang_dirs:
        lang_code = os.path.basename(d).split("=")[1]
        p_file = os.path.join(d, "part-0.parquet")
        if not os.path.exists(p_file):
            continue

        try:
            df = pq.read_table(p_file, columns=["transcript"]).to_pandas()
        except Exception:
            continue

        candidates = [t.strip() for t in df["transcript"].dropna() if len(t.strip().split()) >= 8 and len(t.strip().split()) <= 40]
        if not candidates:
            continue

        # Pick candidate that has punctuation (comma or period)
        chosen = None
        for c in candidates:
            if any(p in c for p in [",", "،", "፣", ".", "።"]):
                chosen = c
                break
        if not chosen:
            chosen = candidates[0]

        decapped_depunctuated = strip_punctuation_and_lower(chosen)

        demo_data[lang_code] = {
            "lang": lang_code,
            "raw_input": decapped_depunctuated,
            "ground_truth": chosen,
            "num_words": len(decapped_depunctuated.split())
        }

    print(f"Extracted held-out test sentences for {len(demo_data)} languages!")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(demo_data, f, ensure_ascii=False, indent=2)

    print(f"Saved to {out_json} ({os.path.getsize(out_json)/1e3:.1f} KB)")

if __name__ == "__main__":
    main()
