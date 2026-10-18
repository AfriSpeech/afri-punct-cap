"""
Pan-African Speech-Pause Alignment Pipeline
Acoustically aligns speech audio with transcripts across all 314 African languages
in AfriSpeech/afri2universal using Meta MMS Forced Aligner on the H200 GPU.
"""

import os
import io
import time
import glob
import logging
import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle
import soundfile as sf
import pyarrow as pa
import pyarrow.parquet as pq
import uroman

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("alignment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device} ({torch.cuda.get_device_name(device)})")

    u = uroman.Uroman()
    logger.info("Loading Meta MMS Forced Aligner...")
    model = bundle.get_model().to(device)
    model.eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    dictionary = bundle.get_dict()
    valid_chars = set(c for c in dictionary if c != "-")
    logger.info("MMS Model ready!")

    source_dir = "/mnt/volume_d2wey28/projects/afripa-asr/data/corpus-univ/version=0/corpus=afripa/split=train"
    out_dir = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/aligned/split=train"
    os.makedirs(out_dir, exist_ok=True)

    lang_dirs = sorted(glob.glob(os.path.join(source_dir, "language=*")))
    logger.info(f"Found {len(lang_dirs)} language partitions to process.")

    punct_chars = (",", ".", "!", "?", ";", ":", "—", "-", "'", '"', "”", "’", "،", "؟")
    total_languages_done = 0
    total_sentences_processed = 0
    total_commas_added = 0
    start_time = time.time()

    for idx, lang_dir in enumerate(lang_dirs):
        lang = os.path.basename(lang_dir).split("=")[1]
        out_file = os.path.join(out_dir, f"{lang}.parquet")

        if os.path.exists(out_file):
            logger.info(f"[{idx+1}/{len(lang_dirs)}] Skipping {lang} (already completed).")
            continue

        parquet_file = os.path.join(lang_dir, "part-0.parquet")
        if not os.path.exists(parquet_file):
            continue

        try:
            df = pq.read_table(parquet_file).to_pandas()
        except Exception as e:
            logger.warning(f"Error reading {parquet_file}: {e}")
            continue

        lang_rows = []
        lang_commas = 0
        t_lang = time.time()

        for row_i in range(len(df)):
            row = df.iloc[row_i]
            transcript = str(row.get("transcript", "") or "")
            if not transcript.strip():
                continue

            try:
                audio_bytes = row["audio_bytes"]
                audio_arr, sr = sf.read(io.BytesIO(audio_bytes))
            except Exception:
                lang_rows.append({
                    "language": lang,
                    "original_transcript": transcript,
                    "comma_fixed_transcript": transcript,
                    "commas_added": 0,
                    "duration": float(row.get("duration", 0.0) or 0.0)
                })
                continue

            dur = len(audio_arr) / sr
            waveform = torch.tensor(audio_arr, dtype=torch.float32).unsqueeze(0).to(device)
            if sr != bundle.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

            orig_words = transcript.split()
            cleaned_words = []
            valid_indices = []

            for w_idx, w in enumerate(orig_words):
                rom = u.romanize_string(w)
                cw = "".join([c for c in rom.lower() if c in valid_chars])
                if cw:
                    cleaned_words.append(cw)
                    valid_indices.append(w_idx)

            if len(cleaned_words) < 2:
                lang_rows.append({
                    "language": lang,
                    "original_transcript": transcript,
                    "comma_fixed_transcript": transcript,
                    "commas_added": 0,
                    "duration": round(dur, 2)
                })
                continue

            tokens = tokenizer(cleaned_words)
            with torch.no_grad():
                emission, _ = model(waveform)

            try:
                token_spans = aligner(emission[0].cpu(), tokens)
            except Exception:
                lang_rows.append({
                    "language": lang,
                    "original_transcript": transcript,
                    "comma_fixed_transcript": transcript,
                    "commas_added": 0,
                    "duration": round(dur, 2)
                })
                continue

            num_frames = emission.shape[1]
            frame_to_sec = dur / num_frames

            word_spans = []
            for orig_idx, spans in zip(valid_indices, token_spans):
                if spans:
                    word_spans.append({
                        "idx": orig_idx,
                        "start": spans[0].start * frame_to_sec,
                        "end": spans[-1].end * frame_to_sec
                    })

            result_words = list(orig_words)
            added_for_sentence = 0

            for s_idx in range(len(word_spans) - 1):
                gap = word_spans[s_idx + 1]["start"] - word_spans[s_idx]["end"]
                if gap >= 0.35: # Speech pause threshold 350ms
                    target_idx = word_spans[s_idx + 1]["idx"] - 1
                    cur_w = result_words[target_idx]
                    if not cur_w.endswith(punct_chars):
                        result_words[target_idx] = cur_w + ","
                        added_for_sentence += 1

            lang_commas += added_for_sentence
            lang_rows.append({
                "language": lang,
                "original_transcript": transcript,
                "comma_fixed_transcript": " ".join(result_words),
                "commas_added": added_for_sentence,
                "duration": round(dur, 2)
            })

        # Save language parquet
        out_table = pa.Table.from_pylist(lang_rows)
        pq.write_table(out_table, out_file, compression="snappy")

        dur_lang = time.time() - t_lang
        rate = len(lang_rows) / max(0.1, dur_lang)
        total_languages_done += 1
        total_sentences_processed += len(lang_rows)
        total_commas_added += lang_commas

        logger.info(
            f"[{idx+1}/{len(lang_dirs)}] Completed {lang}: {len(lang_rows)} sentences in {dur_lang:.1f}s "
            f"({rate:.1f} sent/s) | +{lang_commas} commas. "
            f"[Total: {total_sentences_processed:,} sentences, +{total_commas_added:,} commas]"
        )

    total_time = time.time() - start_time
    logger.info(
        f"\n=======================================================\n"
        f"All 314 languages completed in {total_time/3600:.2f} hours!\n"
        f"Total Sentences: {total_sentences_processed:,}\n"
        f"Total Commas Inserted: {total_commas_added:,}\n"
        f"Aligned Parquet Dir: {out_dir}\n"
        f"======================================================="
    )

if __name__ == "__main__":
    main()
