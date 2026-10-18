import io
import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle
import soundfile as sf
import pyarrow.parquet as pq
import uroman

u = uroman.Uroman()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = bundle.get_model().to(device)
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()
dictionary = bundle.get_dict()
valid_chars = set(c for c in dictionary if c != "-")

punct_chars = (",", ".", "!", "?", ";", ":", "—", "-", "'", '"', "”", "’")

for lang in ["yor", "hau", "swh", "aka"]:
    path = f"/mnt/volume_d2wey28/projects/afripa-asr/data/corpus-univ/version=0/corpus=afripa/split=train/language={lang}/part-0.parquet"
    df = pq.read_table(path).to_pandas().head(10)
    print(f"\n==================== Language: {lang.upper()} ====================")

    for i in range(len(df)):
        row = df.iloc[i]
        audio_arr, sr = sf.read(io.BytesIO(row["audio_bytes"]))
        dur = len(audio_arr) / sr
        waveform = torch.tensor(audio_arr, dtype=torch.float32).unsqueeze(0).to(device)
        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

        orig_words = row["transcript"].split()
        cleaned_words = []
        valid_indices = []
        for idx, w in enumerate(orig_words):
            rom = u.romanize_string(w)
            cw = "".join([c for c in rom.lower() if c in valid_chars])
            if cw:
                cleaned_words.append(cw)
                valid_indices.append(idx)

        if len(cleaned_words) < 2:
            continue

        tokens = tokenizer(cleaned_words)
        with torch.no_grad():
            emission, _ = model(waveform)

        try:
            token_spans = aligner(emission[0].cpu(), tokens)
        except Exception:
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
        new_commas = 0
        for idx in range(len(word_spans) - 1):
            gap = word_spans[idx+1]["start"] - word_spans[idx]["end"]
            if gap >= 0.35:
                target_idx = word_spans[idx+1]["idx"] - 1
                cur_w = result_words[target_idx]
                if not cur_w.endswith(punct_chars):
                    result_words[target_idx] = cur_w + ","
                    new_commas += 1

        fixed = " ".join(result_words)
        if new_commas > 0:
            print(f"\n[Row {i} - {dur:.1f}s] Added {new_commas} commas:")
            print("  ORIG: ", row["transcript"])
            print("  FIXED:", fixed)
