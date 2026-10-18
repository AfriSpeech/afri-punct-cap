import io
import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle
import soundfile as sf
import pyarrow.parquet as pq
import uroman

u = uroman.Uroman()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Loading MMS_FA model on {device}...")
model = bundle.get_model().to(device)
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()
dictionary = bundle.get_dict()
valid_chars = set(c for c in dictionary if c != "-")

print("Testing MMS alignment across African languages...\n")

for lang in ["yor", "hau", "swh", "aka"]:
    path = f"/mnt/volume_d2wey28/projects/afripa-asr/data/corpus-univ/version=0/corpus=afripa/split=train/language={lang}/part-0.parquet"
    t = pq.read_table(path)
    df = t.to_pandas()
    row = df.iloc[0]

    audio_arr, sr = sf.read(io.BytesIO(row["audio_bytes"]))
    waveform = torch.tensor(audio_arr, dtype=torch.float32).unsqueeze(0).to(device)
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    orig_words = row["transcript"].split()
    cleaned_words = []
    valid_indices = []
    for i, w in enumerate(orig_words):
        rom = u.romanize_string(w)
        cw = "".join([c for c in rom.lower() if c in valid_chars])
        if cw:
            cleaned_words.append(cw)
            valid_indices.append(i)

    if len(cleaned_words) < 2:
        continue

    tokens = tokenizer(cleaned_words)
    with torch.no_grad():
        emission, _ = model(waveform)
    token_spans = aligner(emission[0].cpu(), tokens)

    num_frames = emission.shape[1]
    duration = waveform.shape[1] / bundle.sample_rate
    frame_to_sec = duration / num_frames

    word_spans = []
    for orig_idx, spans in zip(valid_indices, token_spans):
        if spans:
            word_spans.append({
                "idx": orig_idx,
                "word": orig_words[orig_idx],
                "start": spans[0].start * frame_to_sec,
                "end": spans[-1].end * frame_to_sec
            })

    pauses = []
    for i in range(len(word_spans) - 1):
        gap = word_spans[i+1]["start"] - word_spans[i]["end"]
        if gap >= 0.35:
            pauses.append((gap, word_spans[i]["word"], word_spans[i+1]["word"]))

    print(f"[{lang.upper()}] Audio: {duration:.2f}s | Words: {len(orig_words)} | Aligned: {len(word_spans)} | Pauses (>=0.35s): {len(pauses)}")
    for gap, w1, w2 in pauses[:3]:
        print(f"   {gap:.2f}s pause after '{w1}' -> '{w2}'")
    print()
