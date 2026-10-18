import io
import time
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

path = "/mnt/volume_d2wey28/projects/afripa-asr/data/corpus-univ/version=0/corpus=afripa/split=train/language=yor/part-0.parquet"
df = pq.read_table(path).to_pandas().head(50)

print("Warming up H200...")
# Warmup
row = df.iloc[0]
audio_arr, sr = sf.read(io.BytesIO(row["audio_bytes"]))
waveform = torch.tensor(audio_arr, dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    _ = model(waveform)

print(f"Benchmarking 50 Yoruba samples on H200...")
t0 = time.time()
processed = 0

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
        processed += 1
    except Exception:
        pass

elapsed = time.time() - t0
rate = processed / elapsed
print(f"Processed {processed} samples in {elapsed:.2f}s ({rate:.1f} samples/s | {1000/rate:.1f} ms/sample)!")
