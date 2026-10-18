import io
import time
import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle
import soundfile as sf
import pyarrow.parquet as pq
import uroman
from torch.nn.utils.rnn import pad_sequence

u = uroman.Uroman()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = bundle.get_model().to(device)
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()
dictionary = bundle.get_dict()
valid_chars = set(c for c in dictionary if c != "-")

path = "/mnt/volume_d2wey28/projects/afripa-asr/data/corpus-univ/version=0/corpus=afripa/split=train/language=yor/part-0.parquet"
df = pq.read_table(path).to_pandas().head(100)

# Pre-load audio & tokenize words
items = []
for i in range(len(df)):
    row = df.iloc[i]
    audio_arr, sr = sf.read(io.BytesIO(row["audio_bytes"]))
    orig_words = row["transcript"].split()
    cleaned_words = []
    valid_indices = []
    for idx, w in enumerate(orig_words):
        rom = u.romanize_string(w)
        cw = "".join([c for c in rom.lower() if c in valid_chars])
        if cw:
            cleaned_words.append(cw)
            valid_indices.append(idx)
    if len(cleaned_words) >= 2:
        waveform = torch.tensor(audio_arr, dtype=torch.float32)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sr, 16000).squeeze(0)
        items.append({
            "waveform": waveform,
            "orig_words": orig_words,
            "cleaned_words": cleaned_words,
            "valid_indices": valid_indices,
            "tokens": tokenizer(cleaned_words)
        })

print(f"Loaded {len(items)} items for batched benchmark...")

# Test batch size 16
batch_size = 16
t0 = time.time()
processed = 0

for b_start in range(0, len(items), batch_size):
    batch = items[b_start:b_start + batch_size]
    waveforms = [item["waveform"] for item in batch]
    padded = pad_sequence(waveforms, batch_first=True).to(device) # (B, max_len)
    
    with torch.no_grad():
        emissions, _ = model(padded)
    emissions_cpu = emissions.cpu()

    for idx, item in enumerate(batch):
        # unpad length
        cur_len = len(item["waveform"])
        cur_frames = int(cur_len / 320)
        emission = emissions_cpu[idx, :cur_frames]
        try:
            token_spans = aligner(emission, item["tokens"])
            processed += 1
        except Exception:
            pass

elapsed = time.time() - t0
rate = processed / elapsed
print(f"Batched (B={batch_size}): {processed} samples in {elapsed:.2f}s ({rate:.1f} samples/s | {1000/rate:.1f} ms/sample)!")
