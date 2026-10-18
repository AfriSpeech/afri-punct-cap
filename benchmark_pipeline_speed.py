import io
import time
import queue
import threading
import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle
import soundfile as sf
import pyarrow.parquet as pq
import uroman

u = uroman.Uroman()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = bundle.get_model().to(device)
model.eval()
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()
dictionary = bundle.get_dict()
valid_chars = set(c for c in dictionary if c != "-")

path = "/mnt/volume_d2wey28/projects/afripa-asr/data/corpus-univ/version=0/corpus=afripa/split=train/language=yor/part-0.parquet"
df = pq.read_table(path).to_pandas().head(100)

# Producer: Thread for decoding audio & tokenizing
item_queue = queue.Queue(maxsize=32)
done_flag = threading.Event()

def producer():
    for i in range(len(df)):
        row = df.iloc[i]
        audio_arr, sr = sf.read(io.BytesIO(row["audio_bytes"]))
        dur = len(audio_arr) / sr

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
            waveform = torch.tensor(audio_arr, dtype=torch.float32).unsqueeze(0)
            if sr != bundle.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)
            tokens = tokenizer(cleaned_words)
            item_queue.put({
                "waveform": waveform,
                "orig_words": orig_words,
                "valid_indices": valid_indices,
                "tokens": tokens,
                "dur": dur
            })
    done_flag.set()

t_prod = threading.Thread(target=producer)
t0 = time.time()
t_prod.start()

processed = 0
punct_chars = (",", ".", "!", "?", ";", ":", "—", "-", "'", '"', "”", "’")

while not done_flag.is_set() or not item_queue.empty():
    try:
        item = item_queue.get(timeout=0.1)
    except queue.Empty:
        continue

    waveform = item["waveform"].to(device)
    tokens = item["tokens"]
    dur = item["dur"]
    orig_words = item["orig_words"]
    valid_indices = item["valid_indices"]

    with torch.no_grad():
        emission, _ = model(waveform)
    
    try:
        token_spans = aligner(emission[0].cpu(), tokens)
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
        for idx in range(len(word_spans) - 1):
            gap = word_spans[idx+1]["start"] - word_spans[idx]["end"]
            if gap >= 0.35:
                target_idx = word_spans[idx+1]["idx"] - 1
                cur_w = result_words[target_idx]
                if not cur_w.endswith(punct_chars):
                    result_words[target_idx] = cur_w + ","

        processed += 1
    except Exception:
        pass

t_prod.join()
elapsed = time.time() - t0
rate = processed / elapsed
print(f"Pipelined Producer-Consumer: {processed} samples in {elapsed:.2f}s ({rate:.1f} samples/s | {1000/rate:.1f} ms/sample)!")
