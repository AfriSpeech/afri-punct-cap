import os
import re
import glob
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
import pyarrow.parquet as pq
from safetensors.torch import load_file
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

model_dir = "/mnt/volume_d2wey28/projects/afri-punct-cap/models/afri-punct-cap-base-50ep"
aligned_dir = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/aligned/split=train"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir)

class AfriPunctCapBaseModel(nn.Module):
    def __init__(self, config, base_model_name="Davlan/afro-xlmr-base", vocab_size=None):
        super().__init__()
        self.config = config
        self.roberta = AutoModel.from_pretrained(base_model_name, config=config)
        if vocab_size and vocab_size != config.vocab_size:
            self.roberta.resize_token_embeddings(vocab_size)
        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.cap_head = nn.Linear(hidden_size, 3)
        self.punct_head = nn.Linear(hidden_size, 5)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = self.dropout(outputs[0])
        return self.cap_head(seq_out), self.punct_head(seq_out)

model = AfriPunctCapBaseModel(config, vocab_size=len(tokenizer)).to(device)
model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")))
model.eval()

def parse_words_and_tags(text):
    raw_tokens = text.split()
    words, c_tags, p_tags = [], [], []
    for tok in raw_tokens:
        p = 0
        clean = tok
        if clean.endswith((",", "،", "፣")):
            p = 1
            clean = clean[:-1]
        elif clean.endswith((".", "።", ";", "፤", ":")):
            p = 2
            clean = clean[:-1]
        clean = re.sub(r'^[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+|[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+$', '', clean)
        if not clean:
            continue
        words.append(clean.lower())
        p_tags.append(p)
    return words, p_tags

def eval_lang(lang):
    p_file = os.path.join(aligned_dir, f"{lang}.parquet")
    if not os.path.exists(p_file):
        return None
    df = pq.read_table(p_file).to_pandas().tail(150) # Use held-out tail

    all_preds = []
    all_trues = []

    for text in df["comma_fixed_transcript"]:
        if not text or len(text.strip()) < 5:
            continue
        words, p_tags = parse_words_and_tags(text)
        if len(words) < 2:
            continue

        full_words = [f"[{lang}]"] + words
        enc = tokenizer(full_words, is_split_into_words=True, max_length=128, truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        word_ids = enc.word_ids(batch_index=0)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _, punct_logits = model(input_ids, attention_mask)
        preds = torch.argmax(punct_logits[0], dim=-1).cpu().numpy()

        current_word = None
        for t_idx, w_id in enumerate(word_ids):
            if w_id is not None and w_id != current_word:
                if w_id > 0 and (w_id - 1) < len(p_tags):
                    all_preds.append(preds[t_idx])
                    all_trues.append(p_tags[w_id - 1])
                current_word = w_id

    c_p = sum(1 for p, t in zip(all_preds, all_trues) if p == 1 and t == 1) / max(1, sum(1 for p in all_preds if p == 1))
    c_r = sum(1 for p, t in zip(all_preds, all_trues) if p == 1 and t == 1) / max(1, sum(1 for t in all_trues if t == 1))
    c_f1 = 2 * c_p * c_r / max(1e-6, c_p + c_r)

    dot_p = sum(1 for p, t in zip(all_preds, all_trues) if p == 2 and t == 2) / max(1, sum(1 for p in all_preds if p == 2))
    dot_r = sum(1 for p, t in zip(all_preds, all_trues) if p == 2 and t == 2) / max(1, sum(1 for t in all_trues if t == 2))
    dot_f1 = 2 * dot_p * dot_r / max(1e-6, dot_p + dot_r)

    return {
        "lang": lang,
        "comma_p": c_p,
        "comma_r": c_r,
        "comma_f1": c_f1,
        "period_f1": dot_f1,
        "total_words": len(all_trues)
    }

print("=== Per-Language Comma Evaluation ===")
test_langs = ["aka", "yor", "swh", "hau", "amh", "ach", "ada", "bem", "zul", "lin"]
for l in test_langs:
    res = eval_lang(l)
    if res:
        print(f"[{res['lang'].upper():<4}] Comma Prec: {res['comma_p']:.3f} | Recall: {res['comma_r']:.3f} | Comma F1: {res['comma_f1']:.3f} | Period F1: {res['period_f1']:.3f} ({res['total_words']} words)")
