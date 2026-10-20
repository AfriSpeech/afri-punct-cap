"""
Compare FP16 vs INT8 Performance on Mega Pan-African Model across Core Languages
Generates exact side-by-side comparison table for Hugging Face Model Card.
"""

import os
import re
import glob
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
import onnxruntime as ort
import pyarrow.parquet as pq
from safetensors.torch import load_file
import numpy as np

model_dir = "/mnt/volume_d2wey28/projects/afri-punct-cap/models/afri-punct-cap-mega-base"
aligned_dir = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/aligned/split=train"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir)

# 1. Load FP16 Model
class AfriPunctCapBaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("Davlan/afro-xlmr-base", config=cfg)
        self.dropout = nn.Dropout(0.2)
        self.cap_head = nn.Linear(cfg.hidden_size, 3)
        self.punct_head = nn.Linear(cfg.hidden_size, 5)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = self.dropout(outputs[0])
        return self.cap_head(seq_out), self.punct_head(seq_out)

fp16_model = AfriPunctCapBaseModel(config).to(device)
fp16_model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")))
fp16_model.eval()

# 2. Load INT8 ONNX Model
quant_onnx_path = os.path.join(model_dir, "model_quantized.onnx")
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
int8_session = ort.InferenceSession(quant_onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

def parse_words_and_tags(text):
    raw_tokens = text.split()
    words, p_tags = [], []
    for tok in raw_tokens:
        p = 0
        clean = tok
        if clean.endswith((",", "،", "፣")):
            p = 1
            clean = clean[:-1]
        elif clean.endswith((".", "።", ";", "፤", ":")):
            p = 2
            clean = clean[:-1]
        clean = re.sub(r"^[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+|[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+$", "", clean)
        if not clean:
            continue
        words.append(clean.lower())
        p_tags.append(p)
    return words, p_tags

def eval_both(lang):
    p_file = os.path.join(aligned_dir, f"{lang}.parquet")
    if not os.path.exists(p_file):
        return None
    df = pq.read_table(p_file).to_pandas().tail(150)

    fp16_preds, int8_preds = [], []
    all_trues = []

    for text in df["comma_fixed_transcript"]:
        if not text or len(text.strip()) < 5:
            continue
        words, p_tags = parse_words_and_tags(text)
        if len(words) < 2:
            continue

        # Zero language tags - pure words
        enc = tokenizer(words, is_split_into_words=True, max_length=128, truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        word_ids = enc.word_ids(batch_index=0)

        # 1. FP16 Inference (GPU)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, fp16_logits = fp16_model(input_ids.to(device), attention_mask.to(device))
        p_fp16 = torch.argmax(fp16_logits[0], dim=-1).cpu().numpy()

        # 2. INT8 ONNX Inference (CPU)
        ort_out = int8_session.run(None, {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()})
        p_int8 = np.argmax(ort_out[1][0], axis=-1)

        current_word = None
        for t_idx, w_id in enumerate(word_ids):
            if w_id is not None and w_id != current_word:
                if w_id < len(p_tags):
                    fp16_preds.append(p_fp16[t_idx])
                    int8_preds.append(p_int8[t_idx])
                    all_trues.append(p_tags[w_id])
                current_word = w_id

    if not all_trues:
        return None

    def calc_metrics(preds, trues):
        c_p = sum(1 for p, t in zip(preds, trues) if p == 1 and t == 1) / max(1, sum(1 for p in preds if p == 1))
        c_r = sum(1 for p, t in zip(preds, trues) if p == 1 and t == 1) / max(1, sum(1 for t in trues if t == 1))
        c_f1 = 2 * c_p * c_r / max(1e-6, c_p + c_r)

        dot_p = sum(1 for p, t in zip(preds, trues) if p == 2 and t == 2) / max(1, sum(1 for p in preds if p == 2))
        dot_r = sum(1 for p, t in zip(preds, trues) if p == 2 and t == 2) / max(1, sum(1 for t in trues if t == 2))
        dot_f1 = 2 * dot_p * dot_r / max(1e-6, dot_p + dot_r)
        return c_p, c_r, c_f1, dot_f1

    fp16_cp, fp16_cr, fp16_cf1, fp16_dotf1 = calc_metrics(fp16_preds, all_trues)
    int8_cp, int8_cr, int8_cf1, int8_dotf1 = calc_metrics(int8_preds, all_trues)

    return {
        "lang": lang,
        "fp16": (fp16_cp, fp16_cr, fp16_cf1, fp16_dotf1),
        "int8": (int8_cp, int8_cr, int8_cf1, int8_dotf1),
        "words": len(all_trues)
    }

languages = [
    ("swh", "Swahili"),
    ("zul", "Zulu"),
    ("hau", "Hausa"),
    ("yor", "Yoruba"),
    ("aka", "Akan/Twi"),
    ("ewe", "Ewe"),
    ("lin", "Lingala"),
    ("kin", "Kinyarwanda"),
    ("wol", "Wolof"),
    ("som", "Somali"),
]

print("\nEvaluating FP16 vs INT8 side-by-side across core African languages...\n")
results = []
for code, name in languages:
    res = eval_both(code)
    if res:
        results.append((code, name, res))
        print(f"[{code.upper()} - {name}]")
        print(f"   FP16: Comma F1: {res['fp16'][2]:.3f} (P: {res['fp16'][0]:.3f}, R: {res['fp16'][1]:.3f}) | Period F1: {res['fp16'][3]:.3f}")
        print(f"   INT8: Comma F1: {res['int8'][2]:.3f} (P: {res['int8'][0]:.3f}, R: {res['int8'][1]:.3f}) | Period F1: {res['int8'][3]:.3f}")
        print()

avg_fp16_cf1 = np.mean([r[2]['fp16'][2] for r in results])
avg_int8_cf1 = np.mean([r[2]['int8'][2] for r in results])
avg_fp16_dotf1 = np.mean([r[2]['fp16'][3] for r in results])
avg_int8_dotf1 = np.mean([r[2]['int8'][3] for r in results])

print("==========================================================================")
print(f"AVERAGE FP16: Comma F1: {avg_fp16_cf1:.3f} | Period F1: {avg_fp16_dotf1:.3f}")
print(f"AVERAGE INT8: Comma F1: {avg_int8_cf1:.3f} | Period F1: {avg_int8_dotf1:.3f} (Retains {avg_int8_cf1/avg_fp16_cf1*100:.1f}% accuracy)")
print("==========================================================================")
