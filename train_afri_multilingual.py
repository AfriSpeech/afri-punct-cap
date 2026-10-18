"""
Pan-African Multilingual Punctuation & Capitalization Restoration
Fine-tunes Davlan/afro-xlmr-mini on 314 African languages from afri2universal
with acoustically aligned speech pauses on the H200 GPU.
"""

import os
import re
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_cosine_schedule_with_warmup
import pyarrow.parquet as pq
from safetensors.torch import save_file
from sklearn.metrics import f1_score

class AfriPunctCapModel(nn.Module):
    def __init__(self, config, base_model_name="Davlan/afro-xlmr-mini"):
        super().__init__()
        self.config = config
        self.roberta = AutoModel.from_pretrained(base_model_name, config=config)
        hidden_size = config.hidden_size # 384
        self.dropout = nn.Dropout(0.2)
        self.cap_head = nn.Linear(hidden_size, 3)
        self.punct_head = nn.Linear(hidden_size, 5)

    def forward(self, input_ids, attention_mask, cap_labels=None, punct_labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = self.dropout(outputs[0])
        cap_logits = self.cap_head(seq_out)
        punct_logits = self.punct_head(seq_out)

        loss = None
        if cap_labels is not None and punct_labels is not None:
            cap_weights = torch.tensor([1.0, 1.8, 2.2], device=input_ids.device)
            punct_weights = torch.tensor([0.6, 2.4, 2.0, 2.8, 2.8], device=input_ids.device)

            loss_cap = nn.CrossEntropyLoss(weight=cap_weights, ignore_index=-100)(
                cap_logits.view(-1, 3), cap_labels.view(-1)
            )
            loss_punct = nn.CrossEntropyLoss(weight=punct_weights, ignore_index=-100)(
                punct_logits.view(-1, 5), punct_labels.view(-1)
            )
            loss = loss_cap + loss_punct

        return {
            "loss": loss,
            "cap_logits": cap_logits,
            "punct_logits": punct_logits,
        }

def parse_multilingual_sentence(text):
    raw_tokens = text.split()
    words = []
    c_tags = []
    p_tags = []

    # Support Latin, Ethiopic (።, ፡), and Arabic (،, ؟) punctuation
    for tok in raw_tokens:
        p = 0
        clean = tok
        if clean.endswith((',', '،')):
            p = 1
            clean = clean[:-1]
        elif clean.endswith(('.', '።', ';', '፤', ':')):
            p = 2
            clean = clean[:-1]
        elif clean.endswith(('?', '؟')):
            p = 3
            clean = clean[:-1]
        elif clean.endswith(('!', '¡')):
            p = 4
            clean = clean[:-1]

        # Capitalization (applies to bicameral scripts like Latin, Coptic; unicameral scripts like Ge'ez/Arabic map to 0)
        if clean.isupper() and len(clean) > 1:
            c = 2 # ALL_CAPS
        elif clean and (clean[0].isupper() or any(part and part[0].isupper() for part in clean.split('-'))):
            c = 1 # Title Case
        else:
            c = 0 # Lowercase / Unicameral

        words.append(clean.lower())
        c_tags.append(c)
        p_tags.append(p)

    return words, c_tags, p_tags

class MultilingualPunctDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, c_tags, p_tags = self.data[idx]
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        word_ids = encoding.word_ids(batch_index=0)
        cap_labels = []
        punct_labels = []
        current_word = None

        for w_id in word_ids:
            if w_id is None:
                cap_labels.append(-100)
                punct_labels.append(-100)
            elif w_id != current_word:
                if w_id < len(c_tags):
                    cap_labels.append(c_tags[w_id])
                    punct_labels.append(p_tags[w_id])
                else:
                    cap_labels.append(-100)
                    punct_labels.append(-100)
                current_word = w_id
            else:
                cap_labels.append(-100)
                punct_labels.append(-100)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "cap_labels": torch.tensor(cap_labels, dtype=torch.long),
            "punct_labels": torch.tensor(punct_labels, dtype=torch.long),
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned_dir", default="/mnt/volume_d2wey28/projects/afri-punct-cap/data/aligned/split=train", help="Directory with aligned language parquet files")
    parser.add_argument("--output_dir", default="/mnt/volume_d2wey28/projects/afri-punct-cap/models/afri-punct-cap-mini", help="Output model directory")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(device)})")

    base_model_name = "Davlan/afro-xlmr-mini"
    print(f"Loading tokenizer & config for {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    config = AutoConfig.from_pretrained(base_model_name)

    # 1. Load all aligned parquet files
    parquet_files = sorted(glob.glob(os.path.join(args.aligned_dir, "*.parquet")))
    print(f"Found {len(parquet_files)} aligned language parquet files in {args.aligned_dir}...")

    parsed_data = []
    total_commas = 0
    t0 = time.time()

    for p_file in parquet_files:
        lang = os.path.basename(p_file).replace(".parquet", "")
        df = pq.read_table(p_file).to_pandas()
        total_commas += int(df.get("commas_added", 0).sum())
        for text in df["comma_fixed_transcript"]:
            if text and text.strip():
                words, c_tags, p_tags = parse_multilingual_sentence(text)
                if words and len(words) >= 2:
                    parsed_data.append((words, c_tags, p_tags))

    print(f"Loaded {len(parsed_data):,} sentences across {len(parquet_files)} languages in {time.time()-t0:.1f}s!")
    print(f"Total pause-aligned commas in dataset: {total_commas:,}")

    # Train / Val Split (95% / 5%)
    np.random.seed(42)
    indices = np.random.permutation(len(parsed_data))
    split = int(0.95 * len(parsed_data))
    train_data = [parsed_data[i] for i in indices[:split]]
    val_data = [parsed_data[i] for i in indices[split:]]
    print(f"Train sentences: {len(train_data):,} | Val sentences: {len(val_data):,}")

    train_ds = MultilingualPunctDataset(train_data, tokenizer)
    val_ds = MultilingualPunctDataset(val_data, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Instantiate Model
    model = AfriPunctCapModel(config, base_model_name=base_model_name).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    print(f"\n--- Training Pan-African Model for {args.epochs} Epochs ({total_steps:,} total steps) ---")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t_epoch = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            cap_labels = batch["cap_labels"].to(device, non_blocking=True)
            punct_labels = batch["punct_labels"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask, cap_labels, punct_labels)
                loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 200 == 0 or (step + 1) == len(train_loader):
                avg_l = total_loss / (step + 1)
                rate = (step + 1) * args.batch_size / (time.time() - t_epoch)
                print(f"Epoch {epoch+1}/{args.epochs} | Step {step+1}/{len(train_loader)} | Loss: {avg_l:.4f} | {rate:.1f} sent/s")

        # Validation
        model.eval()
        all_cap_preds, all_cap_trues = [], []
        all_punct_preds, all_punct_trues = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                cap_labels = batch["cap_labels"].to(device)
                punct_labels = batch["punct_labels"].to(device)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(input_ids, attention_mask)
                cap_pred = torch.argmax(outputs["cap_logits"], dim=-1)
                punct_pred = torch.argmax(outputs["punct_logits"], dim=-1)

                mask = cap_labels != -100
                all_cap_preds.extend(cap_pred[mask].cpu().numpy())
                all_cap_trues.extend(cap_labels[mask].cpu().numpy())
                all_punct_preds.extend(punct_pred[mask].cpu().numpy())
                all_punct_trues.extend(punct_labels[mask].cpu().numpy())

        cap_f1 = f1_score(all_cap_trues, all_cap_preds, average="macro")
        c_p = sum(1 for p, t in zip(all_punct_preds, all_punct_trues) if p == 1 and t == 1) / max(1, sum(1 for p in all_punct_preds if p == 1))
        c_r = sum(1 for p, t in zip(all_punct_preds, all_punct_trues) if p == 1 and t == 1) / max(1, sum(1 for t in all_punct_trues if t == 1))
        c_f1 = 2 * c_p * c_r / max(1e-6, c_p + c_r)

        dot_p = sum(1 for p, t in zip(all_punct_preds, all_punct_trues) if p == 2 and t == 2) / max(1, sum(1 for p in all_punct_preds if p == 2))
        dot_r = sum(1 for p, t in zip(all_punct_preds, all_punct_trues) if p == 2 and t == 2) / max(1, sum(1 for t in all_punct_trues if t == 2))

        print(f"--> Validation Epoch {epoch+1}: Cap F1: {cap_f1:.4f} | Comma Prec: {c_p:.3f}, Rec: {c_r:.3f}, F1: {c_f1:.3f} | Period Prec: {dot_p:.3f}, Rec: {dot_r:.3f}")

    # 3. Save PyTorch & Safetensors (both standard and FP16)
    os.makedirs(args.output_dir, exist_ok=True)
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(args.output_dir, "pytorch_model.bin"))
    save_file(state_dict, os.path.join(args.output_dir, "model.safetensors"))
    
    # Save explicit FP16 safetensors (compact ~230 MB for GPU deployment)
    fp16_state_dict = {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}
    save_file(fp16_state_dict, os.path.join(args.output_dir, "model_fp16.safetensors"))
    
    config.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel weights saved to {args.output_dir} (model.safetensors, model_fp16.safetensors, pytorch_model.bin)")

    # 4. ONNX Export and Quantization
    print("\n--- Exporting to ONNX and INT8 Quantization ---")
    model.eval().to("cpu")
    class OnnxWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, input_ids, attention_mask):
            out = self.m(input_ids, attention_mask)
            return out["cap_logits"], out["punct_logits"]

    onnx_wrapper = OnnxWrapper(model)
    dummy_input = tokenizer("Ahene nanahene aka se ese", return_tensors="pt")
    onnx_path = os.path.join(args.output_dir, "model.onnx")

    torch.onnx.export(
        onnx_wrapper,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["cap_logits", "punct_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "cap_logits": {0: "batch_size", 1: "sequence_length"},
            "punct_logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14
    )
    print(f"ONNX exported: {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

    from onnxruntime.quantization import quantize_dynamic, QuantType
    quant_path = os.path.join(args.output_dir, "model_quantized.onnx")
    quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)
    print(f"INT8 Quantized ONNX exported: {quant_path} ({os.path.getsize(quant_path)/1e6:.1f} MB)")

    print("\nPan-African Training & Export complete!")

if __name__ == "__main__":
    main()
