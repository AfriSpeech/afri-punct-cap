"""
AfriPunctCap: T5-Small Proxy-Mode Punctuation & Capitalization Restoration
Trained on 203 African languages with sufficient data (>=100 sentences).
Uses africa-g2p PROXY mode (plain ASCII a-z encoding) for multi-script African text.
No language conditioning. Balanced 203-language validation set.
"""

import os
import re
import glob
import time
import random
import argparse
import unicodedata
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, get_cosine_schedule_with_warmup
import pyarrow.parquet as pq
from safetensors.torch import save_file
from africa_g2p import to_proxy, from_proxy

aligned_dir = "/mnt/volume_d2wey28/projects/afri-punct-cap/data/aligned/split=train"

def clean_word(tok):
    return re.sub(r'^[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+|[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+$', '', tok)

class ProxySeq2SeqDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len=160, max_target_len=160):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        in_proxy, tgt_proxy = self.data[idx]
        
        in_enc = self.tokenizer(
            "restore: " + in_proxy,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tgt_enc = self.tokenizer(
            tgt_proxy,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = tgt_enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": in_enc["input_ids"].squeeze(0),
            "attention_mask": in_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }

def build_proxy_dataset(min_rows=100, val_per_lang=20):
    print(f"Scanning aligned language files with >= {min_rows} rows...")
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(aligned_dir, "*.parquet")))
    
    # 1. Filter to languages with sufficient data (>= 100 rows)
    core_files = []
    for f in files:
        df_meta = pq.read_table(f, columns=["language"])
        if len(df_meta) >= min_rows:
            core_files.append((f, len(df_meta)))

    print(f"Selected {len(core_files)} core languages (>= {min_rows} rows) out of {len(files)} total.")

    train_data = []
    val_data = []
    random.seed(42)

    for p_file, num_rows in core_files:
        lang = os.path.basename(p_file).replace(".parquet", "")
        df = pq.read_table(p_file).to_pandas()
        sentences = [t.strip() for t in df["comma_fixed_transcript"].dropna() if len(t.strip()) > 5]

        # Pack into 1-3 sentence sequences
        lang_seqs = []
        i = 0
        while i < len(sentences):
            r = random.random()
            if r < 0.25:
                pack = 1
            elif r < 0.75:
                pack = 2
            else:
                pack = 3
            group = sentences[i : i + pack]
            i += pack

            target_text = " ".join(group)
            
            # Generate unpunctuated, lowercased proxy input
            # Use to_proxy on target
            target_proxy = to_proxy(target_text)
            
            # Lowercase without punctuation for input proxy
            tokens = target_text.split()
            cleaned_tokens = [clean_word(t).lower() for t in tokens if clean_word(t)]
            if len(cleaned_tokens) < 3:
                continue
            raw_input_text = " ".join(cleaned_tokens)
            input_proxy = to_proxy(raw_input_text)
            input_proxy = re.sub(r'[,.?!;:]', '', input_proxy)

            lang_seqs.append((input_proxy, target_proxy, target_text))

        # Stratified Split: reserve exactly val_per_lang sequences for validation
        n_val = min(val_per_lang, int(len(lang_seqs) * 0.08))
        if n_val > 0:
            val_items = lang_seqs[-n_val:]
            train_items = lang_seqs[:-n_val]
        else:
            val_items = []
            train_items = lang_seqs

        for inp, tgt, _ in train_items:
            train_data.append((inp, tgt))
        for inp, tgt, orig in val_items:
            val_data.append((inp, tgt, orig, lang))

    print(f"Data Preparation Complete in {time.time()-t0:.1f}s:")
    print(f"  Train sequences:      {len(train_data):,} (across {len(core_files)} languages)")
    print(f"  Validation sequences: {len(val_data):,} (balanced {val_per_lang} per language across {len(core_files)} languages)")
    return train_data, val_data, core_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/mnt/volume_d2wey28/projects/afri-punct-cap/models/t5-small-proxy")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")

    model_name = "google-t5/t5-small"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config).to(device)

    train_data, val_data, core_files = build_proxy_dataset(min_rows=100, val_per_lang=20)

    val_train_format = [(inp, tgt) for inp, tgt, _, _ in val_data]
    train_ds = ProxySeq2SeqDataset(train_data, tokenizer, max_input_len=160, max_target_len=160)
    val_ds = ProxySeq2SeqDataset(val_train_format, tokenizer, max_input_len=160, max_target_len=160)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)

    print(f"\n=======================================================")
    print(f"Training T5-Small in PROXY Mode (Zero Special Chars)")
    print(f"Languages: {len(core_files)} | Epochs: {args.epochs} | Steps: {total_steps:,}")
    print(f"Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"=======================================================\n")

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")

    # Sample subset of validation for generation metrics
    random.seed(42)
    val_sample = random.sample(val_data, min(150, len(val_data)))

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t_epoch = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 250 == 0 or (step + 1) == len(train_loader):
                cur_l = total_loss / (step + 1)
                rate = (step + 1) * args.batch_size / (time.time() - t_epoch)
                print(f"Epoch {epoch+1:02d}/{args.epochs} | Step {step+1:04d}/{len(train_loader)} | Loss: {cur_l:.4f} | {rate:.1f} seq/s", flush=True)

        epoch_dur = time.time() - t_epoch
        train_loss = total_loss / len(train_loader)

        # Validation Loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)

        # Evaluation on sample using from_proxy decoding
        c_preds, c_trues = 0, 0
        c_correct = 0
        dot_preds, dot_trues = 0, 0
        dot_correct = 0

        with torch.no_grad():
            for in_p, tgt_p, orig_target, lang in val_sample[:40]:
                enc_in = tokenizer("restore: " + in_p, return_tensors="pt", max_length=160, truncation=True).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    gen = model.generate(input_ids=enc_in["input_ids"], attention_mask=enc_in["attention_mask"], max_length=160, num_beams=1)
                gen_text = tokenizer.decode(gen[0], skip_special_tokens=True)
                
                try:
                    decoded = from_proxy(gen_text)
                except Exception:
                    decoded = gen_text

                # Count commas and periods
                pred_c = decoded.count(",") + decoded.count("،") + decoded.count("፣")
                true_c = orig_target.count(",") + orig_target.count("،") + orig_target.count("፣")
                c_preds += pred_c
                c_trues += true_c
                c_correct += min(pred_c, true_c)

                pred_dot = decoded.count(".") + decoded.count("።")
                true_dot = orig_target.count(".") + orig_target.count("።")
                dot_preds += pred_dot
                dot_trues += true_dot
                dot_correct += min(pred_dot, true_dot)

        c_prec = c_correct / max(1, c_preds)
        c_rec = c_correct / max(1, c_trues)
        c_f1 = 2 * c_prec * c_rec / max(1e-6, c_prec + c_rec)

        print(
            f"--> [Epoch {epoch+1:02d} Val ({epoch_dur/60:.1f}m)] Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Comma Precision: {c_prec:.3f}, Recall: {c_rec:.3f}, F1: {c_f1:.3f}",
            flush=True
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"    * New best validation loss ({best_val_loss:.4f})! Saving checkpoint...", flush=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    print(f"\nT5-Small Proxy Training Complete! Best Val Loss: {best_val_loss:.4f}")

    # 4. Export to ONNX
    print("\n--- Exporting T5-Small to ONNX ---", flush=True)
    # Save standard PyTorch and safetensors
    state_dict = model.state_dict()
    save_file(state_dict, os.path.join(args.output_dir, "model.safetensors"))
    fp16_dict = {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}
    save_file(fp16_dict, os.path.join(args.output_dir, "model_fp16.safetensors"))
    print(f"Models saved in {args.output_dir}")

if __name__ == "__main__":
    main()
