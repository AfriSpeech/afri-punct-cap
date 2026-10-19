"""
AfriPunctCap: Pan-African Multi-Sentence Punctuation & Capitalization Restoration
Fine-tunes Davlan/afro-xlmr-base on 314 African languages from afri2universal.

Features:
1. Multi-sentence concatenation (1-3 sentences packed together to learn internal sentence boundaries).
2. Language-ID prompting ([yor], [aka], [swh], [amh]...) with random dropout for zero-shot robustness.
3. Multi-script support: Ethiopic/Ge'ez (Amharic/Tigrinya), Arabic, Latin with African phonetic extensions.
4. 50 epochs training with cosine decay, best model checkpointing, and INT8 ONNX export.
"""

import os
import re
import glob
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_cosine_schedule_with_warmup
import pyarrow.parquet as pq
from safetensors.torch import save_file
from sklearn.metrics import f1_score
from huggingface_hub import HfApi

PUNCT_MAP = {
    # Commas (Latin, Arabic, Ethiopic)
    ",": 1, "،": 1, "፣": 1,
    # Periods / Sentence-Enders (Latin, Ethiopic Arat Neteb, semicolons)
    ".": 2, "።": 2, ";": 2, "፤": 2, ":": 2,
    # Question Marks (Latin, Arabic, Ethiopic)
    "?": 3, "؟": 3, "፧": 3,
    # Exclamations
    "!": 4, "¡": 4
}

class AfriPunctCapBaseModel(nn.Module):
    def __init__(self, config, base_model_name="Davlan/afro-xlmr-base", vocab_size=None):
        super().__init__()
        self.config = config
        self.roberta = AutoModel.from_pretrained(base_model_name, config=config)
        if vocab_size and vocab_size != config.vocab_size:
            self.roberta.resize_token_embeddings(vocab_size)
        hidden_size = config.hidden_size # 768
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
            # Balanced class weights for multi-script African text
            cap_weights = torch.tensor([1.0, 1.6, 2.0], device=input_ids.device)
            punct_weights = torch.tensor([0.6, 2.2, 1.8, 2.5, 2.5], device=input_ids.device)

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

def parse_words_and_tags(text):
    raw_tokens = text.split()
    words = []
    c_tags = []
    p_tags = []

    for tok in raw_tokens:
        p = 0
        clean = tok

        # Extract trailing punctuation (supports Latin, Ethiopic, Arabic)
        if clean.endswith((",", "،", "፣")):
            p = 1
            clean = clean[:-1]
        elif clean.endswith((".", "።", ";", "፤", ":")):
            p = 2
            clean = clean[:-1]
        elif clean.endswith(("?", "؟", "፧")):
            p = 3
            clean = clean[:-1]
        elif clean.endswith(("!", "¡")):
            p = 4
            clean = clean[:-1]

        # Strip remaining surrounding brackets/quotes
        clean = re.sub(r'^[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+|[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+$', '', clean)
        if not clean:
            continue

        # Capitalization (Bicameral scripts; Unicameral like Ge'ez/Arabic default to 0)
        if clean.isupper() and len(clean) > 1:
            c = 2 # ALL_CAPS
        elif clean and (clean[0].isupper() or any(part and part[0].isupper() for part in clean.split("-"))):
            c = 1 # Title Case
        else:
            c = 0 # Lowercase

        words.append(clean.lower())
        c_tags.append(c)
        p_tags.append(p)

    return words, c_tags, p_tags

class MultiSentenceAfriDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_len=160):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        lang, words, c_tags, p_tags = self.sequences[idx]

        # 85% chance to use language tag, 15% chance to omit (or use [afr]) for zero-shot generalization
        use_lang_tag = random.random() < 0.85
        lang_prefix = f"[{lang}]" if use_lang_tag else "[afr]"
        
        full_words = [lang_prefix] + words
        full_c_tags = [-100] + c_tags
        full_p_tags = [-100] + p_tags

        encoding = self.tokenizer(
            full_words,
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
                if w_id < len(full_c_tags):
                    cap_labels.append(full_c_tags[w_id])
                    punct_labels.append(full_p_tags[w_id])
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

def build_multi_sentence_sequences(aligned_dir):
    print("Building multi-sentence sequences across all 314 languages...")
    t0 = time.time()
    parquet_files = sorted(glob.glob(os.path.join(aligned_dir, "*.parquet")))
    print(f"Found {len(parquet_files)} language parquet files.")

    all_sequences = []
    total_single_sentences = 0
    total_commas = 0

    for p_file in parquet_files:
        lang = os.path.basename(p_file).replace(".parquet", "")
        df = pq.read_table(p_file).to_pandas()
        sentences = [t for t in df["comma_fixed_transcript"].dropna() if len(t.strip()) > 5]
        total_single_sentences += len(sentences)
        total_commas += int(df.get("commas_added", 0).sum())

        # Pack consecutive sentences into sequences of 1, 2, or 3 sentences
        i = 0
        while i < len(sentences):
            # Choose pack size: 25% 1 sent, 50% 2 sent, 25% 3 sent
            r = random.random()
            if r < 0.25:
                pack_size = 1
            elif r < 0.75:
                pack_size = 2
            else:
                pack_size = 3

            group = sentences[i : i + pack_size]
            i += pack_size

            # Concatenate with space
            combined_text = " ".join(group)
            words, c_tags, p_tags = parse_words_and_tags(combined_text)

            # Limit word length
            if words and len(words) >= 3 and len(words) <= 120:
                all_sequences.append((lang, words, c_tags, p_tags))

    print(f"Packed {total_single_sentences:,} single sentences into {len(all_sequences):,} multi-sentence training sequences in {time.time()-t0:.1f}s!")
    print(f"Total speech-pause commas represented: {total_commas:,}")
    return all_sequences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned_dir", default="/mnt/volume_d2wey28/projects/afri-punct-cap/data/aligned/split=train")
    parser.add_argument("--output_dir", default="/mnt/volume_d2wey28/projects/afri-punct-cap/models/afri-punct-cap-base-50ep")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(device)})")

    base_model_name = "Davlan/afro-xlmr-base"
    print(f"\nLoading {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    config = AutoConfig.from_pretrained(base_model_name)

    # Register 314 language special tokens: [aal], [aao], ..., [zul], [afr]
    parquet_files = sorted(glob.glob(os.path.join(args.aligned_dir, "*.parquet")))
    all_langs = sorted([os.path.basename(p).replace(".parquet", "") for p in parquet_files])
    lang_tokens = [f"[{l}]" for l in all_langs] + ["[afr]"]
    tokenizer.add_special_tokens({"additional_special_tokens": lang_tokens})
    print(f"Added {len(lang_tokens)} language ID tokens. Vocab size: {len(tokenizer)}")

    # 1. Build Multi-Sentence Dataset
    all_sequences = build_multi_sentence_sequences(args.aligned_dir)

    # 95% Train / 5% Validation Split
    random.seed(42)
    random.shuffle(all_sequences)
    split = int(0.95 * len(all_sequences))
    train_seqs = all_sequences[:split]
    val_seqs = all_sequences[split:]
    print(f"Train sequences: {len(train_seqs):,} | Validation sequences: {len(val_seqs):,}")

    train_ds = MultiSentenceAfriDataset(train_seqs, tokenizer, max_len=160)
    val_ds = MultiSentenceAfriDataset(val_seqs, tokenizer, max_len=160)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Instantiate Model
    model = AfriPunctCapBaseModel(config, base_model_name=base_model_name, vocab_size=len(tokenizer)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)

    print(f"\n=======================================================")
    print(f"Training Davlan/afro-xlmr-base for {args.epochs} Epochs")
    print(f"Total Steps: {total_steps:,} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"=======================================================\n")

    best_val_f1 = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

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

            if (step + 1) % 250 == 0 or (step + 1) == len(train_loader):
                avg_loss = total_loss / (step + 1)
                rate = (step + 1) * args.batch_size / (time.time() - t_epoch)
                print(f"Epoch {epoch+1:02d}/{args.epochs} | Step {step+1:04d}/{len(train_loader)} | Loss: {avg_loss:.4f} | {rate:.1f} seq/s", flush=True)

        epoch_dur = time.time() - t_epoch

        # Validation at end of each epoch
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

        composite_score = (cap_f1 + c_f1 + dot_p) / 3

        print(
            f"--> [Epoch {epoch+1:02d} Val ({epoch_dur/60:.1f}m)] Cap F1: {cap_f1:.4f} | "
            f"Comma Prec: {c_p:.3f}, Rec: {c_r:.3f}, F1: {c_f1:.3f} | "
            f"Period Prec: {dot_p:.3f}, Rec: {dot_r:.3f} | Score: {composite_score:.4f}",
            flush=True
        )

        # Checkpoint if best score
        if composite_score > best_val_f1:
            best_val_f1 = composite_score
            print(f"    * New best composite score ({best_val_f1:.4f})! Saving checkpoint...", flush=True)
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(args.output_dir, "pytorch_model.bin"))
            save_file(state_dict, os.path.join(args.output_dir, "model.safetensors"))
            fp16_dict = {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}
            save_file(fp16_dict, os.path.join(args.output_dir, "model_fp16.safetensors"))
            config.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    print(f"\nTraining completed! Best Validation Score: {best_val_f1:.4f}")

    # 3. Export ONNX & INT8 Quantization
    print("\n--- Exporting to ONNX and INT8 Quantization ---", flush=True)
    model.eval().to("cpu")

    class OnnxWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, input_ids, attention_mask):
            out = self.m(input_ids, attention_mask)
            return out["cap_logits"], out["punct_logits"]

    onnx_wrapper = OnnxWrapper(model)
    dummy_input = tokenizer("[yor] a maa n je ounje asaraloore", return_tensors="pt")
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
    print(f"ONNX exported: {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f} MB)", flush=True)

    from onnxruntime.quantization import quantize_dynamic, QuantType
    quant_path = os.path.join(args.output_dir, "model_quantized.onnx")
    quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)
    print(f"INT8 Quantized ONNX exported: {quant_path} ({os.path.getsize(quant_path)/1e6:.1f} MB)", flush=True)

    # 4. Auto-Push Updated Model to AfriSpeech Hugging Face
    print("\n--- Pushing Fine-Tuned afro-xlmr-base Model to AfriSpeech/afri-punct-cap ---", flush=True)
    model_repo = "AfriSpeech/afri-punct-cap"
    api = HfApi()
    api.create_repo(repo_id=model_repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=args.output_dir,
        repo_id=model_repo,
        repo_type="model",
        commit_message=f"Update with Davlan/afro-xlmr-base (50 epochs, multi-sentence context, 314 languages)"
    )
    print(f"Updated model pushed to https://huggingface.co/{model_repo}!", flush=True)

if __name__ == "__main__":
    main()
