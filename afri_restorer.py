"""
AfriRestorer: Pan-African Punctuation & Capitalization Restoration
Fast CPU inference tool supporting 314 African languages.
Restores capitalization and punctuation (commas, periods, question marks).
Supports ONNX INT8 runtime (~15–25ms on CPU) and standard PyTorch.
"""

import sys
import os
import re
import argparse
import time
import numpy as np


class AfriRestorer:
    def __init__(self, model_id="AfriSpeech/afri-punct-cap", backend="auto", device="cpu"):
        """
        backend: "auto", "onnx", or "torch"
        device: "cpu" or "cuda"
        """
        self.model_id = model_id
        self.device = device
        self.rev_punct = {0: "", 1: ",", 2: ".", 3: "?", 4: "!"}

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        use_onnx = False
        if backend in ["auto", "onnx"]:
            try:
                import onnxruntime as ort
                use_onnx = True
            except ImportError:
                if backend == "onnx":
                    raise ImportError("onnxruntime is required for backend='onnx'. Install with: pip install onnxruntime")
                use_onnx = False

        if use_onnx:
            self.backend = "onnx"
            print(f"Loading quantized ONNX model from {model_id} for fast CPU inference...")
            import onnxruntime as ort
            if os.path.isdir(model_id):
                onnx_path = os.path.join(model_id, "model_quantized.onnx")
            else:
                from huggingface_hub import hf_hub_download
                onnx_path = hf_hub_download(repo_id=model_id, filename="model_quantized.onnx")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = min(4, os.cpu_count() or 4)
            self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
            print("ONNX INT8 session initialized!")
        else:
            self.backend = "torch"
            print(f"Loading PyTorch model from {model_id} on {device}...")
            import torch
            import torch.nn as nn
            from transformers import AutoConfig, AutoModel
            from safetensors.torch import load_file

            class AfriPunctCapModel(nn.Module):
                def __init__(self, config, base_model_name="Davlan/afro-xlmr-base", vocab_size=None):
                    super().__init__()
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

            config = AutoConfig.from_pretrained(model_id)
            if os.path.isdir(model_id):
                weights_path = os.path.join(model_id, "model.safetensors")
            else:
                from huggingface_hub import hf_hub_download
                weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
            self.model = AfriPunctCapModel(config, vocab_size=len(self.tokenizer))
            state_dict = load_file(weights_path)
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()
            self.torch = torch
            print("PyTorch model initialized!")

    def restore(self, text: str) -> str:
        """
        Takes raw unpunctuated/uncapitalized text in any of 314 African languages,
        strips any existing punctuation, and restores proper capitalization and punctuation.
        """
        raw_tokens = text.split()
        if not raw_tokens:
            return ""

        clean_words = []
        for w in raw_tokens:
            # Strip outer punctuation while preserving language-specific alphabetic characters
            cw = re.sub(r'^[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+|[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+$', '', w)
            clean_words.append(cw.lower() if cw else w.lower())

        encoding = self.tokenizer(
            clean_words,
            is_split_into_words=True,
            return_tensors="np" if self.backend == "onnx" else "pt"
        )
        word_ids = encoding.word_ids(batch_index=0)

        if self.backend == "onnx":
            ort_inputs = {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
            }
            ort_outputs = self.session.run(None, ort_inputs)
            cap_preds = np.argmax(ort_outputs[0][0], axis=-1)
            punct_preds = np.argmax(ort_outputs[1][0], axis=-1)
        else:
            with self.torch.no_grad():
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                cap_logits, punct_logits = self.model(input_ids, attention_mask)
            cap_preds = self.torch.argmax(cap_logits[0], dim=-1).cpu().numpy()
            punct_preds = self.torch.argmax(punct_logits[0], dim=-1).cpu().numpy()

        current_word = None
        word_caps, word_puncts = [], []
        for token_idx, w_id in enumerate(word_ids):
            if w_id is not None and w_id != current_word:
                word_caps.append(cap_preds[token_idx])
                word_puncts.append(punct_preds[token_idx])
                current_word = w_id

        reconstructed = []
        for w, c, p in zip(clean_words, word_caps, word_puncts):
            if c == 2:
                word_str = w.upper()
            elif c == 1:
                word_str = "-".join(part.capitalize() for part in w.split("-"))
            else:
                word_str = w.lower()

            word_str += self.rev_punct.get(p, "")
            reconstructed.append(word_str)

        # Strip language prompt token from reconstructed output if present
        if raw_tokens and re.match(r"^\[[a-zA-Z]{3}\]", raw_tokens[0]):
            reconstructed = reconstructed[1:]

        return " ".join(reconstructed)


def main():
    parser = argparse.ArgumentParser(description="Pan-African Punctuation & Capitalization Restorer")
    parser.add_argument("text", nargs="?", help="Input African text (uncapitalized/unpunctuated)")
    parser.add_argument("--backend", default="auto", choices=["auto", "onnx", "torch"], help="Inference backend (default: auto)")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    restorer = AfriRestorer(backend=args.backend, device=args.device)

    if args.text:
        t0 = time.time()
        result = restorer.restore(args.text)
        lat = (time.time() - t0) * 1000
        print("\n--- Restored Output ---")
        print(result)
        print(f"\nLatency: {lat:.1f}ms ({restorer.backend})")
    else:
        print("\nInteractive Pan-African Restorer (Type text and press Enter, Ctrl+C to exit):")
        try:
            while True:
                user_in = input("\n>> ")
                if not user_in.strip():
                    continue
                t0 = time.time()
                out = restorer.restore(user_in)
                lat = (time.time() - t0) * 1000
                print(f"Output: {out}")
                print(f"({lat:.1f}ms)")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")


if __name__ == "__main__":
    main()
