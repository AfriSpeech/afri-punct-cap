"""
T5-Small Proxy-Mode Punctuation & Capitalization Restorer
Uses africa-g2p PROXY mode for zero-shot multilingual African text restoration without language conditioning.
"""

import sys
import os
import re
import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from africa_g2p import to_proxy, from_proxy


class T5ProxyRestorer:
    def __init__(self, model_path_or_id="AfriSpeech/afri-punct-cap-t5", device="cpu"):
        self.device = torch.device(device)
        print(f"Loading T5 Proxy Restorer from {model_path_or_id} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path_or_id).to(self.device)
        self.model.eval()
        print("T5 Proxy Restorer initialized!")

    def restore(self, text: str, max_length=160) -> str:
        """
        Takes raw unpunctuated African text (any language/script), converts to proxy mode,
        restores punctuation/casing with T5-small, and decodes back via from_proxy.
        """
        if not text or not text.strip():
            return ""

        # 1. Clean raw text to lowercase words
        raw_tokens = text.strip().split()
        cleaned_words = []
        for w in raw_tokens:
            cw = re.sub(r'^[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+|[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+$', '', w)
            cleaned_words.append(cw.lower() if cw else w.lower())

        clean_text = " ".join(cleaned_words)

        # 2. Convert to ASCII Proxy mode
        input_proxy = to_proxy(clean_text)
        input_proxy = re.sub(r'[,.?!;:]', '', input_proxy)

        # 3. Model input with prefix
        prompt = "restore: " + input_proxy
        enc = self.tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True).to(self.device)

        with torch.no_grad():
            gen_tokens = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_length=max_length,
                num_beams=1, # Greedy decoding for fast CPU inference
            )

        gen_proxy = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()

        # 4. Decode back from proxy to original orthography with restored punctuation & casing
        try:
            restored = from_proxy(gen_proxy)
        except Exception:
            restored = gen_proxy

        return restored


def main():
    parser = argparse.ArgumentParser(description="T5-Small Proxy Multilingual Restorer")
    parser.add_argument("text", nargs="?", help="Input African text (any language, no language tag needed)")
    parser.add_argument("--model", default="AfriSpeech/afri-punct-cap-t5", help="Model path or HF repo ID")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    restorer = T5ProxyRestorer(model_path_or_id=args.model, device=args.device)

    if args.text:
        t0 = time.time()
        result = restorer.restore(args.text)
        lat = (time.time() - t0) * 1000
        print("\n--- Restored Output ---")
        print(result)
        print(f"\nLatency: {lat:.1f}ms on {args.device}")
    else:
        print("\nInteractive T5 Proxy Restorer (Type text and press Enter, Ctrl+C to exit):")
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
