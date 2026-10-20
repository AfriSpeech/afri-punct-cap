import os
import re
import time
import modal
import numpy as np
from pydantic import BaseModel

def download_models():
    from transformers import AutoTokenizer, AutoConfig
    from huggingface_hub import hf_hub_download

    model_id = "AfriSpeech/afri-punct-cap"
    print(f"Pre-caching {model_id} into Modal image layer...")
    AutoTokenizer.from_pretrained(model_id)
    AutoConfig.from_pretrained(model_id)
    hf_hub_download(repo_id=model_id, filename="model_fp16.safetensors")
    hf_hub_download(repo_id=model_id, filename="model_quantized.onnx")
    print("Models pre-cached successfully!")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "torch==2.5.1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "safetensors",
        "onnxruntime>=1.18.0",
        "huggingface_hub",
        "numpy",
    )
    .run_function(download_models)
)

app = modal.App("afri-punct-cap-server", image=image)

class RestoreRequest(BaseModel):
    text: str
    precision: str = "fp16" # "fp16" (default) or "int8"
    lang: str = "afri"      # "afri" (default, zero-shot) or specific tag like "swh", "aka"

class RestoreResponse(BaseModel):
    restored_text: str
    latency_ms: float
    num_words: int
    num_commas: int
    num_periods: int
    precision: str
    mode: str

@app.cls(gpu="L4", cpu=2.0, memory=4096, scaledown_window=60)
class RestorerService:
    @modal.enter()
    def setup(self):
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoConfig, AutoModel
        from safetensors.torch import load_file
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_id = "AfriSpeech/afri-punct-cap"
        print(f"Initializing {model_id} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)

        # 1. FP16 PyTorch Model (Default, highest accuracy on GPU)
        class AfriPunctCapModel(nn.Module):
            def __init__(self, cfg, vocab_size=None):
                super().__init__()
                self.config = cfg
                self.roberta = AutoModel.from_config(cfg)
                if vocab_size and vocab_size != cfg.vocab_size:
                    self.roberta.resize_token_embeddings(vocab_size)
                hidden_size = cfg.hidden_size
                self.dropout = nn.Dropout(0.2)
                self.cap_head = nn.Linear(hidden_size, 3)
                self.punct_head = nn.Linear(hidden_size, 5)

            def forward(self, input_ids, attention_mask):
                outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
                seq_out = self.dropout(outputs[0])
                return self.cap_head(seq_out), self.punct_head(seq_out)

        self.fp16_model = AfriPunctCapModel(config, vocab_size=len(self.tokenizer))
        weights_path = hf_hub_download(repo_id=model_id, filename="model_fp16.safetensors")
        state_dict = load_file(weights_path)
        self.fp16_model.load_state_dict(state_dict)
        if self.device.type == "cuda":
            self.fp16_model = self.fp16_model.half().to(self.device)
        self.fp16_model.eval()

        # 2. INT8 ONNX Engine (Optional lightweight)
        onnx_path = hf_hub_download(repo_id=model_id, filename="model_quantized.onnx")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        self.onnx_session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

        self.rev_punct = {0: "", 1: ",", 2: ".", 3: "?", 4: "!"}
        self.torch = torch
        print("RestorerService initialized with FP16 (default) and INT8 ONNX!")

    def restore_logic(self, text: str, precision: str = "fp16", lang: str = "afri") -> dict:
        t0 = time.time()
        raw_tokens = text.strip().split()
        if not raw_tokens:
            return {
                "restored_text": "",
                "latency_ms": 0.0,
                "num_words": 0,
                "num_commas": 0,
                "num_periods": 0,
                "precision": precision,
                "mode": "Afri Mode (Default, Zero-Shot)" if lang in ["afri", "auto", ""] else f"Language-Specific ([{lang}])",
            }

        # Check for language conditioning: default is "afri" (no tag)
        has_lang_prefix = False
        lang_token = None
        if lang and lang.lower() not in ["afri", "auto", "none", ""]:
            clean_tag = lang.lower().replace("[", "").replace("]", "")
            lang_token = f"[{clean_tag}]"
            has_lang_prefix = True
        elif bool(re.match(r"^\[[a-zA-Z]{3}\]", raw_tokens[0])):
            has_lang_prefix = True

        # Clean outer punctuation
        clean_words = []
        for w in raw_tokens:
            cw = re.sub(r'^[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+|[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+$', '', w)
            clean_words.append(cw.lower() if cw else w.lower())

        if lang_token:
            input_words = [lang_token] + clean_words
        else:
            input_words = clean_words

        use_int8 = (precision.lower() == "int8")

        encoding = self.tokenizer(
            input_words,
            is_split_into_words=True,
            return_tensors="np" if use_int8 else "pt"
        )
        word_ids = encoding.word_ids(batch_index=0)

        if use_int8:
            ort_inputs = {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
            }
            ort_outputs = self.onnx_session.run(None, ort_inputs)
            cap_preds = np.argmax(ort_outputs[0][0], axis=-1)
            punct_preds = np.argmax(ort_outputs[1][0], axis=-1)
        else:
            with self.torch.no_grad():
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                cap_logits, punct_logits = self.fp16_model(input_ids, attention_mask)
            cap_preds = self.torch.argmax(cap_logits[0], dim=-1).cpu().numpy()
            punct_preds = self.torch.argmax(punct_logits[0], dim=-1).cpu().numpy()

        current_word = None
        word_caps, word_puncts = [], []
        for token_idx, w_id in enumerate(word_ids):
            if w_id is not None and w_id != current_word:
                word_caps.append(cap_preds[token_idx])
                word_puncts.append(punct_preds[token_idx])
                current_word = w_id

        # If lang_token was prepended, skip its predictions
        if lang_token:
            word_caps = word_caps[1:]
            word_puncts = word_puncts[1:]

        commas_count = 0
        periods_count = 0
        reconstructed = []

        for w, c, p in zip(clean_words, word_caps, word_puncts):
            if c == 2:
                word_str = w.upper()
            elif c == 1:
                word_str = "-".join(part.capitalize() for part in w.split("-"))
            else:
                word_str = w.lower()

            punct = self.rev_punct.get(p, "")
            if punct == ",":
                commas_count += 1
            elif punct in [".", "?"]:
                periods_count += 1

            word_str += punct
            reconstructed.append(word_str)

        if not lang_token and has_lang_prefix and reconstructed:
            reconstructed = reconstructed[1:]

        latency_ms = (time.time() - t0) * 1000
        mode_label = "Pan-African (Afri Mode)" if not lang_token else f"Language-Specific ({lang_token})"

        return {
            "restored_text": " ".join(reconstructed),
            "latency_ms": round(latency_ms, 1),
            "num_words": len(clean_words) - (1 if has_lang_prefix and not lang_token else 0),
            "num_commas": commas_count,
            "num_periods": periods_count,
            "precision": "FP16 (GPU)" if not use_int8 else "INT8 (CPU ONNX)",
            "mode": mode_label,
        }

    @modal.asgi_app()
    def api(self):
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        web_app = FastAPI(title="AfriPunctCap Serverless API")
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.post("/restore", response_model=RestoreResponse)
        def restore_endpoint(req: RestoreRequest):
            return self.restore_logic(req.text, precision=req.precision, lang=req.lang)

        @web_app.get("/health")
        def health():
            return {
                "status": "ok",
                "model": "AfriSpeech/afri-punct-cap",
                "default_precision": "FP16 (GPU)",
                "default_mode": "Afri Mode (Default, Zero-Shot)",
            }

        return web_app
