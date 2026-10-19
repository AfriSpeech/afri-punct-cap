import os
import re
import time
import modal
import numpy as np
from pydantic import BaseModel

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "onnxruntime>=1.18.0",
        "huggingface_hub",
        "numpy",
    )
)

app = modal.App("afri-punct-cap-server", image=image)

class RestoreRequest(BaseModel):
    text: str

class RestoreResponse(BaseModel):
    restored_text: str
    latency_ms: float
    num_words: int
    num_commas: int
    num_periods: int

@app.cls(cpu=2.0, memory=2048, scaledown_window=60)
class RestorerService:
    @modal.enter()
    def setup(self):
        from transformers import AutoTokenizer
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        model_id = "AfriSpeech/afri-punct-cap"
        print(f"Loading {model_id} into ONNX session...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        onnx_path = hf_hub_download(repo_id=model_id, filename="model_quantized.onnx")

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        self.rev_punct = {0: "", 1: ",", 2: ".", 3: "?", 4: "!"}
        print("ONNX Restorer ready!")

    @modal.method()
    def restore(self, text: str) -> dict:
        t0 = time.time()
        raw_tokens = text.strip().split()
        if not raw_tokens:
            return {
                "restored_text": "",
                "latency_ms": 0.0,
                "num_words": 0,
                "num_commas": 0,
                "num_periods": 0,
            }

        # Check for optional language tag [lang]
        has_lang_prefix = bool(re.match(r"^\[[a-zA-Z]{3}\]", raw_tokens[0]))
        
        # Clean outer punctuation
        clean_words = []
        for w in raw_tokens:
            cw = re.sub(r'^[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+|[^\w\u0190\u0186\u025B\u0254\u1200-\u137F\u0600-\u06FF]+$', '', w)
            clean_words.append(cw.lower() if cw else w.lower())

        encoding = self.tokenizer(
            clean_words,
            is_split_into_words=True,
            return_tensors="np"
        )
        word_ids = encoding.word_ids(batch_index=0)

        ort_inputs = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }
        ort_outputs = self.session.run(None, ort_inputs)
        cap_preds = np.argmax(ort_outputs[0][0], axis=-1)
        punct_preds = np.argmax(ort_outputs[1][0], axis=-1)

        current_word = None
        word_caps, word_puncts = [], []
        for token_idx, w_id in enumerate(word_ids):
            if w_id is not None and w_id != current_word:
                word_caps.append(cap_preds[token_idx])
                word_puncts.append(punct_preds[token_idx])
                current_word = w_id

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

        if has_lang_prefix and reconstructed:
            reconstructed = reconstructed[1:]

        latency_ms = (time.time() - t0) * 1000
        return {
            "restored_text": " ".join(reconstructed),
            "latency_ms": round(latency_ms, 1),
            "num_words": len(clean_words) - (1 if has_lang_prefix else 0),
            "num_commas": commas_count,
            "num_periods": periods_count,
        }

# FastAPI App for Web / HF Static Space
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
    service = RestorerService()
    result = service.restore.remote(req.text)
    return result

@web_app.get("/health")
def health():
    return {"status": "ok", "model": "AfriSpeech/afri-punct-cap", "backend": "ONNX INT8"}

@app.function(scaledown_window=60)
@modal.asgi_app()
def api():
    return web_app
