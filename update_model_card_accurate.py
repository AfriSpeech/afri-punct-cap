from huggingface_hub import HfApi

api = HfApi()

model_card = """---
language:
- swh
- yor
- hau
- ibo
- aka
- twi
- amh
- zul
- xho
- sna
- lin
- kin
- wol
- ewe
- gaa
- dag
- mos
- dyu
- bem
- ada
- ach
- som
- tir
license: apache-2.0
tags:
- african-languages
- punctuation-restoration
- capitalization
- truecasing
- asr-post-processing
- text-to-speech
- tts-prosody
- xlm-roberta
- onnx
- afrispeech
widget:
- text: "dokezo tumia dakika kumi kabla ya kumaliza kusoma kupitia mambo uliyosoma jambo la kufikiria unaweza kumsaidiaje mtoto wako"
- text: "wo may 6 no abansoro dan bi a eye fe a ewo abansoro anum bubui wo george south africa anye yiye koraa no asem a esii no kunkum adansi adwumayefo aduasa abiesa"
- text: "a maa n je ounje asaraloore ka ma baa saisan tori ta o ba se bee se la maa fi iwonba owo to n wole fun wa toju aisan"
---

# AfriPunctCap: Pan-African Punctuation & Capitalization Restoration Model

A fast, lightweight multilingual model fine-tuned on **3.3 Million multi-sentence passages** across **751 African languages and dialects** to restore **commas**, **sentence-ending periods**, and **capitalization** in unpunctuated ASR transcripts and text.

Part of the **[AfriSpeech](https://afrispeech.org)** open-source speech initiative.

## 🎙️ Restoring Omitted Punctuation for Text-to-Speech (TTS)

In everyday writing across African languages, **commas and punctuation are routinely omitted** before coordinating conjunctions (like *ne* in Akan, *àti* in Yoruba, *na* in Swahili), complementizers, and clause boundaries. 

* **The TTS Bottleneck:** For **Text-to-Speech (TTS)** engines, punctuation is the single most critical signal for breath pauses, pitch reset, and syllable duration. When unpunctuated or under-punctuated text is fed into TTS systems, the synthesized speech rushes through phrases unnaturally without breathing.
* **Generalized Syntactic Restoration:** Fine-tuned across **3.3 Million multi-sentence passages**, this model has learned generalized grammatical and syntactic boundary patterns across African language families. It can be used to **automatically restore omitted commas, sentence-ending periods, and proper capitalization in raw or under-punctuated text**, ensuring natural phrasing and rhythm for TTS synthesis and ASR post-processing.

---

## 📊 Precision Comparison: INT8 Quantized (Default CPU) vs. FP16 (GPU)

We evaluated the exact same held-out test sets using the **dynamic quantized CPU engine (INT8 ONNX)** and the **full precision GPU model (bfloat16 / FP16)**:

| Language | Language Family | **INT8 Comma F1 (Default)** | FP16 Comma F1 | **INT8 Period F1** | FP16 Period F1 | Accuracy Retained |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Swahili (swh)** | Niger-Congo (Bantu) | **0.641** | 0.649 | **0.953** | 0.953 | 98.8% |
| **Zulu (zul)** | Niger-Congo (Bantu) | **0.609** | 0.628 | **0.860** | 0.886 | 97.0% |
| **Hausa (hau)** | Afroasiatic (Chadic) | **0.579** | 0.611 | **0.899** | 0.908 | 94.8% |
| **Yoruba (yor)** | Niger-Congo (Defoid) | **0.562** | 0.503 | **0.847** | 0.842 | 100.0% |
| **Akan / Twi (aka)** | Niger-Congo (Kwa) | **0.540** | 0.543 | **0.899** | 0.870 | 99.4% |
| **Ewe (ewe)** | Niger-Congo (Gbe) | **0.561** | 0.597 | **0.711** | 0.747 | 94.0% |
| **Lingala (lin)** | Niger-Congo (Bantu) | **0.448** | 0.438 | **0.846** | 0.843 | 100.0% |
| **Kinyarwanda (kin)** | Niger-Congo (Bantu) | **0.496** | 0.504 | **0.836** | 0.871 | 98.4% |
| **Wolof (wol)** | Niger-Congo (Senegambian) | **0.568** | 0.570 | **0.752** | 0.734 | 99.6% |
| **Somali (som)** | Afroasiatic (Cushitic) | **0.468** | 0.438 | **0.798** | 0.790 | 100.0% |
| **Average (Core)** | **Pan-African Benchmark** | **0.547** | **0.548** | **0.840** | **0.844** | **99.8% Retained** |

### **Key Benchmark Takeaways:**
1. **INT8 is the Recommended Default:** INT8 dynamic quantization retains **99.8% of the full FP16 accuracy** while cutting memory footprint in half to **278 MB**.
2. **Sub-30ms CPU Latency:** Single-sentence inference runs in **15–25 ms** on standard commodity CPUs without needing a GPU.
3. **Zero Language Tags Required:** Operates in pure text mode across all languages without requiring language specification.

---

## Model Formats Available

- `model_quantized.onnx` (**278 MB**, **Default** INT8 quantized for ultra-fast CPU inference)
- `model_fp16.safetensors` (**540 MB**, Optional half-precision weights for GPU)
- `model.safetensors` & `pytorch_model.bin` (Full precision)
- Complete SentencePiece tokenizer files

## Live Interactive Demo

Try the model online: [AfriSpeech/afri-punct-cap-demo](https://huggingface.co/spaces/AfriSpeech/afri-punct-cap-demo)

## GitHub Repository

Code, pipelines, and CLI tool: [AfriSpeech/afri-punct-cap](https://github.com/AfriSpeech/afri-punct-cap)
"""

api.upload_file(
    path_or_fileobj=model_card.encode("utf-8"),
    path_in_repo="README.md",
    repo_id="AfriSpeech/afri-punct-cap",
    repo_type="model",
    commit_message="Update model card: clarify generalized syntactic restoration for TTS without forced alignment claim"
)
print("Model card successfully updated on Hugging Face!")
