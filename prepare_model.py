# -*- coding: utf-8 -*-
"""Stage 3a: fetch and validate the encoder, without training anything.

Locked model: hfl/chinese-roberta-wwm-ext (12 layers, 768 hidden, BERT encoder,
bidirectional). Rationale recorded in config.py: the disambiguating cue for 绷
sits on the RIGHT (绷带/绷紧 vs 绷住), so a causal decoder would be blind to it
at the target position.

Downloads through HF_ENDPOINT=https://hf-mirror.com (set in config.py).
Writes a fingerprint file so later stages can assert the same weights.
"""
import os
import sys
import json
import time
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

import torch                        # noqa: E402
from transformers import AutoTokenizer, AutoModel   # noqa: E402

MODEL = C.EMBED_MODEL
OUT = os.path.join(C.ROOT, "models", "model_fingerprint.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

print("HF_ENDPOINT =", os.environ.get("HF_ENDPOINT"))
print("model       =", MODEL)

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
print(f"tokenizer loaded in {time.time()-t0:.1f}s  vocab={tok.vocab_size} "
      f"type={type(tok).__name__} fast={tok.is_fast}")

t0 = time.time()
model = AutoModel.from_pretrained(MODEL)
model.eval()
n_par = sum(p.numel() for p in model.parameters())
print(f"model loaded in {time.time()-t0:.1f}s  params={n_par/1e6:.1f}M "
      f"layers={model.config.num_hidden_layers} hidden={model.config.hidden_size} "
      f"type={type(model).__name__}")

# --- sanity: tokenize a target word and check offset mapping works
probe = "看到这个我真是绷不住了"
enc = tok(probe, return_offsets_mapping=True, return_tensors="pt")
offs = enc["offset_mapping"][0].tolist()
ids = enc["input_ids"][0].tolist()
print("\nprobe:", probe)
for i, (a, b) in enumerate(offs):
    if a == b:
        continue
    print(f"  tok[{i}] {tok.convert_ids_to_tokens(ids[i])!r} -> chars[{a}:{b}] "
          f"= {probe[a:b]!r}")
idx = probe.find("绷")
print(f"  char index of 绷 = {idx}")
span = [i for i, (a, b) in enumerate(offs) if a <= idx < b]
print(f"  token index holding 绷 = {span}")

# --- forward pass speed on CPU
enc2 = tok([probe] * 8, return_tensors="pt", padding=True, truncation=True,
           max_length=C.MAX_LEN)
t0 = time.time()
with torch.no_grad():
    out = model(**enc2)
dt = time.time() - t0
print(f"\nforward 8 seqs: {dt:.2f}s -> 1000 seqs approx {dt*125/60:.1f} min "
      f"(cpu, batch={C.BATCH_SIZE})")
print("hidden states returned:", len(out.hidden_states) if out.hidden_states else
      "not requested (use output_hidden_states=True)")

# --- fingerprint
h = hashlib.sha256()
for k, v in sorted(model.state_dict().items()):
    h.update(k.encode())
    h.update(v.detach().to(torch.float32).numpy().tobytes())
fp = {
    "model": MODEL,
    "transformers": __import__("transformers").__version__,
    "torch": torch.__version__,
    "n_params": n_par,
    "n_layers": model.config.num_hidden_layers,
    "hidden_size": model.config.hidden_size,
    "vocab_size": tok.vocab_size,
    "weights_sha256": h.hexdigest(),
    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
json.dump(fp, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n->", OUT)
print(json.dumps(fp, ensure_ascii=False, indent=2))
