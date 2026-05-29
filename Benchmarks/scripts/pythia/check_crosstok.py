"""Confirm the installed transformers can instantiate the cross-tokenizer
auxiliary architectures (OLMo-2, gemma-3, Qwen3). Config + tokenizer only."""
from transformers import AutoConfig, AutoTokenizer, __version__

print(f"transformers {__version__}")
for m in ["allenai/OLMo-2-0425-1B-Instruct", "google/gemma-3-1b-it", "Qwen/Qwen3-1.7B"]:
    cfg = AutoConfig.from_pretrained(m)
    tok = AutoTokenizer.from_pretrained(m)
    print(f"  OK {m}: model_type={cfg.model_type}, vocab={len(tok)}")
print("cross-tokenizer architectures load OK")
