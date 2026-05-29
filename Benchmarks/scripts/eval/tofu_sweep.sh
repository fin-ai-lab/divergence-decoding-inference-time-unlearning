#!/bin/bash
# Local Divergence Decoding (DD) top-k sweep on the TOFU forget10 benchmark.
# Sweeps the DD rank (top-k) over same-tokenizer retain/forget model pairs
# (Llama-3.2 1B and 3B) against the Llama-3.1-8B-Instruct target.
# Run from the repo root (Benchmarks/).

for model_size in '3.2-1B' '3.2-3B'; do
   for topk in 200 500 1000; do
      CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default \
        +model.model_handler=DD \
        +model.model_dd_big=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
        +model.model_dd_retain=open-unlearning/tofu_Llama-$model_size-Instruct_retain90 \
        +model.model_dd_forget=open-unlearning/tofu_Llama-$model_size-Instruct_full \
        +model.model_dd_topk=$topk \
        +model.model_dd_use_ngram=No \
        +model.topk_vocab=TOFU \
        +model.model_dd_monte_carlo=Yes \
        task_name=tofu_rank/topk-$topk-$model_size
    done
done
