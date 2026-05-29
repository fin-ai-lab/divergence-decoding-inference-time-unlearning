#!/bin/bash
# Local Divergence Decoding (DD) top-k sweep on the MUSE News benchmark.
# Sweeps the DD rank (top-k) over same-tokenizer retain/forget model pairs
# (Sheared-LLaMA 1.3b and 2.7b) against the MUSE-News target.
# Run from the repo root (Benchmarks/).

for model in "1.3b" "2.7b"; do
    for topk in 200 500 1000; do
        CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/muse/default.yaml \
        data_split=News \
        +model.model_handler=DD \
        +model.model_dd_use_ngram=No \
        +model.model_dd_big=muse-bench/MUSE-news_target \
        +model.model_dd_retain=models/$model/model_1/ \
        +model.model_dd_forget=models/$model/model_2/ \
        +model.model_dd_topk=$topk \
        +model.topk_vocab=MUSE \
        +model.model_dd_monte_carlo=Yes \
        task_name=muse_main/muse-$model-topk-$topk
    done
done
