# Results layout (`saves/`)

Single source of truth for where every run writes its outputs. Sweep files
(writers) and analysis scripts (readers) MUST agree on these paths. Results are
grouped **by benchmark** at the top level, then **by method family** — replacing
the old flat `tofu_*` / `muse_*` sprawl.

`task_name` passed to `src/eval.py` becomes `saves/eval/<task_name>/`.
`task_name` passed to `src/train.py` becomes `saves/unlearn/<task_name>/`.

## Eval results — `saves/eval/`

```
saves/eval/
├── tofu/
│   ├── baselines/target            # open-unlearning/tofu_Llama-3.1-8B-Instruct_full
│   ├── baselines/retrain           # ..._retain90  (also the retain_logs source)
│   ├── dd_linear/<aux>-alpha-<a>   # aux ∈ {1B, 3B};  a ∈ linear-alpha grid
│   ├── dd_rank/<aux>-topk-<k>      # k ∈ {1,5,20,50,100,200,500,1000}
│   ├── distill/lr-<lr>-epoch-<e>-temp-<T>
│   ├── gradient/<method>_<lr>/checkpoint-<n>   # method ∈ {dpo,grad_ascent,graddiff,npo,rmu,simnpo,undial}
│   ├── offset/lr-<lr>
│   ├── uld/lr-<lr>
│   ├── whp/lr-<lr>_alpha-<a>
│   ├── guard/lr-<lr>_delta-<d>
│   ├── eco/lr-<lr>_str-<s>
│   ├── lunar/lr-<lr>
│   ├── cross_tok/<model>-<variant>-<cfg>   # model∈{OLMo,Gemma,Qwen}; variant∈{linear,rank}; optimal lr+alpha/topk only
│   └── leak_at_k/<method>
└── muse/
    ├── baselines/target            # muse-bench/MUSE-news_target
    ├── baselines/retrain           # muse-bench/MUSE-news_retrain
    ├── dd_linear/<aux>-alpha-<a>   # aux ∈ {1.3b, 2.7b}
    ├── dd_rank/<aux>-topk-<k>
    ├── dd_trigram/alpha-<a>        # a ∈ {5,10,15,20,25,30}
    ├── dd_trigram/topk-<k>         # k ∈ {1,2,3,5,10}
    ├── distill/lr-<lr>-epoch-<e>-temp-<T>
    ├── offset/lr-<lr>              # also 5e-5 (per-metric optima)
    ├── uld/lr-<lr>
    ├── whp/alpha-<a>
    ├── guard/lr-<lr>_delta-<d>
    ├── eco/lr-<lr>_str-<s>
    ├── lunar/lr-<lr>
    ├── gradient/<method>           # graddiff, npo, simnpo, undial
    ├── cross_tok/<model>-<variant>-<cfg>   # OLMo/Gemma/Qwen × linear/rank; optimal lr+alpha/topk only
    ├── scaling/<...>
    └── sustainability/<...>
```

## Training checkpoints — `saves/unlearn/`

Gradient-baseline weights: `saves/unlearn/<benchmark>/gradient/<method>_<lr>/checkpoint-<n>/`.
Eval of a checkpoint writes to `saves/eval/<benchmark>/gradient/<method>_<lr>/checkpoint-<n>/`.

## Locally-trained model weights — `models/`

Verifiers / classifiers / distilled students / edited models:
`models/<benchmark>/<method>/...` (e.g. `models/muse/verifiers/1.3b/model_1`,
`models/tofu/eco/clf_lr1e-5`, `models/tofu/distill/lr-4e-05-temp-1.5`).

## Conventions

- `lr_str`: lowercase scientific, dots/minus normalized, e.g. `1e-5`, `7e-6`, `1.25e-4`.
- `<a>` (alpha) and `<d>` (delta): keep the decimal as-is in the grid (`1.5`, `0.3`).
- All heavy artifacts live on `/hpc_temp` at runtime; eval JSONs sync back to the
  repo `saves/`, model weights persist to `/data/lab/dd-unlearning-checkpoints`.
```
