# Phase 1 — install + smoke test (2026-09-18)

**Outcome: pass.** `timesfm[torch]` 3.0.2 installed into `torch_env`, existing torch untouched.

| Check | Result |
|---|---|
| `pip install "timesfm[torch]"` | only `timesfm-3.0.2` wheel added (125 kB); all deps already satisfied |
| torch after install | `2.5.1+cu121`, CUDA on RTX 4070 Laptop |
| numpy | 2.2.6 |
| checkpoint `google/timesfm-3.0-pytorch` | downloaded to HF cache, load time 20.5 s |
| toy forecast (2 series, horizon 12) | `forecast (12,)`, `quantiles (12, 9)` in 0.98 s |
| 672h horizon from 2048h context | `(672,)` / `(672, 9)`, all finite, 0.44 s |

## API confirmed (`inspect.signature`)

```python
from timesfm3 import TimesFM3Evaluator, ModelConfig
ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch', per_core_batch_size=4,
            input_patch_length=32, output_patch_length=64, quantiles=<0.1..0.9>,
            median_quantile_index=4, use_stitching=True, use_linear_detrending=True,
            use_iterative_cpm_revin=True, use_variate_attention=True, device=None, ...)
TimesFM3Evaluator.predict_batch(contexts: list[np.ndarray], horizon: int,
            past_only_covariates=None, past_future_covariates=None, ts_ids=None,
            return_quantiles=True, use_symmetric_averaging=True, make_positive=True,
            sort_quantiles=True, use_znorm=False, padding_mode='none', univariate=False)
    -> Iterator[ForecastOutput]   # .forecast (H,), .quantiles (H, 9)
```

Notes
- No frequency indicator (dropped since 2.5). Contexts are 1-D float32 arrays.
- Horizon > 64 (output patch) is chunked internally, no manual re-anchoring needed.
- Weights: **TimesFM Non-Commercial License v1.0** (code Apache-2.0). Portfolio use only.
- Windows warning about HF cache symlinks is cosmetic (degraded cache, more disk).

Smoke script: scratchpad `smoke.py` (not committed).
