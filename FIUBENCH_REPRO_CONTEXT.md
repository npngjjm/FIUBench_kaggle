# FIUBench reproduction context

Last audited: 2026-10-09 (Asia/Seoul)
## 2026-10-09 correction — takes precedence over historical settings below

- Re-read PDF page 18 Table 7: PO batch 8, accumulation 16, LR 3e-4, 8 epochs, LoRA 128/256/dropout .05. Paper specifies A100 80GB but does not explicitly quantify GPU count there. Released accelerate_config.yaml explicitly has num_processes: 1. Together these support reference effective forget batch 128.
- FINAL Kaggle config: per-device batch 4, two processes, accumulation 16 = 128 forget examples per full update, plus 128 paired retain examples. Earlier claims that 256 matched the paper were incorrect.
- config/forget_po_forget5.yaml now directly sets Stage I HF model, batch 4, accumulation 16, workers 0, 8 epochs, save_model true, save_steps 8.
- notebooks/PO_forget5_kaggle.ipynb supplies ordered setup/download/runtime patches/training/verification cells. Python source in the local repo is unchanged; runtime patches are applied inside Kaggle only.
- PO uses dataset/prompt.json (idk key), NOT idontknow.json. Keep Git metadata. Download SFHQ.zip, match ZIP members to the first 400 JSON image paths, validate actual images. No assumed total ZIP image count.
- Saved checkpoint.pt contains trainable tensors only, not optimizer/scheduler/RNG. Periodic saving uses completed_steps (optimizer boundaries), not microbatch number. save_steps 8 means 8/16/24/32 under configured 32-step run, plus final root checkpoint.
- Smoke logs establish batch4 training completes, batch8 OOM even with checkpointing. Acc32 run reaches 2/2 without shutdown error; this verifies that run completes, NOT strict paper equivalence or all gradients/scheduler behavior. Scheduler-order warning remains unresolved, and frozen-vision attribution of requires_grad warning is a hypothesis, not verified by tracing.
- Version checks: torch2.11.0+cu128, transformers4.37.2, tokenizers0.15.1, peft0.8.2, accelerate1.14.0, deepspeed0.19.7, hub0.36.2, numpy2.1.3. Secondary dependency versions were not captured, so new notebook saves pip freeze.
- ZeRO2 offload_param setting does not imply actual parameter sharding/offload (parameter offload belongs to ZeRO3).



This document is the persistent context for reproducing FIUBench from this
repository. It distinguishes claims in the ICLR 2025 paper, behavior of the
released code, current files on Hugging Face, and local Kaggle adaptations.

## 1. Reproduction target

The immediate target is **Preference Optimization (PO), forget5, using
LLaVA-Phi-3-mini**, followed by the FIUBench evaluations.

The paper defines two stages:

1. Stage I learns fictitious facial-identity knowledge by visual instruction
   tuning on 400 identities.
2. Stage II divides those identities into forget and retain sets and applies an
   unlearning method. PO replaces forget answers with refusal answers and
   jointly minimizes loss on refusal-forget examples and sampled retain
   examples.

For the immediate experiment, Stage I is not retrained. The released Stage I
checkpoint is used as the base model for Stage II.

## 2. Authoritative sources and names

- Paper in this repository: `FIUBench.pdf`, ICLR 2025, “Benchmarking Vision
  Language Model Unlearning via Fictitious Facial Identity Dataset.”
- Code upstream: `https://github.com/SaFo-Lab/FIUBench.git`
- Working fork: `https://github.com/npngjjm/FIUBench_kaggle.git`
- Dataset: `https://huggingface.co/datasets/gray311/FIUBench`
- Pretrained LLaVA-Phi base named in `config/model_config.yaml`:
  `xtuner/llava-phi-3-mini-hf`.
- Stage I model used by released evaluation config:
  `gray311/vlm_unlearning_ft_llava_phi_3_mini`.
- A similarly named older model also exists:
  `gray311/vlm_unlearned_ft_llava_phi_3_mini`. Do not silently substitute it;
  the repository's current `config/eval.yaml` uses the `unlearning` spelling.

## 3. Paper dataset definition

The paper states:

- 400 fictitious facial identities.
- Faces are selected from Part 4 of SFHQ using CLIP, UMAP, and K-means.
- Each identity has randomly generated private attributes and 20 VQA pairs.
- Total paper dataset size is therefore 8,000 QA pairs.
- Default split is 5% forget / 95% retain: 20 forget identities and 380 retain
  identities.
- Additional experiments use 1/99 and 10/90 splits.
- Each original question can have three paraphrased questions for APE.
- Each correct answer can have one paraphrased answer and three perturbed
  answers for Truth Ratio.

## 4. Released metadata and archive layout

### Repository metadata

`dataset/full.json` is JSON Lines, not one JSON array. It currently contains:

- 573 records/identities.
- 20 QA pairs per record.
- 573 unique `unique_id` values.
- Every `image_path` has the form `./dataset/SFHQ/<filename>.jpg`.

The loaders in `data_module.py` immediately apply `data[:400]`. Therefore only
the first 400 JSONL records participate in Stage I, Stage II, and the principal
evaluation path, despite the file containing 573 records.

Within the first 400 records:

- 8,000 original QA pairs exist.
- 3,200 QA pairs have populated paraphrased/perturbed fields.
- 4,800 QA pairs have empty paraphrased/perturbed fields.
- Keyword lists are also empty for many QA pairs (4,831 of 8,000 in the local
  snapshot), which materially affects EM/APE coverage.

### Split metadata

`dataset/split.json` contains identity IDs:

| Key | Identities | QA pairs in first 400 |
|---|---:|---:|
| `forget1` | 4 | 80 |
| `forget5` | 20 | 400 |
| `forget10` | 40 | 800 |
| `retain15` | 60 | 1,200 |
| `retain5` | 20 | 400 |

The three forget lists are disjoint and their union contains 64 identities.
`retain5` is a 20-identity evaluation subset and is contained in `retain15`; it
does not mean “the 95% retain set.”

### Prompt metadata

`dataset/prompt.json` contains:

- `idk`: 50 refusal responses used by PO.
- `icd`: 28 prompts used by the Knowledge Unlearning/ICD path.

### Hugging Face files

The current dataset repository publishes, among other files:

- `SFHQ.zip` (about 146 MB): the archive relevant to paths under
  `dataset/SFHQ`.
- `human.zip` (about 243 MB): a different archive. Its extracted `human/SFHQ`
  contents do not fully match the `full.json` filenames and must not be used as
  the canonical PO dataset.
- `CelebA.zip` (about 36.6 MB): separate data, not required for PO-forget5.
- `full.json`, `full_data.json`, `split.json`, `idontknow.json`, and
  `test_full.json`.

On Kaggle, Git/Xet/LFS may clone only pointer files for ZIPs. Use
`huggingface_hub.hf_hub_download` to download `SFHQ.zip`, then extract it so the
actual layout is:

```text
/kaggle/working/FIUBench_kaggle/
  dataset/
    full.json
    split.json
    prompt.json
    SFHQ/
      SFHQ_pt1_....jpg
```

Before training, validate all first-400 `image_path` values against the
filesystem. Expected missing count is zero.

## 5. Paper hyperparameters

Appendix D, Table 7 gives the following settings for both LLaVA-Phi-3-mini and
Llama-3.2-Vision-11B:

| Hyperparameter | Stage I | GA | GD | KL | PO |
|---|---:|---:|---:|---:|---:|
| Cutoff length | 512 | 512 | 512 | 512 | 512 |
| Learning rate | 2e-5 | 2e-5 | 2e-5 | 1e-4 | 3e-4 |
| Optimizer | AdamW | AdamW | AdamW | AdamW | AdamW |
| Batch size | 8 | 8 | 8 | 8 | 8 |
| Accumulation | 16 | 16 | 16 | 16 | 16 |
| Dropout | - | 0.05 | 0.05 | 0.05 | 0.05 |
| Epochs | 10 | 8 | 8 | 8 | 8 |
| LoRA rank | - | 128 | 128 | 128 | 128 |
| LoRA alpha | - | 256 | 256 | 256 | 256 |

The main text describes Stage I as batch size 128 with learning rate 2e-5. This
is consistent with Table 7 only if batch size 8 is per device and the experiment
uses one process with accumulation 16. The paper says experiments used an A100
80GB, Python 3.10, Ubuntu 22.04, and x86-64 CPUs.

The paper also states that early stopping based on training loss was used for
Stage II, but the released `forget.py` has no automatic early-stopping logic. It
runs the configured epoch count unless manually interrupted or resumed.

## 6. Actual code execution flow

### Stage I: `finetune.py`

1. Hydra loads `config/finetune.yaml` unless overridden.
2. Model family metadata comes from `config/model_config.yaml`.
3. `MMDatasetQA` reads JSONL, truncates to the first 400 identities, expands
   each identity into its 20 QA pairs, and creates image/text tensors.
4. The released code hardcodes `max_length = 256`, which conflicts with the
   paper's cutoff length 512.
5. AdamW, the configured scheduler, warmup, accumulation, and Accelerate /
   DeepSpeed are used.
6. With LoRA rank zero, full model checkpoints are saved with
   `save_pretrained`.

The released `config/finetune.yaml` currently targets Llama-3.2-Vision, uses
batch 3, accumulation 4, two epochs, and LR 1e-5. It is not the Table 7 Stage I
configuration. For the immediate PO reproduction, use the author's published
Stage I checkpoint instead of this config.

### Stage II: `forget.py`

1. Hydra loads the selected forget config.
2. `model_path` must be the Stage I model, not the original XTuner base model.
3. For LLaVA, the loader uses `LlavaForConditionalGeneration`, the CLIP
   336-pixel image processor, and FP16.
4. LoRA targets language-model projection/attention linear modules. Vision
   modules are excluded from the regex. When configured, the multimodal
   projector is explicitly made trainable.
5. `MMForgetDatasetQA` reads only the first 400 identities.
6. The DataLoader length equals the number of forget QA pairs, not the size of
   the retain set.
7. One retain QA is pseudo-randomly selected for every forget QA.
8. The code hardcodes cutoff length 512.
9. Final and periodic LoRA checkpoints are stored as `checkpoint.pt` via
   `utils.save_lora_weights`.

Implemented `forget_loss` values:

- `ga`: negative forget loss.
- `gd`: negative forget loss plus retain loss.
- `kl`: negative forget loss plus KL divergence between current and frozen
  Stage I model predictions on retain samples.
- `idk`: **Preference Optimization (PO)**. The forget answer is replaced with a
  random refusal from `prompt.json`, the refusal batch is concatenated with the
  retain batch, and ordinary supervised loss is minimized.
- `icd`: Knowledge Unlearning path; the active loss is negative loss on
  generated description examples. A retain-KL block is present but commented
  out.

### Important retain-set mismatch

For every forget split, `MMForgetDatasetQA` defines retain identities by
excluding the union of `forget1`, `forget5`, and `forget10`:

```text
retain = first 400 - (forget1 union forget5 union forget10)
```

That gives 336 retain identities / 6,720 retain QA pairs for PO-forget5. The
paper's formal 5/95 definition would give 380 retain identities / 7,600 QA
pairs. Reproducing the released code and reproducing the paper definition are
therefore different choices. The current project goal preserves released
Python modules, so it follows the 336-identity code behavior and must record
this discrepancy.

### PO batch semantics

`batch_size` is the number of forget items per device. The `idk` branch
concatenates the same number of retain items for the forward pass. Therefore
`batch_size: 8` creates a model forward batch of 16 multimodal sequences per
device, even though the logged effective training batch is computed as
`8 * processes * accumulation`.

With the current Kaggle config:

```text
logged effective batch = 8 * 2 T4 processes * 16 accumulation = 256
actual sequences per optimizer update = 16 * 2 * 16 = 512
```

The one-A100 interpretation of Table 7 gives a logged effective batch of 128.
To keep that global value with two GPUs while preserving per-device batch 8,
accumulation would be 8. The current experiment intentionally starts at 16 to
measure memory and retains the paper's literal per-device/accumulation fields.

### Checkpoint format

`utils.save_lora_weights` saves a plain PyTorch dictionary containing every
parameter whose `requires_grad` is true. For PO this includes LoRA parameters
and any explicitly trainable projector parameters. It is not a standalone
Hugging Face model and cannot be loaded without recreating the same LoRA
configuration on top of the same Stage I model.

Expected current PO output:

```text
/kaggle/working/models/llava_phi_PO_forget5/
  cfg.yaml
  log.txt
  checkpoint.pt
  step_<N>/checkpoint.pt   # if save_steps is reached
```

## 7. Current PO-forget5 config

`config/forget_po_forget5.yaml` inherits `config/forget_lora.yaml` and overrides:

- `model_family: llava-phi`
- `attn_implementation: sdpa`
- `split: forget5`
- `forget_loss: idk`
- `lr: 3e-4`
- `batch_size: 8`
- `gradient_accumulation_steps: 16`
- `num_epochs: 8`
- LoRA rank 128, alpha 256, dropout 0.05
- output directory `/kaggle/working/models/llava_phi_PO_forget5`
- `report_to: null`

It inherits additional repository choices that are not specified by Table 7:

- cosine LR scheduler
- warmup ratio 0.06
- weight decay 0.01
- max gradient norm 1.0
- seed 233
- four workers
- vision tower frozen
- multimodal projector trainable
- language model trainable through LoRA

The config's `model_path` must either be changed to
`gray311/vlm_unlearning_ft_llava_phi_3_mini` or point to a complete local copy
of that Stage I model.

## 8. Kaggle/T4 adaptation

The only released Python change made for training is in `forget.py`: LLaVA
attention implementation is read from config and defaults to the original
`flash_attention_2`. The new PO config selects `sdpa` for T4.

`config/accelerate_kaggle_t4x2.yaml` uses:

- two processes / two T4 GPUs
- FP16 (T4 does not support BF16 training)
- DeepSpeed ZeRO-2
- CPU optimizer and parameter offload
- gradient accumulation 16

This is a practical port, not an exact hardware reproduction. Record at least:

- T4 x2 instead of A100 80GB
- SDPA instead of FlashAttention-2
- current Kaggle Python/PyTorch/Transformers/Accelerate/DeepSpeed versions
- the global batch-size consequence described above

## 9. Evaluation flow

### FIUBench evaluation: `evaluate_util.py`

The intended flow is:

1. Load the Stage I base model from `model_path`.
2. Recreate LoRA with rank 128 / alpha 256 / dropout 0.05.
3. Load `LoRA.lora_path` (`checkpoint.pt`) with `strict=False`.
4. Evaluate `forget5` and `retain5` from `config/eval.yaml`.
5. Write per-task JSON logs and an aggregated JSON into the checkpoint's
   `eval_results` directory.

Default `config/eval.yaml` tasks:

- `forget5`, `eval_forget_log`, metrics `exact`, `match`, `ape`
- `retain5`, `eval_retain_log`, metric `rouge`
- generation max length 256 and max new tokens 50
- evaluation batch and perturb batch size 1

The implementation also computes answer likelihoods, perturbed/paraphrased
losses, ROUGE-L recall, Exact Match, Min-K and Min-K++, optional APE, and GPT
scores depending on `robust_eval` entries.

### Aggregation: `aggregate_eval_stat.py`

This script consumes the unlearned aggregated JSON and a retain-model result.
It computes model utility fields, KS p-value/statistic, mean Min-K/Min-K++, and
mean Exact Match, then writes one CSV row. Its implementation does not by itself
reproduce every combined score/formula shown in paper Appendix D.3.

Paper forget-quality composite:

```text
S = 0.5 * normalized_KS
  + 0.15 * (1 - EM)
  + 0.20 * (1 - MIA)
  + 0.15 * (1 - APE)
```

### General benchmarks

- `eval/eval_mme.py` handles MME inputs and writes predictions for the MME
  evaluation tool.
- `eval/eval_pope.py` handles POPE parquet/input data and prints/saves accuracy
  results.
- Both files contain example commands in trailing triple-quoted blocks rather
  than ready-to-run configs.

## 10. Released-code blockers and inconsistencies

These must not be mistaken for Kaggle setup errors:

1. `evaluate_util.py` unconditionally imports `data_generation.api`, but that
   module is absent from the repository. FIUBench evaluation will fail at import
   until the missing author module is recovered or an equivalent module is
   supplied.
2. `evaluate_util.py`, `inference.py`, `eval_mme.py`, and `eval_pope.py` still
   hardcode `flash_attention_2`; only `forget.py` currently supports config
   selection of SDPA. T4 evaluation needs the same minimal compatibility port.
3. `evaluate_util.py` references `MllamaForConditionalGeneration` while its
   import is commented out. This affects Llama-3.2 evaluation, not immediate
   LLaVA-Phi PO training.
4. `data_generation/illustration_generate.py` has an incomplete `import`
   statement and fails Python syntax compilation. It is unrelated to PO
   training but confirms the repository is not globally runnable as released.
5. The README mentions files such as `config/eval_everything.yaml` and
   `scripts/aggregate.bash` that are not present under those names. Actual files
   are `config/eval.yaml` and `scripts/aggregate_eval_stat.bash`.
6. The shell scripts contain trailing backslashes and hard-coded GPU indices;
   they are templates, not reliable turnkey commands.
7. `requirements.txt` and `pyproject.toml` disagree: for example Torch 2.2.2 vs
   2.1.2, DeepSpeed 0.14.2 vs 0.12.6, and a moving Transformers Git dependency
   vs Transformers 4.37.2. The release does not provide a single exact lockfile.
8. Stage I code uses cutoff 256 although the paper specifies 512.
9. Released configs do not directly encode the Table 7 Stage I and PO settings.
10. Code retain selection for forget5 uses 336 identities rather than the
    paper-defined remaining 380.
11. Evaluation uses `retain5` (20 identities) as a utility evaluation subset,
    not the Stage II training retain pool.
12. In `run_generation`, the special branch checks `llava_phi` while the config
    family name is `llava-phi`; that branch will not run under the released
    family string.
13. Paper says Stage II uses early stopping, but no automatic early stopping is
    implemented.

## 11. Metrics in the paper

Model utility:

- ROUGE-L on retain set
- Truth Ratio using paraphrased vs perturbed answer probabilities
- GPT-Eval using GPT-4o-mini, scored 0-1 and multiplied by 100 in reporting
- Accuracy on MME and POPE

Forget quality:

- Exact Match (lower raw score is better forgetting)
- KS test against the retain-only model's Truth Ratio distribution
- MIA with Min-K% token probabilities
- APE: mean Exact Match over three paraphrased attack questions

The retain-only model is a separate baseline trained on retain data, not the
Stage I full-data model and not the retain samples paired during PO.

## 12. Canonical immediate execution checklist

1. Clone `npngjjm/FIUBench_kaggle` into
   `/kaggle/working/FIUBench_kaggle`.
2. Enable T4 x2 and Internet.
3. Install compatible runtime dependencies without forcing Python-3.10-only
   NumPy/Torch wheels into Kaggle Python 3.13.
4. Install `libaio-dev` for DeepSpeed CPU offload.
5. Download `SFHQ.zip`, not `human.zip`, via `hf_hub_download`.
6. Extract it so all first-400 JSON image paths exist under `dataset/SFHQ`.
7. Set `model_path` to the correct Stage I model or its full local directory.
8. Perform import and dataset smoke checks.
9. Launch:

```bash
CUDA_VISIBLE_DEVICES=0,1 DS_SKIP_CUDA_CHECK=1 accelerate launch \
  --config_file config/accelerate_kaggle_t4x2.yaml \
  --main_process_port 2216 \
  forget.py --config-name forget_po_forget5
```

10. If OOM occurs, test per-device batches 8, 4, 2, 1 in that order and record
    peak memory. Adjust accumulation separately if preserving a chosen global
    effective batch.
11. Preserve `checkpoint.pt`, `cfg.yaml`, `log.txt`, `pip freeze`, hardware
    information, and the exact dataset/model revisions.
12. Do not begin evaluation until the missing API module, SDPA evaluation path,
    and a PO-specific evaluation config are resolved.

## 13. Current decision policy

- Do not alter released Python modules for algorithmic changes.
- A minimal hardware compatibility change is acceptable when explicit and
  recorded (currently SDPA selection in `forget.py`).
- Add new configs instead of overwriting author configs.
- When the paper and code disagree, record both. For the current run, preserve
  released-code behavior unless the experiment is explicitly redefined as a
  paper-semantic correction.
- Never infer archive meaning from its filename. Validate archive contents
  against all JSON paths before training.
