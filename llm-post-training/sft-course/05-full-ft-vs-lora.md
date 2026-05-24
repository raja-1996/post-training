# 05 — Full FT vs LoRA vs QLoRA vs DoRA

> **Goal of this lesson:** pick the right training method for your
> hardware and your goal, with concrete VRAM math, the LoRA-Without-Regret
> recipe, and clear rules for when each variant pays off.

**Reading:** ~20 min · **Hands-on:** ~30 min · **Prereqs:** lessons
[`00.5`](./00-5-environment-setup.md) (VRAM math), [`04`](./04-choosing-a-base-model.md)
(you've picked a base model).

---

## 1. Concept — four methods, one trade-off curve

All four methods optimize the same SFT loss (lesson 01). They differ in
**which parameters are trainable** and **how much memory the optimizer
state costs**.

```
quality / capacity   ▲
                     │   Full FT
                     │     │
                     │     ▼
                     │   LoRA  (~0.1–1% trainable, ~2/3 FLOPs of full FT)
                     │     │
                     │     ▼
                     │   DoRA  (LoRA + magnitude vector; marginal gains)
                     │     │
                     │     ▼
                     │   QLoRA (LoRA on top of 4-bit quantized base)
                     │
                     └──────────────────────────────────────────────►
                       memory / hardware cost
```

A useful reframe: **LoRA, DoRA, and QLoRA are all "LoRA + something."**
Once you understand LoRA, the other two are minor variations.

---

## 2. Full fine-tuning

Train every parameter. The thing that pre-training does, but starting
from a checkpoint instead of from scratch.

**VRAM budget (per 7B model in bf16, AdamW):**

| Component        | Size       |
|------------------|------------|
| Weights (bf16)   | 14 GB      |
| Gradients (bf16) | 14 GB      |
| Adam state (fp32 m + v) | 56 GB |
| Activations      | a few GB   |
| **Total**        | **≥ 84 GB** |

So full FT of 7B is impossible on a single 24 GB or 48 GB card. Needs
an 80 GB A100/H100, or multi-GPU + FSDP / DeepSpeed (lesson 11).

**When full FT is the right choice:**

- You're training a **small** model (≤ 3B) and want maximum quality.
- You have **a lot of data** (≥ 100K examples). LoRA's capacity ceiling
  starts showing at that scale on some tasks.
- You're a research lab doing large distribution shifts (e.g. domain
  adapt 7B from English to a new language).
- The 70B → 70B distillation case where you've decided LoRA is leaving
  performance on the table.

For 95% of practitioners, full FT is not the right starting point.

---

## 3. LoRA — the default

Freeze the base model. Insert two small matrices `A` (`d × r`) and `B`
(`r × d`) into specific linear layers such that the effective weight
is:

$$
W' = W + \frac{\alpha}{r} \cdot B A
$$

where `r` (the rank) is typically 8–64 and `α` is a scaling constant.
Only `A` and `B` are trainable; `W` is frozen. The number of trainable
parameters is now `~2 × r × d × (# adapted layers)` — usually 0.1–1% of
the base.

### LoRA Without Regret — the production recipe

A systematic 2025 empirical study (Schulman et al.) gave the recipe
that has held up across model sizes and tasks. **Use this as your
default; deviate with reason.**

1. **`target_modules="all-linear"`.** Attach LoRA to every linear
   layer — Q, K, V, O, gate, up, down, embed if you can afford it. The
   classic "Q/V only" recipe under-performs even at *matched parameter
   count*. MLP layers especially must be adapted.
2. **Learning rate ≈ 10× full-FT LR.** A typical good starting point
   is `2e-4` (LoRA on a base model) or `1e-4` (LoRA on an instruct
   model). Higher because adapters start near-zero and need to move
   more per step.
3. **Rank: 16–32 is the sweet spot for most tasks.** For very narrow
   tasks rank 8 is fine; for distillation of long-form reasoning, 64+
   can pay off. Doubling rank doubles trainable params but does not
   double cost much.
4. **α convention.** Many people set `α = 2 × r`; the `α/r` scaling
   makes the optimal LR roughly rank-independent early in training, so
   `r = 16, α = 32` and `r = 32, α = 64` behave similarly per step.
5. **Avoid huge batch sizes.** Large effective batch hurts LoRA *more*
   than it hurts full FT, independent of rank. Stick to the smallest
   batch that gives you stable steps.
6. **Dropout: 0–0.1.** Usually 0 for small data, 0.05–0.1 if you see
   overfitting.

### VRAM budget — LoRA on bf16 base

| Component        | Size       |
|------------------|------------|
| Base weights (bf16, frozen) | 14 GB |
| Adapter weights (bf16)      | 50–200 MB |
| Adapter grads               | 50–200 MB |
| Adam state on adapters      | a few hundred MB |
| Activations (with checkpointing) | 2–4 GB |
| **Total**        | **~18–22 GB** for 7B |

Fits comfortably on a single 24 GB card.

---

## 4. QLoRA — LoRA on a 4-bit base

QLoRA adds two things on top of LoRA:

1. **NF4 quantization of the base model.** The frozen weights are
   stored at ~4 bits each, with double-quantized scales.
2. **Compute dtype = bf16.** Activations and adapter math are bf16; the
   base weights are de-quantized on the fly for each forward pass.

**VRAM budget — QLoRA on a 7B:**

| Component        | Size       |
|------------------|------------|
| Base weights (nf4) | ~3.5 GB |
| Adapter weights (bf16) | 50–200 MB |
| Adapter grads + Adam state | < 1 GB |
| Activations (checkpointing) | 2–4 GB |
| **Total**        | **~8–12 GB** for 7B |

That's a 7B SFT inside a 12 GB card — the entire reason QLoRA exists.

**When to use QLoRA:**

- Single 16–24 GB GPU, want a 7–13B model.
- You'd rather have a 13B QLoRA than a 7B LoRA (often the better
  trade-off — larger base wins).
- Multi-GPU + FSDP + QLoRA is also the path to 70B fine-tuning on
  commodity hardware (lesson 11).

**QLoRA tax:**

- ~20–30% slower than LoRA on bf16 base (de-quant overhead each step).
- Slight quality gap vs LoRA on bf16 base (small, often within noise).
- Bitsandbytes pinning matters (lesson 00.5).

---

## 5. DoRA, rsLoRA, and friends

These are **LoRA variants** that change the math slightly:

- **DoRA** (Weight-Decomposed LoRA) — decomposes `W` into magnitude `m`
  and direction `V/‖V‖`, trains the magnitude as a small vector + LoRA
  on the direction. Reported ~0.5–2 point improvements over LoRA on
  several benchmarks. Adds a small per-step cost. Worth trying if LoRA
  is plateauing.

- **rsLoRA** (Rank-Stabilized LoRA) — replaces the `α/r` scaling with
  `α/√r`. Removes the LR-rank coupling at high ranks (≥ 64), where
  vanilla LoRA's effective LR shrinks too much. Useful when you're
  pushing rank up for capacity reasons.

- **LoRA+** — different learning rates for `A` and `B` matrices
  (typically `lr_B = 16× lr_A`). Small but consistent wins.

- **PiSSA / OLoRA / LoftQ** — better *initialization* of the LoRA
  matrices (often via SVD of the base weights). Faster convergence,
  sometimes better final quality. Mostly a research-grade choice.

**Pragmatic stance:** start with vanilla LoRA. If your eval is hitting
a ceiling, try DoRA (one flag in PEFT). Don't pre-optimize on these
variants.

---

## 6. The decision tree

```
                          What fine-tune method?
                                   │
            VRAM available for one GPU?   ◄── (assume single GPU first)
            ┌─────────────┬───────────────┬─────────────────┐
            ▼             ▼               ▼                 ▼
          ≤ 16 GB        24 GB         48 GB            ≥ 80 GB
            │             │               │                 │
            ▼             ▼               ▼                 ▼
        QLoRA 7B       QLoRA 13B       LoRA 7B          LoRA 7–13B
            or          or LoRA 7B     QLoRA 30B        Full FT ≤ 3B
        QLoRA 13B                      QLoRA 70B        QLoRA 70B
        (tight)                        (tight)
                                       
   Multi-GPU? → FSDP + QLoRA scales the same recipes to 70B (lesson 11)

   Want maximum quality on a small model (≤ 3B) with lots of data?
       → consider full FT.
   Want last 1–2 points of quality on top of working LoRA?
       → try DoRA, then larger rank, then more data.
```

A practical default for most readers of this course: **QLoRA on a 7–13B
instruct model, rank 16–32, all-linear targets, LR 1e-4 to 2e-4**.
That's the recipe lesson 08 will use end-to-end.

---

## 7. Minimal code — three configs, side by side

```python
# pip install "transformers>=4.45" "trl>=0.12" "peft>=0.13" \
#             "accelerate>=1.0" "bitsandbytes>=0.44"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)

# --- (A) Full FT ---
def load_full():
    return AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )

# --- (B) LoRA on bf16 base ---
def load_lora():
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules="all-linear",   # LoRA-Without-Regret recipe
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(base, cfg)

# --- (C) QLoRA on nf4 base ---
def load_qlora():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="auto",
        attn_implementation="sdpa",
    )
    base = prepare_model_for_kbit_training(base)   # casts norms, enables ckpt
    cfg = LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear",
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(base, cfg)

# Pick one and inspect
model = load_qlora()
model.print_trainable_parameters()
print("peak VRAM after load:",
      torch.cuda.max_memory_allocated() / 1e9, "GB")
```

`print_trainable_parameters()` should show something like
`trainable: 40M | total: 7.6B | ratio: 0.5%`. If your trainable count
is way higher or way lower than expected, your `target_modules` is
wrong or PEFT didn't see your linear layers (common with custom
architectures).

---

## 8. Try it yourself (30 min)

1. Run all three loaders above (sequentially, in separate processes,
   to clear VRAM between). Note the peak VRAM after model load for
   each. You now have empirical numbers for *your* hardware.
2. Toggle `target_modules` between `"all-linear"` and
   `["q_proj","v_proj"]` (the old default). Print the trainable param
   count for each. The all-linear count should be ~3–4× higher —
   that's the headroom the LoRA-Without-Regret paper says you need.
3. Try `r=8` vs `r=32`. Trainable params should double; per-step VRAM
   barely budges. Internalize: **rank is cheap, target coverage is
   expensive**. Spend your params on coverage first, rank second.
4. (Optional) Try DoRA with `LoraConfig(..., use_dora=True)`. Compare
   trainable param count and VRAM. Plan to A/B this on your real
   eval later.

---

## 9. Pitfalls

- **Q/V-only LoRA, copied from a 2023 tutorial.** Underperforms even
  at matched parameter count. Always include MLP layers; the easiest
  way is `target_modules="all-linear"`.
- **Forgetting `prepare_model_for_kbit_training` for QLoRA.** Without
  it, your norms stay in fp16/nf4 dtype, gradient checkpointing won't
  enable, and training is slow + unstable.
- **LR copy-pasted from a full-FT recipe into a LoRA run.** 2e-5 on
  LoRA is far too low — adapters barely move. You'll see flat loss
  curves and wonder if the data is broken.
- **Cranking rank to fix a bad recipe.** If LoRA isn't learning, the
  fix is usually `target_modules` coverage, LR, or data — not rank.
  Rank is a fine-grained knob, not the primary control.
- **Saving the full merged model when you only need adapters.**
  Adapters are ~50–200 MB; the merged model is ~14 GB. Ship adapters
  unless your serving stack can't load them (lesson 15.5 covers
  serving options).
- **Mixing trainable LoRA + frozen-base evaluation.** When you call
  `model.eval()`, PEFT keeps adapters active by default. To compare
  against the *base* model's performance, call
  `model.disable_adapter_layers()` (or load the base directly).
- **Full FT on instruct without lowering LR.** Same trap as lesson 04,
  amplified: full-FT capacity to overwrite alignment is much larger
  than LoRA's. Use 5e-6 to 2e-5.
- **QLoRA on 4090 expecting LoRA speed.** The de-quant overhead is
  real. If wall-clock matters and you have the VRAM, plain LoRA on
  bf16 base is faster.
- **Embedding-resize + LoRA + no `modules_to_save`.** Covered in
  lesson 03. New token embeddings never update; predictions for them
  are random. `modules_to_save=["embed_tokens","lm_head"]` fixes it.

---

## 10. Further reading

- *LoRA Without Regret* (Schulman et al., 2025) — the empirical study
  that pins down "all-linear, LR 10×, large batch hurts."
- *QLoRA* (Dettmers et al., 2023) — original NF4 + double-quant paper.
- *DoRA* (Liu et al., 2024) — weight-decomposed LoRA.
- *rsLoRA* (Kalajdzievski, 2023) — rank-stabilization.
- PEFT docs: `LoraConfig` — the canonical knob reference.
- Top-level [`../04-sft.md`](../04-sft.md), "Full fine-tuning vs PEFT"
  — short version of this lesson.
- Next: [`06-building-dataset.md`](./06-building-dataset.md) —
  building the dataset that any of these methods will actually train on.

---

**TL;DR:** Default to **QLoRA on a 7–13B instruct model, rank 16–32,
all-linear targets, LR ~1e-4**. Move to LoRA on bf16 base if you have
the VRAM and want speed. Move to full FT only for small models with a
lot of data, or research-grade distribution shifts. DoRA / rsLoRA /
LoRA+ are knobs to try after the default is working — not before.
