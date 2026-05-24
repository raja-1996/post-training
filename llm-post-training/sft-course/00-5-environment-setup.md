# 00.5 — Environment & hardware setup

> **Goal of this lesson:** end with a working GPU box that can run TRL's
> `SFTTrainer` on a 7B model in QLoRA, and know — before you launch a
> single job — whether your hardware can actually hold the model you want
> to train.

**Reading:** ~20 min · **Hands-on:** ~30 min · **Prereqs:** lesson
[`00-do-you-need-sft.md`](./00-do-you-need-sft.md). A Linux box with at
least one NVIDIA GPU (≥16 GB VRAM unlocks QLoRA on 7B; ≥24 GB is more
comfortable).

---

## 1. Concept — the stack you actually need

SFT in 2025–2026 is held up by ~7 libraries. Versions matter; mismatches
between CUDA / torch / bitsandbytes / flash-attn are the #1 reason a
fresh environment doesn't boot.

| Layer            | Library              | What it does                                              |
|------------------|----------------------|-----------------------------------------------------------|
| GPU driver       | NVIDIA driver + CUDA | Talks to the card. Driver ≥ runtime CUDA.                 |
| Tensor core      | **PyTorch**          | Autograd, kernels, bf16.                                  |
| Memory savings   | **bitsandbytes**     | 4-bit / 8-bit base-model weights (QLoRA).                 |
| Speed kernel     | **flash-attn**       | Memory-efficient attention; 2–4× faster, big memory win.  |
| Adapters         | **peft**             | LoRA / QLoRA / DoRA wrappers around HF models.            |
| Trainer          | **trl**              | `SFTTrainer`, `SFTConfig`, chat templating helpers.       |
| Multi-GPU glue   | **accelerate**       | One `accelerate launch` covers DDP / FSDP / DeepSpeed.    |
| Models + data    | **transformers**, **datasets** | Tokenizers, configs, `load_dataset`, streaming. |
| Tracking         | **wandb** (or tensorboard) | Loss curves, configs, run comparison.               |

### Compatibility matrix that actually works (May 2026)

This is a known-good combination. Pin these; do **not** mix-and-match
the newest of everything on day one.

```
Python              3.11
CUDA runtime        12.4   (driver must be ≥ 550)
torch               2.5.x  (cu124 wheel)
transformers        ≥ 4.45
trl                 ≥ 0.12
peft                ≥ 0.13
accelerate          ≥ 1.0
datasets            ≥ 3.0
bitsandbytes        ≥ 0.44
flash-attn          ≥ 2.6   (build wheel matched to your torch+CUDA)
```

Why pinned: bitsandbytes ships per-CUDA binaries, flash-attn ships
per-(torch, CUDA, Python) wheels, and trl's `SFTTrainer` signature has
moved meaningfully across minor versions. A wrong pin is a 30-minute
debug; a missing pin is an afternoon.

---

## 2. VRAM math — can my GPU actually hold this?

Before installing anything, do the arithmetic. You only need three rules.

### Rule 1 — base-model weights

| Precision | Bytes / param |
|-----------|---------------|
| fp32      | 4             |
| bf16/fp16 | 2             |
| int8      | 1             |
| nf4 (QLoRA) | ~0.5        |

So a 7B model:

- fp32: 28 GB &nbsp;·&nbsp; bf16: **14 GB** &nbsp;·&nbsp; int8: 7 GB &nbsp;·&nbsp; nf4: **~3.5 GB**

### Rule 2 — optimizer states (full FT only)

AdamW keeps **2 momentum tensors per trainable param**, typically fp32:

```
optimizer_state ≈ 8 bytes × trainable_params
```

For full FT of a 7B in bf16 weights + fp32 Adam state:
weights 14 GB + grads 14 GB + Adam state 56 GB ≈ **84 GB just to stand
still** before activations. That's why nobody full-FTs 7B on one GPU.

**LoRA changes this completely:** only the adapter params (often 0.1–1%
of total) are trainable, so Adam state is tiny.

### Rule 3 — activations (the part everyone underestimates)

Activations scale with `batch × seq_len × hidden × layers`. Two levers
collapse them:

- **Gradient checkpointing** — recomputes activations on the backward
  pass. ~2× memory reduction, ~20–30% slower.
- **FlashAttention-2/3** — avoids materializing the full attention
  matrix. Huge win at long context.

### A working budget for 24 GB (e.g. RTX 4090, L4, A10G)

| Setup                                  | 7B   | 13B  | 70B  |
|----------------------------------------|------|------|------|
| **QLoRA** (nf4 base + LoRA adapters)   | ✅   | ✅   | needs multi-GPU + FSDP |
| **LoRA** (bf16 base + LoRA adapters)   | ✅   | tight | ❌ |
| **Full FT** (bf16 + AdamW)             | ❌   | ❌   | ❌ |

QLoRA on a 24 GB card with seq_len 2048, batch 1, grad_accum 16,
checkpointing on is the **default starting point** for the rest of this
course.

---

## 3. Minimal code — install, then verify

### 3a. Install

Use a fresh virtualenv or conda env. Do not install into system Python.

```bash
# system: NVIDIA driver ≥ 550, CUDA 12.4 toolkit available (or use torch's bundled cuDNN)
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# torch first, matched to CUDA 12.4
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# core stack
pip install \
    "transformers>=4.45" \
    "trl>=0.12" \
    "peft>=0.13" \
    "accelerate>=1.0" \
    "datasets>=3.0" \
    "bitsandbytes>=0.44" \
    "wandb" \
    "sentencepiece" "protobuf"

# flash-attn: build wheel must match (torch, CUDA, python). Use prebuilt.
pip install flash-attn==2.6.3 --no-build-isolation
```

If `flash-attn` fails to install, do **not** spend an hour on it now —
skip it and pass `attn_implementation="sdpa"` to your model loader.
You'll lose some speed, not correctness.

### 3b. Verify — one script, four checks

Save as `verify_env.py`:

```python
import torch, importlib, sys

def v(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "n/a")
    except Exception as e:
        return f"MISSING ({e.__class__.__name__})"

print("python              ", sys.version.split()[0])
print("torch               ", torch.__version__)
print("torch.cuda.is_available", torch.cuda.is_available())
print("torch CUDA build    ", torch.version.cuda)
for name in ["transformers", "trl", "peft", "accelerate",
             "datasets", "bitsandbytes", "flash_attn", "wandb"]:
    print(f"{name:20s}", v(name))

assert torch.cuda.is_available(), "No CUDA device visible"
print("\nGPUs:")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/1e9:.1f} GB  "
          f"sm_{p.major}{p.minor}")

# bf16 sanity
x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("\nbf16 matmul OK, peak alloc:",
      torch.cuda.max_memory_allocated() / 1e6, "MB")
```

Run it:

```bash
python verify_env.py
```

Expected: every library prints a version (not `MISSING`), CUDA is
available, and at least one GPU with `total_memory` matching what you
paid for.

### 3c. Smoke test — load a model in 4-bit

This is the smallest possible "the QLoRA path actually works" test.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # tiny; swap for 7B once this works

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb,
    device_map="auto",
    attn_implementation="sdpa",  # swap to "flash_attention_2" if flash-attn installed
)

inputs = tok("The capital of France is", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(tok.decode(out[0]))
print("peak VRAM:", torch.cuda.max_memory_allocated() / 1e9, "GB")
```

If this runs and prints something coherent, your QLoRA training stack
is wired up correctly. You are ready for lesson 01.

### 3d. Configure `accelerate` once

```bash
accelerate config
```

Single GPU? Pick: no distributed, bf16, no DeepSpeed. We'll revisit
this in lesson 11 for multi-GPU.

---

## 4. Try it yourself (30 min)

1. Stand up the environment above on whatever GPU you have. Note the
   versions `verify_env.py` prints — **screenshot or save them**, you'll
   want them when something breaks two weeks from now.
2. Do the VRAM math for your card and the model you want to fine-tune:
   - weights (pick precision)
   - + LoRA adapter params (~1% of total, trainable)
   - + Adam state on adapters only (8 bytes × trainable)
   - + activations budget (≈ a few GB at seq 2048, batch 1, checkpointing)
   - Total should leave ≥ 2 GB headroom.
3. Run the 1.5B smoke test. Then swap `MODEL` for a 7B (e.g.
   `Qwen/Qwen2.5-7B`) and see if it still fits. If OOM, that tells you
   exactly what knobs you'll need to turn in lesson 10.

---

## 5. Pitfalls

- **CUDA mismatch.** Installing `torch==2.5.1+cu124` on a host whose
  driver only supports CUDA 12.1 → silent fallback to CPU or a cryptic
  `CUDA error: no kernel image is available`. Check `nvidia-smi`
  driver row first.
- **Two CUDAs on PATH.** A system CUDA 11.8 and a conda CUDA 12.4
  fighting for `nvcc` will produce flash-attn builds that segfault.
  Prefer torch's bundled runtime; don't compile against system CUDA
  unless you have to.
- **Building flash-attn from source.** It takes 30+ min and frequently
  fails on the wrong wheel of `ninja` / `packaging`. Use a prebuilt
  wheel, or skip it and use SDPA. Flash-attn is an optimization, not a
  correctness requirement.
- **bitsandbytes "no GPU detected".** Almost always a CUDA-libs path
  problem. `python -m bitsandbytes` prints a diagnostic; read it.
- **Mixing pip and conda.** Pick one. Hybrid envs are how `libcudart`
  ends up appearing twice with different versions.
- **Forgetting `sentencepiece` / `protobuf`.** Many tokenizers (Llama,
  Mistral, Gemma) silently fail to load without them.
- **`device_map="auto"` on a multi-GPU box for a single-GPU run.** It
  will helpfully shard your tiny model across cards and confuse every
  downstream tool. Set `CUDA_VISIBLE_DEVICES=0` explicitly.
- **VRAM math forgetting activations.** Weights + optimizer fits → "I
  have headroom!" → first forward pass OOMs at seq 4096. Always
  budget for activations, and enable gradient checkpointing by default
  when in doubt.
- **No headroom for eval.** Eval often uses higher effective batch /
  longer sequences than training. Leaving 0 GB free means your training
  succeeds and your eval OOMs.

---

## 6. Further reading

- bitsandbytes README — diagnostics for the "where is libcudart"
  failure mode.
- HuggingFace *flash-attn install guide* — prebuilt wheel index by
  (torch, CUDA, python).
- `accelerate` docs — the YAML it writes is small; read it once so
  you understand what `accelerate launch` is doing.
- Next: [`01-what-sft-is.md`](./01-what-sft-is.md) — the loss, the
  masking, the one-paragraph mental model.

---

**TL;DR:** Pin the stack, do the VRAM math before you launch, and prove
the QLoRA path works on a 1.5B model before you reach for a 7B.
Everything in the rest of this course assumes `verify_env.py` and the
4-bit smoke test both pass.
