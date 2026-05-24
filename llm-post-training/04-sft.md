# 03 — Supervised Fine-Tuning (SFT)

SFT is **imitation learning**: show the model many `(prompt, response)`
pairs and train it to reproduce the responses. It's the first and often
most impactful post-training step.

## What SFT does

- Teaches **format** (chat turns, markdown, tool calls)
- Teaches **task patterns** (summarize, classify, code, etc.)
- Establishes a **baseline persona / tone**
- Unlocks instruction-following from the base model

## How SFT differs from pre-training

Both use next-token cross-entropy loss. The difference:

- **Pre-training** — predict every token in raw text
- **SFT** — only compute loss on the **assistant's** response tokens
  (the prompt is masked out)

This **loss masking** keeps the model from learning to generate user-style
text and focuses gradient on what we want it to produce.

## Chat templates

Each model family uses its own format to mark turns. Examples:

- **ChatML** — `<|im_start|>user ... <|im_end|>`
- **Llama-3** — `<|start_header_id|>user<|end_header_id|>`
- **Mistral** — `[INST] ... [/INST]`

Using the **wrong template** at fine-tune or inference time is a top-3
silent failure mode in practice.

## Full fine-tuning vs PEFT

| Approach           | Params trained | Compute     | Use when                  |
|--------------------|----------------|-------------|---------------------------|
| Full fine-tuning   | 100%           | Highest     | Frontier labs, big shifts |
| **LoRA**           | ~0.1–1%        | Low         | Most practitioners        |
| **QLoRA**          | ~0.1–1%        | Very low    | Single-GPU fine-tuning    |
| **DoRA, rsLoRA**   | Variants       | Low         | Marginal gains over LoRA  |

**LoRA** trains small "low-rank" adapter matrices alongside frozen weights.
**QLoRA** adds 4-bit quantization of the base model on top.

### LoRA practical recipe (Schulman et al., 2025)

A systematic empirical study ("LoRA Without Regret") found that LoRA
matches full fine-tuning when set up correctly. Key takeaways:

- **Learning rate ~10× higher** than full FT
- **Apply LoRA to all layers** — especially **MLPs / MoE layers**.
  Attention-only LoRA underperforms even at matched parameter count
- **Large batch sizes hurt LoRA more** than full FT (independent of rank)
- For **RL post-training**, **rank-1 is often enough** — policy-gradient
  methods carry only ~1 bit of info per episode, so RL is inherently
  low-capacity
- LoRA uses ~**2/3 the FLOPs** of full FT per pass
- The 1/r scaling in `W' = W + (α/r)BA` makes optimal LR ~rank-independent
  early in training

## Practical SFT engineering

- **Sequence packing** — concatenate short examples to fill context, big throughput win
- **Gradient checkpointing** — trade compute for memory
- **Mixed precision (bf16)** — standard
- **FSDP / ZeRO** — shard model across GPUs
- **Learning rate** — much lower than pre-training (1e-5 to 5e-5 typical)
- **Epochs** — usually 1–3; more risks overfitting and memorization

## Catastrophic forgetting

Fine-tuning too aggressively can wipe out pre-trained knowledge.
Mitigations:
- Low learning rate
- Few epochs
- Mix in some pre-training data ("replay")
- Use LoRA (smaller capacity change)
- KL regularization to the base model

## Common failure modes

- **Format collapse** — model outputs the same template every time
- **Mode collapse** — short, generic, hedging answers
- **Knowledge regression** — base-model facts get worse
- **Wrong-template inference** — silent quality degradation
- **Exposure mismatch** — teacher-forced training prefixes diverge from
  learner-rollout prefixes at deployment (the scheduled-sampling / DAgger
  problem). The structural blind spot of off-policy SFT, motivating
  on-policy follow-ups.

## Model-aware / distribution-aware SFT (2025–2026)

A growing line of work redesigns SFT so that supervision is shaped to the
**learner's current distribution** rather than treating SFT as vanilla
likelihood maximization. The survey [arxiv 2604.07941](../papers/2604.07941-post-training-unified-view.md)
calls this an emerging direction (Sec. 7.1.1).

- **PRISM** (Zhao et al., 2026) — disentangles SFT and RL data via gradient
  concentration, reducing interference between the two regimes
- **Proximal SFT** (Zhu et al., 2026) — constrains policy drift during SFT
- **Anchored SFT** (Zhu et al., 2026) — preserves anchoring to base-model
  behavior, reducing collateral capability loss
- **Probability-based SFT objectives** (Li et al., 2025) — moves beyond
  log-likelihood for capability-matched supervision
- **Self-Distillation Bridges Distribution Gap** (Yang et al., 2024) — uses
  the model's own outputs to shrink the policy gap before SFT
- **"Best Instruction-Tuning Data are Those That Fit"** (Zhang et al., 2025) —
  selecting responses that match the model's existing distribution
- **Mind the Gap / SFT data rewriting** (Zhao et al., 2025)
- **Selective Self-to-Supervised Fine-Tuning** (Gupta et al., 2025)
- **SFT generalization with reward rectification** (Wu et al., 2026) — RL
  perspective on SFT generalization

The shared theme: vanilla token-target SFT can both **expand** support (when
demonstrations expose unreachable behaviors) and **reshape** the policy (when
demonstrations only restyle reachable behaviors), and the two regimes need
different supervision designs.

## TRL support

| Feature                             | Added in        |
|-------------------------------------|-----------------|
| `SFTTrainer`                        | v0.4.2          |
| QLoRA / 4-bit + bitsandbytes        | v0.4.2          |
| FSDP + QLoRA                        | v0.8            |
| VLM SFT (Llava)                     | v0.8.2          |
| Native VLM SFT                      | v0.22           |
| Tools in SFT (JSON schema)          | v0.19           |
| Assistant-only training             | v0.19           |
| FFD packing                         | v0.19           |
| BFD packing                         | v1.0            |
| Padding-free SFT                    | v0.16           |
| Chunked CE loss (-50% VRAM)         | v1.4            |
| Liger kernel integration            | v0.18 / v0.22   |

## What to learn next

- Llama-3 / Mistral / Qwen instruct recipes
- LoRA and QLoRA papers
- Tulu-3 SFT mixture
- Loss masking implementation in TRL or Axolotl
