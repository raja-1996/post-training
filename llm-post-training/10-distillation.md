# 10 — Distillation

Taking a big, expensive, well-aligned model and **compressing its behavior**
into a smaller, cheaper one. Increasingly central to the post-training stack.

## Why distill

- **Cost** — small models are cheaper to serve
- **Latency** — small models are faster
- **Edge deployment** — phones, browsers, on-device
- **Capability transfer** — bring frontier behavior to open-weight smaller models

## Basic idea

```
   Teacher (large, aligned)
        │
        │  generates responses / probabilities
        ▼
   Student (small)
        │
        │  trained to imitate teacher
        ▼
   Compact, behaviorally similar model
```

## Variants

### Sequence-level KD (hard distillation)
The teacher generates `(prompt, response)` pairs. The student does SFT
on them. Simple, no special infra. This is what most "distilled" open
models actually do.

### Logit / probability distillation (soft distillation)
The student matches the teacher's **full output distribution**, not just
the sampled tokens. Richer signal but needs access to teacher logits and
matching tokenizers.

### On-policy distillation
The **student** generates a response. The teacher **scores or corrects**
it. Closes the train/test distribution gap.

A recent specific recipe (Thinking Machines, 2025) makes this concrete:

- Sample trajectories from the student
- Score **each token** by reverse KL to the teacher's distribution
- Train on per-token KL — a **dense** signal (O(N) bits per episode)
  vs RL's sparse final reward (O(1) bits per episode)

Reported results on math reasoning:
- **7–10× faster** than RL to reach the same policy
- ~**50–100× cumulative compute savings**
- ~74% on AIME'24 at ~1,800 GPU-hours (RL needed ~18,000)
- **Multi-epoch training works** — RL would memorize the answer; on-policy
  distillation learns the full distribution
- Lets you **recover post-trained behaviors after domain SFT**, enabling
  continual learning without expensive re-RL

The intuition: RL is expensive because it searches the strategy space.
Once a good strategy exists in a teacher, distillation just transfers it.

### Reasoning distillation
Specifically: copy the teacher's **long chains of thought**. R1's released
distilled variants (R1-Distill-Llama, R1-Distill-Qwen) showed this works
remarkably well — small models inherit much of the reasoning behavior.

## Domain specialization via distillation

Distill a generalist teacher's behavior on a narrow domain into a small
specialist model:
- Code completion models
- Medical / legal assistants
- Customer-support bots
- Local document assistants

## Limits

- Student can't exceed the teacher (mostly)
- Risk of inheriting the teacher's failure modes wholesale
- Licensing — distilling closed-source model outputs is legally murky
- Tokenizer / template mismatches cause silent quality loss

## Distillation and the open-source ecosystem

Much of the open-weight model quality jump in 2024–2025 came from
distillation of frontier closed models (directly or via permissively
licensed teachers like Llama-3 and DeepSeek). Strong base models +
high-quality teacher outputs = competitive open models.

## TRL support

| Trainer / Feature                              | Added in   |
|------------------------------------------------|------------|
| `GKDTrainer` (Generalized KD)                  | v0.11      |
| Sequence-Level KD in GKD                       | v0.12      |
| GKD/GOLD buffer + vLLM inference               | v1.0       |
| `MiniLLM` trainer                              | v0.26      |
| `GOLD` trainer                                 | v0.26      |
| `DistillationTrainer` (on-policy KD)           | v1.1       |
| `SSDTrainer` (Simple Self-Distillation, exp.)  | v1.2       |

## What to learn next

- DistilBERT and original Hinton distillation paper
- R1's distilled model release
- "On-policy distillation" recent papers
- Orca / Orca-2 (Microsoft) — instruction-tuning via distillation
