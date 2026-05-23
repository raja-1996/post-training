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

## What to learn next

- DistilBERT and original Hinton distillation paper
- R1's distilled model release
- "On-policy distillation" recent papers
- Orca / Orca-2 (Microsoft) — instruction-tuning via distillation
