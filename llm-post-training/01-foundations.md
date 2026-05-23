# 01 — Foundations

## Why post-training exists

A base model trained only on raw text is a **next-token predictor**, not an
assistant. Ask it "What is 2+2?" and it might continue with another math
question instead of answering, because that's what its training distribution
(the internet) often looks like.

Post-training is the bridge between **"knows things"** and **"is useful"**.

## The base-model gap

What a base model **has**:
- Broad world knowledge
- Strong language modeling
- Latent reasoning ability
- Latent tool-use ability

What a base model **lacks**:
- Instruction following
- Conversational format
- Refusal of harmful requests
- Calibrated honesty / "I don't know"
- Reliable chain-of-thought
- Tool-use / function-calling format
- A consistent persona

Post-training elicits and shapes these.

## The "alignment tax"

Aligning a model often costs raw capability on some benchmarks
(e.g., perplexity, raw recall). The goal is to **minimize the tax** while
maximizing usefulness — and modern recipes have largely closed this gap.

## Pre-training vs Mid-training vs Post-training

| Phase         | Data                         | Loss            | Goal                       |
|---------------|------------------------------|-----------------|----------------------------|
| Pre-training  | Web, books, code (trillions) | Next-token      | Build a world model        |
| Mid-training  | Curated, long-context, math  | Next-token      | Sharpen the base           |
| Post-training | Instructions, preferences    | Mixed (SFT, RL) | Shape behavior & alignment |

The line between mid- and post-training is fuzzy and shifting.

## Taxonomy of post-training techniques

```
                  POST-TRAINING
                       │
        ┌──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
       SFT       Preference Opt.     RL          Distillation
   (imitation)   (DPO, KTO, ORPO)  (PPO, GRPO)  (teacher→student)
        │              │              │              │
        └──── often combined into multi-stage pipelines ────┘
```

## What the rest of these notes cover

- **Data** — where instruction and preference data come from
- **SFT** — imitation learning on demonstrations
- **Reward modeling** — turning preferences into a scalar signal
- **RLHF and successors** — optimizing against that signal
- **Reasoning RL** — the o1 / R1 paradigm shift
- **Agents, safety, distillation, evaluation, infra, frontier**
