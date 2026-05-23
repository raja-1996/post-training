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

### A unified view: trajectory provenance (Zhao et al., 2026)

A useful re-organizing lens from [arxiv 2604.07941](../papers/2604.07941-post-training-unified-view.md):
the most consequential distinction is **not the algebraic form of the loss** but
**where the training trajectories come from**.

- **Off-policy** — train on externally supplied trajectories (demonstrations,
  preference pairs, teacher outputs, replayed traces). SFT, DPO-family,
  off-policy distillation all live here.
- **On-policy** — train on rollouts sampled from the **current policy** with
  feedback computed on those fresh trajectories. RLHF, RLAIF, RLVR, online
  preference optimization, on-policy distillation, iterative self-improvement.

This is related to but **not** the same as offline/online RL: off-policy/on-policy
here is about whether trajectories come from outside the learner or from the
learner itself.

A second axis describes the **supervision interface** — token targets,
preferences, scalar rewards, verifier/process feedback, or teacher-guided
targets. The same interface (e.g. "preferences") can appear in either
provenance regime with very different roles (offline DPO vs. online DPO).
Distillation is **not** a third regime — it's a teacher-guided *interface* that
can run in either provenance mode.

A third axis describes the **functional role** of each stage:
- **Effective support expansion** — bringing previously unreachable behaviors
  into the model's effective output distribution
- **Policy reshaping** — improving ranking/choice **within** already-reachable
  behaviors (calibration, alignment, style)
- **Behavioral consolidation** — making induced behavior survive handoff
  (across stages, into smaller students, without scaffolds)

Modern frontier recipes combine multiple regimes precisely because no single
regime addresses all three roles. See [13-frontier.md](./13-frontier.md) for
how this plays out in current recipes.

## What the rest of these notes cover

- **Data** — where instruction and preference data come from
- **SFT** — imitation learning on demonstrations
- **Reward modeling** — turning preferences into a scalar signal
- **RLHF and successors** — optimizing against that signal
- **Reasoning RL** — the o1 / R1 paradigm shift
- **Agents, safety, distillation, evaluation, infra, frontier**
