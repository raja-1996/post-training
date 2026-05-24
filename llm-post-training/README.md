# LLM Post-Training — Overview

A high-level map of how a pre-trained language model becomes a useful, aligned, reasoning assistant.

> **Scope:** These notes are intentionally **high-level**. Each file gives the
> shape of the topic — definitions, why it matters, key methods, and what to
> dig into later. Deep math, code, and recipes will be added in a follow-up.

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                      THE LLM LIFECYCLE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Pre-training         Mid-training         Post-training        │
│  ───────────          ───────────          ─────────────        │
│  Trillions of         Cooldown,            Behavior shaping,    │
│  tokens, raw          long-context,        alignment,           │
│  text, next-          domain mix,          reasoning,           │
│  token loss           annealing            tool use             │
│                                                                 │
│  "Knows things"    →  "Refined base"    →  "Useful assistant"   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Post-training is the difference between a model that **completes text** and
one that **follows instructions, reasons, uses tools, and refuses harm**.

---

## Reading Order

| #  | File                                                 | Topic                                  |
|----|------------------------------------------------------|----------------------------------------|
| 01 | [foundations](./01-foundations.md)                         | Why post-training exists; taxonomy        |
| 02 | [mid-training](./02-mid-training.md)                       | Annealing, capability injection, recipes  |
| 03 | [data](./03-data.md)                                       | Instruction data, synthetic data          |
| 04 | [sft](./04-sft.md)                                         | Supervised fine-tuning                    |
| 05 | [reward-modeling](./05-reward-modeling.md)                 | Learning human preferences                |
| 06 | [rlhf-classical](./06-rlhf-classical.md)                   | PPO-based RLHF (InstructGPT era)          |
| 07 | [preference-optimization](./07-preference-optimization.md) | DPO, KTO, ORPO, GRPO, SimPO               |
| 08 | [process-supervision](./08-process-supervision.md)         | PRMs, step-level rewards                  |
| 09 | [reward-hacking](./09-reward-hacking.md)                   | Specification gaming and mitigations      |
| 10 | [reasoning-post-training](./10-reasoning-post-training.md) | o1 / R1-style RL with verifiable rewards  |
| 11 | [distillation](./11-distillation.md)                       | Compressing big models into small ones    |
| 12 | [long-context](./12-long-context.md)                       | Training to actually use 200K+ windows    |
| 13 | [agentic-tool-use](./13-agentic-tool-use.md)               | Function calling, agent training          |
| 14 | [agentic-coding](./14-agentic-coding.md)                   | Real-codebase agents, SWE-Bench recipes   |
| 15 | [computer-use](./15-computer-use.md)                       | Screenshot → action, UI grounding         |
| 16 | [safety-alignment](./16-safety-alignment.md)               | Red-teaming, Constitutional AI, RLAIF     |
| 17 | [evaluation](./17-evaluation.md)                           | Benchmarks and judging                    |
| 18 | [infrastructure](./18-infrastructure.md)                   | Frameworks and compute                    |
| 19 | [trl-library](./19-trl-library.md)                         | TRL release timeline & algorithm history  |
| 20 | [frontier](./20-frontier.md)                               | Open problems                             |

---

## The Standard Pipeline

```
   Base Model
       │
       ▼
   ┌────────┐
   │  SFT   │  Supervised fine-tuning on instruction data
   └────┬───┘
        ▼
   ┌────────┐
   │   RM   │  Train a reward model on human preferences
   └────┬───┘  (skipped in DPO-family methods)
        ▼
   ┌────────┐
   │  RLHF  │  Optimize policy against reward (PPO / DPO / GRPO)
   └────┬───┘
        ▼
   ┌────────┐
   │ Safety │  Refusal training, red-team patches, constitution
   └────┬───┘
        ▼
  Aligned Model
```

Modern frontier recipes (DeepSeek-R1, Llama-3 Instruct, Tulu-3, Qwen2.5)
all variations on this skeleton, with reasoning-focused RL layered on top.

---

## Paper deep-dives

Detailed summaries of individual surveys live in [`../papers/`](../papers/):

- [2604.07941 — LLM Post-Training: A Unified View of Off-Policy and
  On-Policy Learning](../papers/2604.07941-post-training-unified-view.md)
  (Zhao et al., Apr 2026)
