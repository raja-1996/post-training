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
| 01 | [foundations](./01-foundations.md)                   | Why post-training exists; taxonomy     |
| 02 | [data](./02-data.md)                                 | Instruction data, synthetic data       |
| 03 | [sft](./03-sft.md)                                   | Supervised fine-tuning                 |
| 04 | [reward-modeling](./04-reward-modeling.md)           | Learning human preferences             |
| 05 | [rlhf-classical](./05-rlhf-classical.md)             | PPO-based RLHF (InstructGPT era)       |
| 06 | [preference-optimization](./06-preference-optimization.md) | DPO, KTO, ORPO, GRPO, SimPO      |
| 07 | [reasoning-post-training](./07-reasoning-post-training.md) | o1 / R1-style RL with verifiable rewards |
| 08 | [agentic-tool-use](./08-agentic-tool-use.md)         | Function calling, agent training       |
| 09 | [safety-alignment](./09-safety-alignment.md)         | Red-teaming, Constitutional AI, RLAIF  |
| 10 | [distillation](./10-distillation.md)                 | Compressing big models into small ones |
| 11 | [evaluation](./11-evaluation.md)                     | Benchmarks and judging                 |
| 12 | [infrastructure](./12-infrastructure.md)             | Frameworks and compute                 |
| 13 | [frontier](./13-frontier.md)                         | Open problems                          |

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
