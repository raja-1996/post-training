# 07 — Reasoning Post-Training (the o1 / R1 era)

The most exciting frontier as of 2024–2025. Instead of training models to
**imitate** good answers, train them to **produce long chains of thought
that lead to verifiable correct answers**.

## The shift

Old paradigm:
> Better outputs by aligning to human preferences.

New paradigm:
> Better **thinking** by reinforcing chains of thought that reach correct
> answers — verified programmatically.

OpenAI's **o1** and DeepSeek's **R1** are the public landmarks.

## RLVR — Reinforcement Learning with Verifiable Rewards

For tasks where correctness can be **checked automatically**:
- Math — does the final answer match?
- Code — do unit tests pass?
- Logic puzzles — does the answer satisfy the constraints?

Use the verifier as the reward signal. No reward model, no human labels at
training time, no reward hacking (mostly — the verifier is the ground truth).

This is the cleanest reward signal we have, and it scales.

## Chain-of-thought as a training target

Earlier work that paved the way:
- **STaR** — train on the rationales that led to correct answers
- **Quiet-STaR** — reason silently at every token
- **Rationale distillation** — copy long CoT from a stronger teacher

These showed that long, explicit reasoning traces can be **learned**, not
just elicited via prompting.

## Process vs outcome supervision

- **Outcome reward** — final answer right/wrong
- **Process reward** — each reasoning step labeled good/bad (PRMs, file 04)

Process supervision is more sample-efficient on hard problems but expensive
to label. Recent work generates process labels automatically by checking
whether each step leads to eventual success.

## The DeepSeek-R1 recipe (high-level)

1. **Cold-start SFT** on a small set of high-quality long-CoT examples
2. **Large-scale RL** with verifiable rewards (math, code)
3. **Rejection sampling** — collect the model's best responses
4. **More SFT** on those, plus general data
5. **Another RL stage** for alignment and general helpfulness

The striking finding from R1-Zero: doing RL **directly on a base model**,
with no SFT, produces strong reasoning — and the model discovers behaviors
like reflection ("wait, let me reconsider") on its own.

## Emergent behaviors

After enough RL with verifiable rewards, models start to:
- **Reflect** — notice their own mistakes mid-reasoning
- **Backtrack** — abandon a wrong line of thought
- **Verify** — double-check their answer before committing
- **Explore** — try multiple approaches

These aren't programmed in — they emerge because they're useful for
reaching correct answers.

## Inference-time compute

Reasoning post-training changes the **inference compute curve**: spending
more tokens "thinking" reliably improves accuracy. The model learns to use
that budget productively.

This is a new axis of scaling, complementing parameter and data scaling.

## Open challenges

- Generalizing beyond verifiable domains (writing, judgment, taste)
- Cost — long CoT is expensive at both train and inference time
- Distillation — copying R1's behavior into smaller models works surprisingly well
- Reward hacking when verifiers are imperfect

## TRL support

The reasoning-RL stack landed in TRL roughly in the v0.14 → v1.x window:

| Feature                                     | Added in     |
|---------------------------------------------|--------------|
| `GRPOTrainer`                               | v0.14        |
| `PRMTrainer` (process reward model)         | v0.13 / v0.14|
| Reasoning reward function (math eval)       | v0.26        |
| **Dr. GRPO** loss variants                  | v0.17        |
| GRPO scaling to 70B+ (multi-node vLLM + NCCL)| v0.16       |
| **DAPO** loss                               | v0.22        |
| **GSPO** (sequence-level GRPO)              | v0.20        |
| **GSPO-token**                              | v0.24        |
| **GFPO** loss                               | v0.24 / v0.25|
| **CISPO** (ScaleRL)                         | v0.26        |
| **SAPO** (soft gating)                      | v0.26        |
| **GDPO** (group reward-decoupled norm)      | v0.27        |
| **TIS** (truncated importance sampling)     | v0.23        |
| **DFT** (dynamic fine-tuning loss)          | v0.23        |
| Async GRPO (decoupled gen / gradient)       | v1.0         |
| **VESPO**, **DPPO**, **SDPO** (experimental)| v1.0         |
| GRPO agent training with tools              | v0.26        |

## What to learn next

- DeepSeek-R1 paper
- OpenAI o1 system card and blog posts
- STaR and Quiet-STaR
- PRM800K and Math-Shepherd
- "Let's Verify Step by Step" (OpenAI)
