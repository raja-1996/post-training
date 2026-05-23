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

## Switchable reasoning ("reasoning on/off" + thinking budget)

A different paradigm from o1/R1's always-reasoning behavior: train the
model so reasoning is **optional and controllable**. Used by NVIDIA's
Llama-Nemotron and Nemotron Nano 2.

- **Reasoning On/Off** — a system prompt (e.g. `/no_think`) flips the
  model between chain-of-thought and direct-answer modes. Both modes are
  trained explicitly (mixed reasoning-on / reasoning-off SFT data).
- **Thinking budget** — client injects `</think>` to cap reasoning tokens.
  Up to ~60% inference-cost reduction with minimal accuracy loss.
- Useful when reasoning is expensive but only sometimes needed (agent
  pipelines that mix easy and hard turns).

## Progressive context-length RL

Train GRPO at **growing context lengths** (e.g. 8K → 16K → 24K) rather
than starting at full length. Reported by NVIDIA's NeMo-RL team
reproducing the DeepScaleR recipe (Qwen-1.5B beat o1 on AIME 2024).
Stabilizes early training and saves compute before scaling sequences out.

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

## Capability boundary: expansion vs elicitation

A subtle but important 2025–2026 debate: when RLVR improves a model's
reasoning, is it **expanding** the set of behaviors the model can reach
(genuine new capability), or only **reshaping** the model's choice within
already-latent capability (better elicitation)?

- **"The Invisible Leash: Why RLVR May or May Not Escape Its Origin"**
  (Wu et al., 2025) — uses pass@k at large k as a proxy. If the base model
  can already produce the right answer at large k but RLVR shifts probability
  toward it at low k, that's elicitation, not expansion.
- **"Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?"**
  (Yue et al., 2025) — similar argument: RL frequently sharpens existing
  ability rather than introducing new strategies.

This is one of the most consequential open empirical questions for reasoning
RL. The unified-view survey [arxiv 2604.07941](../papers/2604.07941-post-training-unified-view.md)
treats this as the central diagnostic gap: distinguishing genuine support
expansion from improved selection within existing effective support.

## Guided on-policy learning (rollout-time scaffolds)

A bridge between RLVR and SFT-style guidance: train with on-policy rollouts
but inject hints, templates, or rubrics that nudge the rollout distribution
toward useful regions:

- **Self-Hinting LMs** (Liao et al., 2026)
- **ExPO** (Zhou et al., 2025) — self-explanation-guided RL
- **TemplateRL** (Wu et al., 2025) — structured template-guided RL
- **HiPO** (Deng et al., 2026) — self-hint policy optimization for RLVR
- **Rubric-Scaffolded RL** (Zhou et al., 2025)

The mechanism: by altering what the learner repeatedly encounters in its own
rollouts, you can selectively expand effective support beyond what unaided
RL would reach.

## Interleaved / unified SFT–RL

Rather than the rigid `SFT → RL` pipeline, recent work softens or removes the
boundary:

- **SRFT** (Fu et al., 2026) — single-stage Supervised + Reinforcement FT
- **UFT** (Liu/Farina/Ozdaglar, 2025) — Unifying Supervised and Reinforcement FT
- **Interleaved Online FT for Hardest Questions** (Ma et al., 2026)
- **Dynamic Weighting of SFT + RL** (Zhang et al., 2026)
- **RePO** (Li et al., 2025) — replay-enhanced policy optimization

Motivation: pure RL is unstable on tasks where the policy can't yet generate
correct rollouts; mixing in off-policy supervision on the same step
addresses that without the gradient-interference issues of a hard SFT→RL
handoff.

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
