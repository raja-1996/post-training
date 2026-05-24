# 04 — Reward Modeling

A reward model (RM) turns the fuzzy idea of "this response is better" into a
**scalar score** that an RL algorithm can optimize.

## Why we need it

We can't ask a human to score every one of millions of training rollouts.
So we:
1. Collect a smaller dataset of human preferences once
2. Train an RM on it
3. Use the RM as a cheap stand-in for human judgment during RL

## Preference data

For a prompt `x`, humans see two responses `y_a` and `y_b` and pick one.
This gives `(x, y_chosen, y_rejected)` triples.

## Bradley-Terry model

The standard assumption: each response has an underlying scalar reward, and
the probability a human prefers `y_a` over `y_b` is a logistic function of
the reward difference.

The RM is trained to make `reward(chosen) > reward(rejected)` by a margin.

## RM architectures

- **Scalar-head RM** — base LLM with a linear head outputting one number
- **Pairwise classifier** — directly outputs "A is better" probability
- **Multi-objective RM** — separate heads for helpfulness, harmlessness, honesty, etc.
- **Ensemble RMs** — average several RMs to reduce reward hacking

## Outcome (ORM) vs Process (PRM) reward models

- **ORM** — scores the **final answer only**. Cheap, widely used.
- **PRM** — scores **each reasoning step**. Better for math/code, expensive
  to label. Datasets like PRM800K (OpenAI) and Math-Shepherd.

PRMs power much of the modern reasoning-RL work (see file 07).

## Reward hacking (Goodhart's law)

> "When a measure becomes a target, it ceases to be a good measure."

The model finds adversarial patterns that score high under the RM but are
bad to a human. Examples:
- Excessively long answers (length bias)
- Confident but wrong claims
- Sycophancy ("Great question!")
- Format tricks the RM rewards

**This is the central failure mode** of all RL post-training. Mitigations:
- Stronger / bigger RM
- KL constraint to a reference policy (don't drift too far)
- Continual RM retraining on new failure modes
- Verifiable rewards instead of learned RMs (see file 07)

## Verifiable rewards: the new alternative

For domains like math (`is answer correct?`) and code (`do tests pass?`),
you can skip the RM entirely and use a **programmatic checker**. This is
the foundation of o1- and R1-style reasoning training and largely sidesteps
reward hacking — when applicable.

## TRL support

| Feature                                  | Added in     |
|------------------------------------------|--------------|
| `RewardTrainer`                          | v0.4.2       |
| WinRate LLM-as-judge callback            | v0.10        |
| Pairwise judges for online methods       | v0.12        |
| `AllTrueJudge` (mixture of judges)       | v0.13        |
| `PRMTrainer` (process reward model)      | v0.13 / v0.14|
| Async reward functions                   | v0.27        |
| Reasoning reward function (math eval)    | v0.26        |

## What to learn next

- InstructGPT paper (RM training section)
- Anthropic HH-RLHF dataset
- PRM800K paper
- Reward model evaluation: RewardBench
