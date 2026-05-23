# 19 — Process Supervision

Rewarding the **reasoning steps**, not just the final answer. The
foundation of modern reasoning RL when verifiable outcomes are too sparse
or too easy to game.

## Outcome vs process

- **Outcome reward** — final answer correct? +1. Wrong? 0.
- **Process reward** — each step labeled good / bad / neutral.

Outcome rewards are cheap but sparse. Process rewards are dense but
expensive to collect.

## Why process supervision matters

- **Credit assignment** — long chains of thought have many tokens; outcome
  rewards spread blame thinly
- **Partial credit** — a mostly-right derivation with one arithmetic slip
  is more useful signal than "wrong"
- **Reward hacking resistance** — graders that check steps catch shortcut
  reasoning that lucky-guesses the answer
- **Interpretability** — step labels reveal *where* the model fails

## Process Reward Models (PRMs)

A PRM scores each step in a chain of thought:

```
Step 1: Let x = 2.        → 1.0  (correct)
Step 2: Then x + 3 = 5.   → 1.0
Step 3: So 5 × 2 = 12.    → 0.0  (wrong)
Step 4: Therefore x = 12. → 0.0  (propagated error)
```

The PRM is trained on labeled step trajectories. At RL time, it provides
dense reward signal per step instead of one number at the end.

## Sources of step labels

- **Human annotation** — gold standard, very expensive. PRM800K is the
  canonical dataset (OpenAI, 800K step labels for math).
- **Automatic via rollouts** — from a given partial trace, sample many
  continuations. If most reach a correct answer, that step is "good"
  (Math-Shepherd, OmegaPRM).
- **Verifier-based** — formal verifier checks intermediate state
  (math systems, theorem provers).
- **Strong-model judging** — ask a stronger model to grade each step.

Automatic methods scale; human methods are cleaner. Hybrids are common.

## "Let's Verify Step by Step"

The OpenAI paper that put process supervision on the map. Headline
findings:
- PRMs outperform outcome reward models on hard math
- The gap grows with problem difficulty
- PRMs generalize across problem types

This work directly informed the o1 line of reasoning models.

## How PRMs are used

Three modes:

1. **Best-of-N selection** — sample N completions, score each with the
   PRM, pick the highest-scoring one. Cheap, no RL needed.
2. **Search-time guidance** — beam / tree search over reasoning steps
   guided by PRM scores (Tree-of-Thoughts style).
3. **RL reward signal** — PRM scores feed directly into PPO / GRPO as
   per-step or aggregated reward.

## Limitations

- **Label noise** — automatic step labels are noisy, especially for
  long traces
- **Step granularity** — what is a "step"? Tokens, sentences, equations?
  Choice matters and isn't standardized
- **Spurious correlations** — PRMs learn surface features (presence of
  certain phrases) as proxies for correctness
- **Reward hacking** — once you RL against a PRM, it gets gamed too
  (see file 18)
- **Domain narrowness** — works best where steps are crisp (math, code);
  fuzzier for open-ended reasoning

## Outcome supervision strikes back

DeepSeek-R1 showed that **pure outcome RL** with verifiable rewards
produces strong reasoning models without an explicit PRM. The tradeoff:
- Outcome RL needs many more samples
- But it sidesteps PRM hacking and labeling cost
- It also produces more diverse reasoning behaviors

The field has not settled. Both paradigms coexist; many recipes blend
them.

## Process supervision beyond math

- **Code** — per-step labels: did this edit improve test pass-rate?
- **Agents** — per-action labels: did this tool call move toward the goal?
- **Writing** — paragraph-level quality scores
- **Reasoning over long documents** — per-claim citation correctness

Each domain needs its own definition of "step" and its own labeling pipeline.

## TRL support

- **`PRMTrainer`** — trains process reward models (v0.13 / v0.14)
- Reasoning reward functions for math grading (v0.26)
- GRPO / PPO can consume PRM outputs as rewards

## What to learn next

- "Let's Verify Step by Step" (Lightman et al., OpenAI)
- PRM800K dataset
- Math-Shepherd (automatic step labeling)
- OmegaPRM (Google)
- DeepSeek-R1 paper (counterpoint — outcome RL alone)
- "Solving math word problems with process- and outcome-based feedback"
  (DeepMind)
