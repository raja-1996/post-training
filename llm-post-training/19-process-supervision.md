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
  See **Process Supervision-Guided Policy Optimization for Code Generation**
  (Dai et al., 2024).
- **Agents** — per-action labels: did this tool call move toward the goal?
  **AgentPRM** (Xi et al., 2025) trains process reward models for LLM agents.
- **SQL** — **Reward-SQL** (Zhang et al., 2025) — stepwise reasoning with
  process-supervised rewards for text-to-SQL
- **Writing** — paragraph-level quality scores
- **Reasoning over long documents** — per-claim citation correctness

Each domain needs its own definition of "step" and its own labeling pipeline.

## PRMs as credit assignment

A reframing argued in [arxiv 2604.09459](../papers/2604.09459-credit-assignment-rl-llms.md):
**PRMs are not a separate "reward modelling" technique — they ARE a
credit-assignment mechanism.** A PRM scoring each step $i$ with $r_i$ is
performing step-level decomposition of the terminal reward $R(\tau)$.
The PRM literature (Math-Shepherd, OmegaPRM) and the CA literature
(VinePPO, SCAR) are two views of the same problem.

Two consequences:

1. PRM design is fundamentally about *credit*, so the same trade-offs
   apply — finer granularity vs. computational cost, forward vs. hindsight
   estimation, presence/absence of an auxiliary model.
2. PRM-style supervision generalizes naturally beyond math: turn-level
   PRMs for agents (AgentPRM, SWEET-RL), info-theoretic credit (IGPO),
   counterfactual credit (HCAPO, C3, CCPO), and critic-free group
   comparison (GiGPO, CARL).

## Beyond static PRMs

Newer process-supervision work moves past one-shot PRM training:

- **Rewarding Progress** (Setlur et al., 2025) — scaling automated process
  verifiers; trains verifiers to predict eventual success probability rather
  than per-step "good/bad"
- **Critique-GRPO** (Zhang et al., 2025) — combines natural-language critique
  with numerical reward as step-level feedback
- **PURE** (Cheng et al., ICML 2025) — replaces sum-form PRM credit with
  **min-form** $V(s_t)=\mathbb{E}[\min_{t'\geq t} r_{t'}]$ to prevent
  reward hacking via "safe" intermediate steps
- **SPRO** (Fei et al., 2025) — masked step advantage as self-supervised
  step credit; reports 3.4× training efficiency over GRPO
- **PRL** (Yao et al., 2026) — derives the optimal process reward as the
  advantage under entropy-regularized RL — first theoretical (not just
  heuristic) PRM grounding
- **InT** (Yang et al., 2026) — self-proposed counterfactual interventions
  per step; high credit when intervention flips the outcome
- **CAPO** (Xie et al., 2025) — LLM-as-GenPRM: same LLM critiques each step
  in natural language, converted to scalar reward
- **HICRA** (Wang et al., 2025) — focuses credit on **planning tokens**
  rather than uniformly; identifies a two-phase learning dynamic (procedural
  skills → strategic planning)
- **Search/Verify/Feedback** survey (Guan et al., [arxiv 2411.11504](https://arxiv.org/abs/2411.11504))
- **PRM survey** (Zheng et al., [arxiv 2510.08049](https://arxiv.org/abs/2510.08049))

The unified-view survey [arxiv 2604.07941 §5.3](../papers/2604.07941-post-training-unified-view.md)
frames process supervision as "**fine-grained learner-state reshaping**" —
denser credit assignment than terminal reward, but still subject to the same
verifier-mismatch and proxy-overoptimization failure modes.

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
