# 06 — The Preference-Optimization Revolution

Starting in 2023, a family of simpler algorithms largely **replaced PPO** in
practice. They turn preference data directly into a policy update — no
reward model, no value model, no on-policy rollouts.

## DPO — Direct Preference Optimization

The breakthrough (Rafailov et al., 2023). The key insight:

> The optimal policy under a KL-constrained reward objective has a
> **closed-form relationship** to the reward function. So you can train
> the policy directly on preference pairs and skip the RM entirely.

DPO requires:
- Pairs `(prompt, chosen, rejected)`
- A frozen reference model (the SFT model)
- One forward/backward pass per example

What you **don't** need: an RM, a value model, on-policy generation, or PPO.
This makes DPO **orders of magnitude cheaper** than RLHF.

## Why DPO took over

- Trains like SFT — stable, fast, no RL gymnastics
- Reuses existing preference datasets
- Open-source releases (Zephyr, Tulu-2, Llama-3 Instruct) used it
- Matches or beats PPO on most public benchmarks

## The DPO family

| Method   | Idea                                                    |
|----------|---------------------------------------------------------|
| **DPO**  | Closed-form policy from preference pairs                |
| **IPO**  | Fixes DPO's tendency to overfit on easy pairs           |
| **KTO**  | Uses single examples (good/bad), no pairs needed        |
| **ORPO** | Combines SFT + preference in one stage (no SFT pre-step)|
| **SimPO**| Length-normalized, drops the reference model            |
| **CPO**  | Combines DPO loss with an SFT-style term                |
| **sDPO** | Stepwise / staged DPO with multiple reference updates   |
| **R-DPO**| Robust variant against noisy preferences                |

Each fixes a specific DPO pain point.

## GRPO — Group Relative Policy Optimization

DeepSeek's PPO replacement, used in DeepSeek-Math and DeepSeek-R1.

Idea:
- For each prompt, sample a **group of responses**
- Score them all with a reward (RM or verifier)
- Compute advantages **relative to the group's mean reward**
- No value model needed (the group baseline replaces it)

This drops one of PPO's four models and is much more stable. GRPO is now
a popular choice for reasoning-RL on math and code.

## REINFORCE++, RLOO, ReMax

Even leaner on-policy methods that revisit older policy-gradient ideas with
LLM-scale engineering. Often competitive with PPO at a fraction of the cost.

## Choosing among them

Rough guide:

- **Have preference pairs, want simplicity** → DPO (or ORPO)
- **Have only thumbs-up/down signals** → KTO
- **Want to skip the SFT stage** → ORPO
- **Want to skip the reference model** → SimPO
- **Doing reasoning RL with verifiable rewards** → GRPO
- **Frontier lab with infinite compute** → still some PPO somewhere

## The bigger pattern

DPO-family methods exposed that **a lot of RLHF's complexity was
unnecessary** for the preference-alignment problem. RL is making a
comeback now — but for **reasoning** (file 07), where exploration
genuinely helps.

## TRL support

| Algorithm / Feature       | Added in        |
|---------------------------|-----------------|
| **DPO** (`DPOTrainer`)    | v0.5            |
| **IPO** (DPO loss type)   | v0.7.5          |
| **cDPO**                  | v0.7.5          |
| **KTO** (`KTOTrainer`)    | v0.8            |
| **ORPO** (`ORPOTrainer`)  | v0.8.2          |
| **CPO** (`CPOTrainer`)    | v0.8.2          |
| **RLOO** (`RLOOTrainer`)  | v0.9.3          |
| **BCO** loss              | v0.9.3          |
| **TR-DPO, IRPO, Robust DPO, PNCA** | v0.9.3 |
| **SimPO** (via CPO)       | v0.9.6          |
| **EXO**                   | v0.9.6          |
| **Online DPO**            | v0.10.1         |
| **APO** (apo_zero/down)   | v0.10.1         |
| **XPO** (`XPOTrainer`)    | v0.11           |
| **NashMD** (`NashMDTrainer`) | v0.11        |
| **WPO** (weighted PO)     | v0.12           |
| **DiscoPOP**              | v0.13           |
| **GRPO** (`GRPOTrainer`)  | v0.14           |
| **AlphaPO** (via CPO)     | v0.22           |
| **MPO** (mixed PO)        | v0.20           |
| **TPO** (triple PO, exp.) | v1.3            |
| Length-normalized DPO sigmoid | v1.4        |

## What to learn next

- DPO paper (Rafailov et al., 2023)
- KTO and ORPO papers
- DeepSeek-Math (GRPO introduction)
- SimPO paper
