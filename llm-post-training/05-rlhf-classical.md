# 05 — Classical RLHF (PPO Era)

The original recipe that turned GPT-3 into ChatGPT, and the template every
later post-training method either extended or replaced.

## The InstructGPT pipeline

```
   Base LLM
      │
      ▼
   ┌──────┐
   │ SFT  │  Demonstrations from human labelers
   └──┬───┘
      ▼
   ┌──────┐
   │  RM  │  Preferences from human labelers
   └──┬───┘
      ▼
   ┌──────┐
   │ PPO  │  Optimize policy against RM, KL-constrained to SFT model
   └──┬───┘
      ▼
   Aligned LLM
```

Introduced in **InstructGPT (2022)** and made famous by ChatGPT.

## PPO — Proximal Policy Optimization

PPO is the RL algorithm. In LLM-land it juggles **four models** at once:

1. **Policy** (the one being trained) — generates responses
2. **Reference model** (frozen copy of SFT model) — anchors the KL penalty
3. **Reward model** (frozen) — scores responses
4. **Value model** (trained alongside) — estimates expected reward

Each rollout: generate a response, score it with the RM, compute advantage,
update the policy — while penalizing it from drifting too far (in KL
divergence) from the reference.

## The KL constraint

Without it, the policy chases the RM into nonsense ("reward hacking"). With
it, the policy stays close to the sensible SFT model and only nudges toward
higher-reward behavior.

Trade-off: too much KL → no learning. Too little → reward hacking.

## Why classical RLHF is painful

- **4 models in memory** — huge compute footprint
- **Unstable** — hyperparameter-sensitive, can collapse mid-run
- **Slow** — on-policy means constant generation
- **Hard to debug** — many moving parts
- **Reward over-optimization** — RM becomes the enemy after enough steps

These pains are exactly why DPO and its successors took over (file 06).

## Anthropic's HH variant

Anthropic's **HH-RLHF** (Helpful & Harmless) dataset and method showed
RLHF could **simultaneously** improve helpfulness and reduce harm — not a
zero-sum trade-off, given good data. This shaped how the whole field
thought about safety.

## Where classical RLHF still wins

- Frontier labs with massive compute still use PPO-family methods, often
  on top of DPO warmup
- When the reward signal is noisy / dense / online (e.g., agentic RL)
- When you genuinely need exploration (not just imitation of preferences)

## TRL support

| Feature                                      | Added in   |
|----------------------------------------------|------------|
| `PPOTrainer` (original)                      | v0.1 era   |
| Naive Pipeline Parallelism for big PPO       | v0.4.1     |
| "10× PPO" with ZeRO-3                        | v0.8.2     |
| Experimental `PPOv2Trainer`                  | v0.9.3     |
| PPOv2 renamed to `PPOTrainer` (old deprecated)| v0.12     |
| PPO Lite (reward scaling by batch std)       | v0.22      |
| `PPOTrainer` moved to experimental           | v0.26      |

## What to learn next

- InstructGPT paper (Ouyang et al., 2022)
- PPO paper (Schulman et al., 2017)
- Anthropic HH paper (Bai et al., 2022)
- "Secrets of RLHF" papers — practical pain points
