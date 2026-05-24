# 13 — Frontier & Open Problems

Where post-training is going next, and what nobody has fully solved yet.

## 1. Self-improvement loops

Can a model **bootstrap past its teacher**?

- Generate problems and solutions, verify, train on the wins
- Iterative rounds of self-play
- Risks: drift into model-specific quirks, "model collapse" on its own outputs

Early signs are promising for verifiable domains (math, code). Open
elsewhere.

## 2. Long-horizon agentic RL

Real-world tasks span dozens of turns and tool calls. Open problems:

- **Credit assignment** over long trajectories
- **Sparse rewards** — task succeeds or fails only at the end
- **Sandboxing** — running real tools at training scale
- **Cost** — agent rollouts are slow and expensive

Probably the single hottest area in post-training right now.

## 3. Multimodal post-training

Vision, audio, video, and action models all need alignment too. Today
mostly handled by:
- Adapting text post-training recipes
- Vision-specific preference data
- Multimodal RLHF / DPO variants

Reasoning RL is starting to extend to multimodal — promising but immature.

## 4. Personalization without forgetting

Users want models tuned to **them** — voice, preferences, context — without
losing general capability or safety. Active areas:
- Memory systems vs in-weights personalization
- Per-user LoRAs
- Continual learning without catastrophic forgetting
- Privacy-preserving personalization

## 5. Scalable oversight

As models get smarter than their human (or AI) overseers:
- How do we still produce useful training signal?
- How do we trust the model isn't deceiving us during training?
- Debate, recursive reward modeling, weak-to-strong generalization

This is the long-term alignment problem.

## 6. Reward modeling at the frontier

Reward hacking gets worse as policies get smarter. Open questions:
- Can RMs scale alongside policies?
- Are verifiable rewards enough for the open-ended tasks we care about?
- Process supervision at scale
- Hybrid human + AI + verifier reward stacks

## 7. Inference-time compute as a first-class axis

Reasoning post-training tied training and inference together: the model is
**trained to use** an inference-time thinking budget. New questions:
- How to allocate compute adaptively per query?
- How to train for variable thinking budgets?
- How to compress long CoT into short reasoning?

## 8. Open vs closed gap

Frontier closed models still lead, but the gap on most tasks has narrowed
to months. The drivers:
- Open distillation of closed models
- Open-source recipes catching up (Tulu, R1, Qwen, Llama)
- Verifiable rewards democratizing reasoning RL
- Compute still favoring frontier labs

### Notable 2025–2026 frontier recipes

| Model / system            | Distinctive post-training move                                         |
|---------------------------|------------------------------------------------------------------------|
| **DeepSeek-R1**           | SFT cold-start → GRPO RLVR → rejection-sampled SFT → alignment RL → distill |
| **Llama-3 Herd**          | SFT + DPO + reward modeling stages, extensive data curation            |
| **Tulu-3**                | Open recipe: SFT + DPO + RLVR; reproducible end-to-end                 |
| **Kimi K2**               | Agentic intelligence post-training, long-horizon tool use              |
| **DeepSeek-V3.2**         | Multi-stage pipeline; further RLVR + agentic refinement                |
| **GLM-5**                 | "Vibe coding" → "agentic engineering"; coding-agent post-training      |
| **MiMo-V2-Flash**         | Efficient post-training pipeline for small reasoning models            |
| **Nemotron-Cascade 2**    | Cascade RL + multi-domain on-policy distillation                       |
| **AgentArk**              | Distill multi-agent intelligence into a single-LLM agent               |

The shared structural pattern (see [arxiv 2604.07941](../papers/2604.07941-post-training-unified-view.md)):
**expansion stage** (SFT or specialist synthesis) → **reshaping stage**
(RL / preference optimization / verifier-guided correction) → **consolidation
stage** (distillation, merging, or replay). The interesting differences are
*which interface* each stage uses and *which behavior must survive* the
handoff.

## 9. What "alignment" even means at AGI-adjacent capability

The current alignment toolkit was built for chatbots. As models become
agents that act in the world over long horizons:
- "Following instructions" stops being enough
- We need values, goals, and corrigibility
- Interpretability becomes load-bearing

## Where to keep watching

- DeepSeek, Qwen, AI2 (Tulu), Meta (Llama), Mistral — open recipe releases
- Anthropic, OpenAI, GDM — papers on alignment, scalable oversight, interpretability
- arXiv categories: cs.CL, cs.LG, cs.AI — daily firehose
- Workshops: NeurIPS / ICML alignment workshops, RLHF / RL-LLM workshops

## A closing thought

Post-training in 2026 looks nothing like RLHF circa 2022. Expect the
landscape to keep shifting fast — the algorithms, the data sources, the
evaluation methods, even the *concept* of what we're training for. These
notes are a snapshot, not a textbook.
