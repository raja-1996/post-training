# 14 — TRL Library: Evolution Timeline

[`huggingface/trl`](https://github.com/huggingface/trl) is the dominant
open-source library for post-training. Its release history is essentially
a **timeline of the field** — almost every major algorithm shows up here
within months of its paper.

This file collects what each TRL release added. Individual algorithm
details live in the topical files; here we look at the **library as an
artifact**.

> Dates omitted on purpose — GitHub's release page summarization was
> unreliable on timestamps. Versions and content are correct; check the
> upstream repo for exact dates.

---

## Algorithm-introduction order (the headline view)

The roughly **chronological order** in which each major post-training
method first appeared in TRL:

```
v0.x early    →  PPO
v0.4          →  PEFT, QLoRA, SFTTrainer, RewardTrainer
v0.5          →  DPO
v0.6          →  DDPO (diffusion RLHF)
v0.7          →  Text Environments (early agents)
v0.7.5        →  IPO, cDPO, NEFTune, SLiC
v0.8          →  KTO, ORPO, CPO, VLM-SFT
v0.9.3        →  RLOO, PPOv2, TR-DPO, IRPO, Robust DPO, PNCA, BCO
v0.9.6        →  SimPO, EXO, AlignProp
v0.10         →  Online DPO, APO, DPO for VLMs, Liger kernel in SFT
v0.11         →  GKD, XPO, NashMD
v0.12         →  WPO, Seq-Level KD, pairwise judges
v0.13         →  PRMTrainer, DiscoPOP, AllTrueJudge
v0.14         →  GRPO  ← the GRPO era begins
v0.16         →  GRPO multi-node (70B+)
v0.17         →  Dr. GRPO
v0.18–0.19    →  Tools in SFT, Liger-DPO, FFD packing
v0.20         →  GSPO, MPO, FSDP2, GPT-OSS support
v0.22         →  DAPO, AlphaPO, native VLM SFT, BEMA
v0.23         →  Context Parallelism, DFT, TIS
v0.24         →  GSPO-token, GFPO, accuracy reward fn
v0.25–0.26    →  CISPO (ScaleRL), SAPO, GRPO agents, MiniLLM, GOLD, PAPO
v0.27         →  GDPO, async reward functions
v1.0          →  Async GRPO, VESPO, DPPO, SDPO (experimental)
v1.1          →  DistillationTrainer (on-policy KD)
v1.2          →  SSDTrainer
v1.3          →  TPO (Triple Preference Optimization)
v1.4          →  Chunked CE (-50% VRAM), length-normalized DPO sigmoid
```

---

## Major release table (1.x)

| Version | Headline additions                                                     |
|---------|------------------------------------------------------------------------|
| v1.4.0  | Chunked CE loss, OpenReward standard, length-norm DPO sigmoid, KTO↔DPO alignment, MFU helpers |
| v1.3.0  | Qwen 3.6 integration, **TPO** trainer (experimental), speculative decoding in `trl vllm-serve` |
| v1.2.0  | **SSDTrainer**, expanded tool-calling (LLaMA 3.1/3.2, DeepSeek-V3)     |
| v1.1.0  | **DistillationTrainer**, chunked LM head for AsyncGRPO, multimodal tool responses |
| v1.0.0  | **Async GRPO**, **VESPO** / **DPPO** / **SDPO** (experimental), `vllm_mode=colocate` default; v0→v1 migration guide |

## Major release table (0.x)

| Version | Headline additions                                                                                |
|---------|---------------------------------------------------------------------------------------------------|
| v0.27   | **GDPO**, async reward fns, ORPO catastrophic-cancellation fix                                    |
| v0.26   | **GRPO agent training with tools**, **CISPO**, **SAPO**, reasoning reward fn, MiniLLM/GOLD/PAPO   |
| v0.25   | vLLM sleep mode L2, Trackio logging, **GFPO** loss, OpenEnv integration, Python 3.9 dropped       |
| v0.24   | Accuracy reward fn, **GSPO-token**, BEMA for ref models, GRPO replay buffer                       |
| v0.23   | **Context Parallelism**, **DFT** loss, **TIS**, MoE aux loss, vLLM sleep in colocated GRPO/RLOO   |
| v0.22   | **Native VLM SFT**, **DAPO** loss, **AlphaPO** (via CPO), PPO Lite, BEMA callback                 |
| v0.21   | **GPT-OSS / Harmony** support, vLLM transformers backend                                          |
| v0.20   | **GSPO**, **MPO**, FSDP2, continuous batching                                                     |
| v0.19   | Tools in SFT (JSON schema), **FFD packing**, Liger-DPO loss, `clone_chat_template`                |
| v0.18   | PEFT + Liger GRPO, FSDP + Liger, co-located vLLM, two-sided GRPO clipping                         |
| v0.17   | **Dr. GRPO** loss variants, data-parallel vLLM (4× gen), V1 engine                                |
| v0.16   | **GRPO scaling to 70B+** via multi-node vLLM + NCCL, padding-free SFT, 6× faster multi-step       |
| v0.15   | GRPO + vLLM prefix caching, decoupled loss/gen, torch.compile, iterative GRPO, ZeRO-3             |
| v0.14   | **GRPOTrainer**, **PRMTrainer**, LLaVA-Next in DPO                                                |
| v0.13   | **PRMTrainer**, MergeModelCallback, **DiscoPOP** loss, **AllTrueJudge**                           |
| v0.12   | PPOv2 → PPO (old PPO deprecated), **WPO**, pairwise judges, Seq-Level KD                          |
| v0.11   | **GKDTrainer**, **XPOTrainer**, **NashMDTrainer**, Online DPO + LoRA                              |
| v0.10   | **Online DPO**, **APO** (apo_zero / apo_down), DPO for VLMs, Liger Triton in SFT, WinRate judge   |
| v0.9.6  | **SimPO** (via CPO), **AlignProp**, **EXO**                                                       |
| v0.9.3  | **RLOOTrainer**, experimental **PPOv2**, **TR-DPO / IRPO / Robust DPO / PNCA**, **BCO** in KTO    |
| v0.8.2  | **ORPO**, **CPO**, VLM SFT (Llava), "10× PPO" with ZeRO-3                                         |
| v0.8.0  | **KTOTrainer**, CLI for SFT/DPO/chat, **FSDP+QLoRA**                                              |
| v0.7.5  | **IPO**, **cDPO**, **NEFTune**, **SLiC** hinge loss, unsloth integration, Ascend NPU              |
| v0.7.3  | **IterativeTrainer**, NEFTune via forward hooks                                                   |
| v0.7.0  | **Text Environments** — agent infra with Python/search/calculator tools                           |
| v0.6.0  | **DDPO** (Denoising Diffusion Policy Optimization), reward scaling/clipping                       |
| v0.5.0  | **DPOTrainer** introduced (no reward model needed)                                                |
| v0.4.5  | `DataCollatorForCompletionOnlyLM`, multi-adapter RL (MARL)                                        |
| v0.4.2  | **SFTTrainer** + **RewardTrainer** introduced, QLoRA / bitsandbytes integration                   |
| v0.4.1  | Naive Pipeline Parallelism for 20B+ on consumer GPUs                                              |
| v0.4.0  | Full **PEFT integration**                                                                         |
| v0.3.0  | `set_seed`, minibatching, bf16, max_grad_norm                                                     |
| v0.2.0  | General decoder support beyond GPT-2, encoder-decoder (T5), push_to_hub, Accelerate integration   |

---

## What the release pattern shows

A few observations the timeline makes obvious:

1. **TRL ships fast.** Major papers (DPO, GRPO, SimPO, GSPO, DAPO, R1-style)
   typically land within weeks-to-months of release.
2. **The cadence of new algorithms is brutal.** Roughly one new
   preference/RL method per minor release in 2024–2025.
3. **GRPO is a watershed.** Everything after v0.14 is increasingly oriented
   around GRPO and its variants — reasoning RL is *the* dominant theme of
   the modern stack.
4. **Distillation is climbing.** Multiple trainers added across v1.x
   (DistillationTrainer, SSDTrainer) reflect the field's shift toward
   teacher→student post-training.
5. **Tools and agents are now first-class.** From Text Environments (v0.7)
   → tools in SFT (v0.19) → GRPO agents (v0.26) → multimodal tool
   responses (v1.1).
6. **Infrastructure work is constant.** Liger, vLLM modes, FSDP2, context
   parallelism, packing strategies — performance compounds across releases.

---

## How to use this file

- **Looking for when an algorithm became practical?** Find it in the timeline.
- **Picking a TRL version to base work on?** v1.x is the modern API; the v0→v1
  migration guide lives in the repo as `MIGRATION.md`.
- **Comparing to other frameworks?** TRL's coverage roughly defines the
  open-source state of the art — if it's not in TRL yet, expect it soon
  or look at OpenRLHF / verl.

## What to learn next

- TRL repository: `huggingface/trl`
- The `MIGRATION.md` (v0 → v1) document
- TRL examples directory for end-to-end recipes
- OpenRLHF and verl for comparison
