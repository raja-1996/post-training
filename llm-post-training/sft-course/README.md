# SFT Course — Train SFT for Any Task

A hands-on curriculum to take you from zero → being able to design, run, debug,
evaluate, and ship a supervised fine-tune for an arbitrary task.

> **How to use this folder:** Each lesson is a standalone markdown file with
> concept → minimal code → "try it yourself" → pitfalls. Lessons are ordered;
> read them top-to-bottom on first pass. After that, the cheat-sheet and bugs
> appendices are the things you'll keep coming back to.

---

## Curriculum

### Part A0 — Pre-flight

| #    | Lesson                                                        | Goal                                                                 |
|------|---------------------------------------------------------------|----------------------------------------------------------------------|
| 00   | [Do you even need SFT?](./00-do-you-need-sft.md)              | Decision tree: prompting → few-shot → RAG → SFT → DPO/RL             |
| 00.5 | [Environment & hardware setup](./00-5-environment-setup.md)   | CUDA / torch / trl / peft / bnb / accelerate / flash-attn; VRAM math |

### Part A — Foundations

| #  | Lesson                                                        | Goal                                                                |
|----|---------------------------------------------------------------|---------------------------------------------------------------------|
| 01 | [What SFT actually is](./01-what-sft-is.md)                   | Next-token CE on responses; when SFT is the right tool              |
| 02 | [Loss masking](./02-loss-masking.md)                          | `labels = -100` on prompt tokens; multi-turn masking                |
| 03 | [Chat templates & tokenizer mechanics](./03-chat-templates.md) | ChatML/Llama-3/Mistral/Qwen/Gemma; special tokens; embedding resize |

### Part B — Decisions before you train

| #  | Lesson                                                        | Goal                                                                 |
|----|---------------------------------------------------------------|----------------------------------------------------------------------|
| 04 | [Choosing a base model](./04-choosing-a-base-model.md)        | Base vs instruct; size; license; context length; tokenizer quality   |
| 05 | [Full FT vs LoRA vs QLoRA vs DoRA](./05-full-ft-vs-lora.md)   | Memory math; LoRA targets="all-linear"; LR ×10; when each fits       |

### Part C — Data (where 80% of quality lives)

| #   | Lesson                                                        | Goal                                                                 |
|-----|---------------------------------------------------------------|----------------------------------------------------------------------|
| 06  | [Building an SFT dataset for *your* task](./06-building-dataset.md) | Sourcing, schema design, diversity, length, 1K-good > 100K-meh  |
| 6.5 | [Synthetic data generation](./06-5-synthetic-data.md)         | Distillation, self-instruct, Evol-Instruct, rejection sampling       |
| 6.7 | [Data quality filtering & dedup](./06-7-data-filtering.md)    | Perplexity / RM filtering; MinHash; PII; safety; Tulu-3 recipe       |
| 07  | [Data preprocessing pipeline](./07-preprocessing.md)          | Tokenization, truncation, packing (FFD/BFD), padding-free            |
| 7.5 | [Data mixing & curriculum](./07-5-data-mixing.md)             | Mixture weights, task balancing, replay, easy→hard curriculum        |

### Part D — Hands-on training

| #    | Lesson                                                        | Goal                                                                 |
|------|---------------------------------------------------------------|----------------------------------------------------------------------|
| 08   | [First SFT run with TRL `SFTTrainer`](./08-first-sft-run.md)  | ≤80-line script; `SFTConfig` flags; reading loss curves              |
|      | …with framework comparison sub-section: TRL vs Axolotl vs Unsloth vs LLaMA-Factory vs torchtune |                            |
| 09   | [LoRA / QLoRA with PEFT + TRL](./09-lora-qlora.md)            | `LoraConfig`, bnb 4-bit, save/merge/reload adapters                  |
| 10   | [Hyperparameter recipe](./10-hyperparameters.md)              | LR, batch, grad accum, epochs, schedule, bf16, grad checkpointing    |
| 11   | [Multi-GPU scaling](./11-multi-gpu.md)                        | DDP / FSDP / DeepSpeed ZeRO; FSDP + QLoRA; Accelerate config         |
|      | …with speed/memory: FlashAttention-2/3, Liger, chunked CE, padding-free |                                                          |
| 11.5 | [Experiment tracking & reproducibility](./11-5-tracking.md)   | W&B, seeds, deterministic flags, data+config+git versioning, cost    |

### Part E — Beyond the basics

| #  | Lesson                                                        | Goal                                                                 |
|----|---------------------------------------------------------------|----------------------------------------------------------------------|
| 12 | [Multi-turn, tool-use, structured-output SFT](./12-multi-turn-tools.md) | Assistant-only multi-turn; JSON-schema tools; long-CoT SFT  |
| 13 | [Catastrophic forgetting & mitigations](./13-forgetting.md)   | Replay, KL, low LR, Anchored / Proximal SFT                          |
| 14 | [Common failure modes & debugging](./14-failure-modes.md)     | Format/mode collapse, knowledge regression, exposure bias            |
| E1 | [Continual / iterative SFT](./E1-continual-sft.md)            | SFT on top of an already-SFT'd model without nuking it               |
| E2 | [Domain vs task adaptation](./E2-domain-vs-task.md)           | Broad+low-LR vs narrow+higher-LR recipes                             |
| E3 | [VLM / multimodal SFT](./E3-vlm-sft.md)                       | Image-text pairs, processor vs tokenizer, frozen vision encoder      |
| E4 | [Safety considerations during SFT](./E4-safety.md)            | Refusal shaping, jailbreak-resistant data, where SFT-only fails      |

### Part F — Validate & ship

| #    | Lesson                                                        | Goal                                                                 |
|------|---------------------------------------------------------------|----------------------------------------------------------------------|
| 15   | [Evaluating an SFT model](./15-evaluation.md)                 | Held-out loss vs behavioral eval; LLM-as-judge; capability regression|
| 15.5 | [Inference & serving the trained model](./15-5-serving.md)    | Adapters vs merged; sampling; vLLM / TGI / llama.cpp / Ollama        |
| 15.7 | [Iteration loop & error analysis](./15-7-iteration.md)        | Slice failures → data gaps → targeted regen → re-train               |
| 15.8 | [Shipping artifacts](./15-8-shipping.md)                      | HF Hub upload, model card, eval card, GGUF / AWQ / GPTQ export       |
| 16   | [Capstone: end-to-end SFT for a chosen task](./16-capstone.md)| LoRA capstone + small-model full-FT capstone                         |

### Appendices

| #  | Appendix                                                      | Goal                                                                 |
|----|---------------------------------------------------------------|----------------------------------------------------------------------|
| A1 | [Cheat sheet](./A1-cheat-sheet.md)                            | One-pager: LR / batch / epochs / LoRA targets for 1B / 7B / 13B / 70B|
| A2 | [Common bugs](./A2-common-bugs.md)                            | Unmasked labels, wrong template, stray BOS, LR×LoRA mismatch, etc.   |

---

## How to request a lesson

When you're ready to write one, just say e.g. *"write lesson 02"* or *"write
the loss-masking lesson"*. Each lesson will be a self-contained file under
this folder, with runnable code where applicable.

## Per-lesson shape

Every lesson aims for roughly:
- **~20 min reading + ~30 min hands-on**
- Sections: *Concept* → *Minimal code* → *Try it yourself* → *Pitfalls* → *Further reading*
- Framework spine: **TRL** (mentions of Axolotl / Unsloth / LLaMA-Factory / torchtune where relevant)

## Prerequisites

- Comfortable Python + PyTorch
- Read the top-level [`03-sft.md`](../03-sft.md) overview first
- Access to at least one GPU (16 GB VRAM unlocks QLoRA on 7B; more is better)
