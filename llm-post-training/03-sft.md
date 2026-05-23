# 03 — Supervised Fine-Tuning (SFT)

SFT is **imitation learning**: show the model many `(prompt, response)`
pairs and train it to reproduce the responses. It's the first and often
most impactful post-training step.

## What SFT does

- Teaches **format** (chat turns, markdown, tool calls)
- Teaches **task patterns** (summarize, classify, code, etc.)
- Establishes a **baseline persona / tone**
- Unlocks instruction-following from the base model

## How SFT differs from pre-training

Both use next-token cross-entropy loss. The difference:

- **Pre-training** — predict every token in raw text
- **SFT** — only compute loss on the **assistant's** response tokens
  (the prompt is masked out)

This **loss masking** keeps the model from learning to generate user-style
text and focuses gradient on what we want it to produce.

## Chat templates

Each model family uses its own format to mark turns. Examples:

- **ChatML** — `<|im_start|>user ... <|im_end|>`
- **Llama-3** — `<|start_header_id|>user<|end_header_id|>`
- **Mistral** — `[INST] ... [/INST]`

Using the **wrong template** at fine-tune or inference time is a top-3
silent failure mode in practice.

## Full fine-tuning vs PEFT

| Approach           | Params trained | Compute     | Use when                  |
|--------------------|----------------|-------------|---------------------------|
| Full fine-tuning   | 100%           | Highest     | Frontier labs, big shifts |
| **LoRA**           | ~0.1–1%        | Low         | Most practitioners        |
| **QLoRA**          | ~0.1–1%        | Very low    | Single-GPU fine-tuning    |
| **DoRA, rsLoRA**   | Variants       | Low         | Marginal gains over LoRA  |

**LoRA** trains small "low-rank" adapter matrices alongside frozen weights.
**QLoRA** adds 4-bit quantization of the base model on top.

## Practical SFT engineering

- **Sequence packing** — concatenate short examples to fill context, big throughput win
- **Gradient checkpointing** — trade compute for memory
- **Mixed precision (bf16)** — standard
- **FSDP / ZeRO** — shard model across GPUs
- **Learning rate** — much lower than pre-training (1e-5 to 5e-5 typical)
- **Epochs** — usually 1–3; more risks overfitting and memorization

## Catastrophic forgetting

Fine-tuning too aggressively can wipe out pre-trained knowledge.
Mitigations:
- Low learning rate
- Few epochs
- Mix in some pre-training data ("replay")
- Use LoRA (smaller capacity change)
- KL regularization to the base model

## Common failure modes

- **Format collapse** — model outputs the same template every time
- **Mode collapse** — short, generic, hedging answers
- **Knowledge regression** — base-model facts get worse
- **Wrong-template inference** — silent quality degradation

## What to learn next

- Llama-3 / Mistral / Qwen instruct recipes
- LoRA and QLoRA papers
- Tulu-3 SFT mixture
- Loss masking implementation in TRL or Axolotl
