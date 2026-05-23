# 12 — Infrastructure & Tooling

The boring-but-essential layer underneath every post-training recipe.

## Open-source frameworks

| Framework         | Strengths                                       |
|-------------------|-------------------------------------------------|
| **TRL** (HF)      | SFT + DPO + PPO + GRPO, easy entry, popular     |
| **Axolotl**       | YAML configs for SFT/DPO, hides boilerplate     |
| **LLaMA-Factory** | GUI + CLI, supports many model families         |
| **OpenRLHF**      | RLHF/PPO at scale, Ray-based                    |
| **verl**          | RL framework focused on agentic/reasoning RL    |
| **NeMo-Aligner**  | NVIDIA's enterprise alignment stack             |
| **Unsloth**       | Single-GPU SFT, very fast for small fine-tunes  |

## Distributed training building blocks

- **DDP** — basic data parallelism
- **FSDP** — fully sharded data parallel (PyTorch native)
- **ZeRO** (DeepSpeed) — similar idea, stage 1/2/3 sharding
- **Tensor parallel** — split a single layer across GPUs
- **Pipeline parallel** — split the model by layer across GPUs
- **3D parallelism** — all three combined, for huge models

## Memory tricks

- **Mixed precision** (bf16) — standard
- **Gradient checkpointing** — recompute activations, save memory
- **Sequence packing** — fewer pad tokens, more throughput
- **Flash Attention** — faster, lower-memory attention
- **Quantized base model** (QLoRA-style) — 4-bit weights during SFT

## Compute footprints (rough order of magnitude)

| Task                          | Hardware                       |
|-------------------------------|--------------------------------|
| LoRA SFT of 7B                | 1× A100 / 24GB GPU             |
| Full SFT of 7B                | 4–8× A100                      |
| DPO of 7B                     | 4–8× A100                      |
| PPO of 7B                     | 8–32× A100                     |
| GRPO reasoning RL of 7B       | 32–128× A100/H100              |
| Anything at frontier scale    | Thousands of accelerators      |

## Reproducibility headaches

- Random seeds rarely fully reproducible at scale
- Hardware (A100 vs H100) changes numerics slightly
- Library versions matter — pin everything
- Tokenizer mismatches silently corrupt training

## Practical workflow

1. **Start tiny** — get the recipe working on a 1B model with 100 examples
2. **Scale up data** — verify the recipe is data-bound, not bug-bound
3. **Scale up model** — only after the recipe is stable
4. **Evals on every checkpoint** — catch regressions early
5. **Save artifacts** — data hashes, config, seeds, exact code

## What to learn next

- HF TRL docs and examples
- Axolotl docs
- DeepSpeed and FSDP guides
- "Survey of post-training infrastructure" papers and blog posts
