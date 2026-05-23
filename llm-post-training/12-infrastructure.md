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
| **NeMo-RL**       | NVIDIA RL post-training: GRPO/DPO, vLLM + PyTorch/Megatron, single-GPU → 1000-GPU |
| **NeMo-Skills**   | NVIDIA workflow glue: SDG + SFT + RL + eval, Slurm + Docker |
| **NeMo-Gym**      | NVIDIA multi-environment RL library for agent training |
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

## Training-inference numerical consistency

A subtle but important post-training infra concern: your **inference engine**
(vLLM, TGI, custom) and your **training forward pass** often produce
slightly different logits for the same input. Why this matters:

- In on-policy RL (PPO, GRPO), the sampler and the trainer **must agree**
  on logprobs; if they don't, "on-policy" silently becomes off-policy
- Standard mitigation is **importance-weighted correction** — adds bias
  and variance you didn't ask for
- Root cause is often **batch-size-dependent kernels** — reduction order,
  tile sizes, and split strategies all change with the batch dimension,
  so the same input produces different output at batch=1 vs batch=64

### Batch-invariant kernels (Thinking Machines, 2025)

"Defeating Nondeterminism in LLM Inference" showed you can fix this with:
- Fixed reduction strategies (no batch-dependent ordering)
- Consistent matmul tile sizes
- Fixed-size split strategies in attention
- Pre-computed KV-cache layouts so cached and current tokens reduce identically

Cost: ~**20% slower matmul**, ~**2× slower throughput overall**. Benefit:
**bitwise-identical inference**, and truly on-policy RL with **zero KL**
between sampling and training policies.

This is increasingly relevant as RL becomes the dominant post-training
stage (GRPO, R1-style reasoning) — sample efficiency depends directly on
the sampler and trainer staying in lockstep.

### End-to-end FP8 RL training (NVIDIA, 2026)

A different angle on the same problem: apply **FP8 to both** generation
(vLLM) and training (Megatron), rather than running them in different
precisions. Done naively, the precision mismatch between engines is the
biggest source of train/inference drift; matching them all the way down
reduces it.

- ~15–25% speedup on dense models; ~48% with FP8 KV cache + attention
- Block-wise quantization (`[128, 128]` blocks, FP32 scales)
- **Dynamic QKV scale recalibration** synced between training and
  inference each step
- Validation accuracy matches BF16 baselines when paired with importance
  sampling correction
- Token multiplicative probability error < 1.05 vs BF16

Demonstrated on GRPO with Llama-3.1-8B and Qwen3-30B.

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

## Key TRL infrastructure features

Beyond hosting most algorithms, TRL adds a steady stream of infra and
memory tricks. A sample:

| Feature                                       | Added in     |
|-----------------------------------------------|--------------|
| PEFT integration (20B+ on 24GB)               | v0.4         |
| QLoRA / 4-bit                                 | v0.4.2       |
| FSDP + QLoRA                                  | v0.8         |
| FSDP2                                         | v0.20        |
| Liger kernel integration                      | v0.18 / v0.22|
| vLLM sleep mode (colocated GRPO/RLOO)         | v0.23 / v0.25|
| Co-located vLLM with training                 | v0.18        |
| Multi-node vLLM + NCCL (GRPO 70B+)            | v0.16        |
| vLLM data parallelism (4× faster gen)         | v0.17        |
| vLLM prefix caching, V1 engine                | v0.15 / v0.17|
| Context Parallelism (long sequences)          | v0.23        |
| FFD / BFD sequence packing                    | v0.19 / v1.0 |
| Padding-free SFT                              | v0.16        |
| Chunked CE loss (-50% VRAM)                   | v1.4         |
| Chunked LM head for AsyncGRPO (-44× peak mem) | v1.1         |
| `trl vllm-serve` (+ speculative decoding)     | v1.3         |
| MFU helper functions                          | v1.4         |

See also: [`14-trl-library.md`](./14-trl-library.md) for the full version
timeline and algorithm-introduction order.

## What to learn next

- HF TRL docs and examples
- Axolotl docs
- DeepSpeed and FSDP guides
- "Survey of post-training infrastructure" papers and blog posts
