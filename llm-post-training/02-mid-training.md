# 02 — Mid-Training

## What it is

Mid-training is the phase **between bulk pre-training and post-training**.
The objective is still next-token prediction — so it's not SFT — but the
data is curated, the learning rate is decaying, and the goal is to **set
up the base model for everything downstream**.

A common framing: pre-training builds the world model, post-training
shapes behavior, and mid-training **sharpens the base** so post-training
has something good to work with.

> "You can't RL your way out of a bad base." Mid-training is where the
> ceiling for SFT/RLHF/RLVR gets set.

## Terminology

Many names point at overlapping ideas. Treat these as near-synonyms when
you read papers:

| Term                              | Emphasis                                   |
|-----------------------------------|--------------------------------------------|
| Mid-training                      | Distinct phase in the pipeline             |
| Annealing phase / decay phase     | The LR schedule view                       |
| Cooldown                          | Same as annealing (end-of-pretrain decay)  |
| Continued pre-training (CPT)      | Same objective, different data mix         |
| Domain-adaptive pre-training (DAPT) | CPT focused on a single domain           |
| Capability injection              | What you're trying to achieve              |

In **continued pre-training** the mixture weight on the original
pre-train data is often zero; in **mid-training** the original mix
typically remains 70–85% with 15–30% high-quality / domain data layered
in. The schedule and intent differ more than the loss function.

## Pre / mid / post comparison

| Phase         | Tokens         | Data                        | Loss             | LR                  |
|---------------|----------------|-----------------------------|------------------|---------------------|
| Pre-training  | 1T–15T+        | Web, books, code            | Next-token       | Peak, slowly decay  |
| Mid-training  | 10B–1T         | Curated HQ, math, code, CoT | Next-token       | Anneal toward zero  |
| Post-training | <10B           | Instructions, preferences   | SFT + RL / DPO   | Small, often constant |

The boundary between mid- and post-training is **fuzzy and shifting**.
Reasoning-heavy recipes (o1/R1-era) push more capability into
mid-training and lighter behavior shaping into post-training.

## The three axes

A useful organizing taxonomy from the 2025 mid-training surveys (arxiv
[2510.06826](https://arxiv.org/abs/2510.06826) and
[2510.23081](https://arxiv.org/abs/2510.23081)):

```
                    MID-TRAINING
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   Data distribution   LR schedule    Long-context
   (what to feed)      (how to decay) (how to extend)
```

These are designed jointly — e.g. WSD schedules pair naturally with a
high-quality decay-phase mix, and length extension typically uses its
own LR sub-phase.

## Axis 1 — Data distribution

Seven recurring data types in mid-training mixes:

1. **High-quality filtered web** — FineWeb-Edu, DCLM-baseline, Dolma.
   Stricter quality filters than pre-train.
2. **Code and math** — Stack v2, OpenWebMath, FineMath, Algebraic Stack.
3. **Instruction / QA data** — UltraChat, OpenOrca, FLAN. Light
   instruction priors *before* SFT proper. Timing is unresolved.
4. **Synthetic textbooks / knowledge-dense** — Cosmopedia, TuluMath,
   Phi-style "textbooks are all you need" data.
5. **Long-context filtered docs** — books, repos, concatenated
   documents, retrieval-bundled context.
6. **Reasoning / chain-of-thought traces** — math proofs, code
   executions, step-by-step solutions.
7. **Fill-in-Middle (FIM)** — bidirectional infilling targets for code
   models, sometimes generalized to text.

### Mix ratios

Typical pattern: **70–85% original pre-train + 15–30% high-quality /
domain data**. Some recipes go further — Zamba v1 uses 60/40. The point
is to avoid catastrophic forgetting while still shifting the
distribution.

### Empirical findings

- Switching the mid-training mix from *math + code* to *math + code +
  science* gave +3–6 points avg on reasoning benchmarks. The same data
  change applied **in RL instead** produced negligible gain.
- A small fraction of domain-specific data can produce disproportionate
  performance gains — high signal-to-noise tokens dominate.
- **Repetition substitutes for scale**: smaller HQ corpora repeated can
  beat larger un-repeated corpora.
- IBM (2025): mid-training boosted overall reasoning capability **3–4×**
  across model sizes/architectures while preserving pretrain knowledge.

## Axis 2 — Learning-rate scheduling

Mid-training is largely **defined by its LR schedule**. The schedule is
what makes "annealing" different from continued pre-train at a constant
LR.

| Schedule                          | Idea                                       |
|-----------------------------------|--------------------------------------------|
| Linear                            | Uniform decay from peak → 0                |
| Cosine                            | Cosine-shaped decay (still common)         |
| Exponential                       | Continuous exponential reduction           |
| Knee                              | Two-phase explore/exploit                  |
| Cyclical                          | Oscillating between bounds                 |
| **WSD (Warmup-Stable-Decay)**     | Hold flat, then sharp decay at the end     |
| Power                             | Power-law decay (scaling-invariant)        |
| Multi-step                        | Discrete drops at fixed milestones         |

**WSD** has become the dominant frame for modern mid-training (MiniCPM,
SmolLM2, OLMo 2). The "stable" portion is bulk pre-training; the
"decay" portion **is** mid-training, run on a higher-quality data mix.
This makes "where does pre-training end and mid-training begin?"
operationally precise — it's the schedule transition.

### Why the schedule matters

- Sharply decaying LR amplifies the influence of late-stage data — what
  the model sees at the end has outsized impact.
- This is **why** the data mix shifts to higher quality during decay:
  you're trading exploration for refinement.
- Heuristic in practice: most labs admit the schedule choice is
  empirical, not theoretically grounded.

## Axis 3 — Long-context extension

Length extension is almost always done in a dedicated mid-training
sub-phase (rather than during bulk pre-train or in SFT). The methods
all remap RoPE base frequencies:

| Method                  | Idea                                            |
|-------------------------|-------------------------------------------------|
| Position Interpolation  | Uniform frequency compression                   |
| NTK-aware               | Differential scaling by frequency band          |
| NTK-by-parts            | Piecewise frequency treatment                   |
| YaRN                    | NTK + length-dependent stabilization            |
| LongRoPE                | Learned non-uniform remapping, progressive      |

Length extension is typically done in **multiple stages** (e.g.
4K → 32K → 128K → 256K) with the data filtered for actually-long
documents at each stage.

Full treatment: see [12-long-context.md](./12-long-context.md).

## Theoretical framing

Three lenses for why mid-training works:

- **Gradient noise scale** — high-quality data raises the
  signal-to-noise ratio of each gradient update; decaying LR concentrates
  the update budget on these higher-quality batches.
- **Information bottleneck** — late training compresses representations
  toward task-relevant features; the curated mix biases what counts as
  "task-relevant".
- **Curriculum learning** — progression from diverse pre-train to
  specialized mid-train mirrors structured skill acquisition.

None of these are airtight, but they pin down why a phase that looks
algorithmically identical to pre-training is treated as distinct.

## Capability-injection patterns

What mid-training is actually used for in practice:

- **Reasoning warmup** — heavy math + code + CoT mix; the largest
  single source of reasoning gains for o1/R1-style recipes.
- **Long context** — staged length extension as above.
- **Domain adaptation** — medical, legal, finance, scientific corpora.
- **Multilingual** — upsample target languages; sometimes a dedicated
  multilingual mid-train stage.
- **Tool-use / function-calling priors** — synthetic tool-call traces
  introduced before SFT so the model knows the format.
- **Format priors** — instruction-shaped sequences without becoming SFT
  proper (the "instruction data timing" open question).

## Frontier recipes

Representative mid-training designs:

| Model              | Data strategy                          | LR schedule    | Length extension          |
|--------------------|----------------------------------------|----------------|---------------------------|
| Llama 3 (405B)     | 50% general / 25% math / 17% code      | cosine → linear| 6-phase → 256K            |
| Qwen3              | 30T → 5T → long-context phases         | cosine         | progressive               |
| Phi-4              | 10T general + heavy synthetic reasoning| linear         | curated long-context      |
| DeepSeek-V3        | 14.8T general + 120B context-extension | multi-stage    | 32K → 128K (2-phase)      |
| OLMo 2             | Dolmino Mix 1124 (first public mid-mix)| WSD            | progressive               |
| MiniCPM            | 1T coarse + 20B fine-grained anneal    | WSD            | single annealing phase    |
| SmolLM2            | 11T across 3 + 1 stages                | WSD            | 2K → 8K progressive       |
| Zamba v1           | 60% pretrain / 40% HQ                  | —              | —                         |

Common pattern: a **fast-decaying final phase** trained on the highest-
quality fraction of the corpus, with length extension either tucked
into that phase or run as its own sub-phase right after.

## Relationship to post-training

- **Mid-training sets the ceiling.** SFT and RL can only elicit and
  shape what the base already has. Reasoning that doesn't appear at all
  in mid-training is very hard to RL into the model later.
- **Front-loading reasoning** ([arxiv 2510.03264](https://arxiv.org/abs/2510.03264))
  argues that mid-training and post-training data are synergistic —
  reasoning data placed in mid-training amplifies the gain from later
  RLVR more than the same data placed in SFT.
- **Bridging view** ([arxiv 2510.14865](https://arxiv.org/abs/2510.14865))
  frames mid-training as an explicit **distribution bridge** between
  pre-train and post-train, smoothing the shift that pure SFT would
  otherwise have to absorb.
- Practical consequence: when an RL run plateaus, the right knob is
  often to add to the mid-train mix, not to tune the RL.

## Evaluation during mid-training

- Loss on held-out high-quality slices (math, code, reasoning).
- Benchmark deltas across the annealing curve — track every N% of
  decay, not just the endpoint.
- Capability emergence vs memorization — if a benchmark improves but
  loss on a paraphrased version doesn't, you have memorization.
- Forgetting checks — keep a "pre-train representative" eval to catch
  catastrophic drift.

## Common pitfalls

- **Catastrophic forgetting** — mid-train mix too far from pre-train,
  general capability collapses.
- **Over-annealing** — LR decays so aggressively the model loses
  generative diversity; recovery is hard.
- **Distributional discontinuity** — sudden mix shifts (vs gradual
  ramping) hurt more than they help.
- **Instruction-data contamination** — too much instruction-style data
  blurs into SFT and causes downstream RL instability.
- **Long-context cliffs** — extending length without ramping data
  composition leaves the model technically able but practically bad at
  long inputs.

## Emerging directions

- **RL-guided mid-training** (ReMiT) — use an RL-tuned reasoner's
  priors to **reweight tokens** during mid-training so the next base
  model is easier to RL. A feedback loop between mid- and post-training.
- **Mid-training-first reasoning recipes** — pushing more of the
  reasoning capability budget out of expensive RL and into cheaper
  next-token mid-training.
- **Synthetic-data scaling in mid-train** — teacher-generated,
  verified, and rephrased corpora replacing scarcer organic HQ data.

## Open problems

- Optimal placement of instruction-style data (mid? post? both?).
- Scheduler design lacks theoretical grounding.
- Long-context vs short-context performance trade-offs are not
  characterized.
- Scheduler × batch size × model scale interactions are system-dependent.
- No public consensus on how much capability should sit in mid-train
  vs post-train for a given downstream goal.

## What to learn next

- *Mid-Training of Large Language Models: A Survey*
  ([arxiv 2510.06826](https://arxiv.org/abs/2510.06826))
- *A Survey on LLM Mid-Training*
  ([arxiv 2510.23081](https://arxiv.org/abs/2510.23081))
- *Midtraining Bridges Pretraining and Posttraining Distributions*
  ([arxiv 2510.14865](https://arxiv.org/abs/2510.14865))
- *Front-Loading Reasoning: The Synergy between Pretraining and
  Post-Training Data* ([arxiv 2510.03264](https://arxiv.org/abs/2510.03264))
- OLMo 2 / Dolmino Mix 1124 release notes — first public systematic
  mid-train dataset
- MiniCPM technical report — WSD schedule in detail
- Llama 3 paper — annealing + 6-phase length extension
