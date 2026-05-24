# 04 — Choosing a base model

> **Goal of this lesson:** make the model-selection decision *before*
> you build a dataset, not after. Pick along five axes (base-vs-instruct,
> size, license, context length, tokenizer quality) and know how each
> one changes your downstream recipe.

**Reading:** ~20 min · **Hands-on:** ~30 min · **Prereqs:** lessons
[`00`](./00-do-you-need-sft.md) through [`03`](./03-chat-templates.md).

---

## 1. Concept — five axes, ranked by impact

Most "which model should I fine-tune?" questions collapse to these
five, in roughly this order of impact on your final result:

1. **Base vs instruct** — what the model already knows about chat.
2. **Size** — capacity ceiling and how much VRAM you'll need.
3. **Tokenizer / vocab quality** — coverage of your domain.
4. **Context length** — does your data fit?
5. **License** — can you actually ship this?

Get these wrong and no amount of data or hyperparameter tuning fixes
it. Get them right and a small dataset goes a long way.

---

## 2. Base vs instruct — the single most important choice

### Base models

A **base model** has only seen pre-training data — raw text, no chat
template, no instruction-following. Outputs are completions of whatever
prefix you give it.

- Examples: `meta-llama/Llama-3.1-8B`, `Qwen/Qwen2.5-7B`, `mistralai/Mistral-7B-v0.3`.
- No chat template (or a placeholder one). You bring your own.
- Refusal behavior, safety conditioning: none.
- Knowledge: maximal — nothing has been "trained away."

### Instruct models

An **instruct model** is a base model that has already been SFT'd
(and often DPO'd or RLHF'd) on chat data.

- Examples: `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`,
  `mistralai/Mistral-7B-Instruct-v0.3`.
- Ships with a chat template that *must* be respected.
- Already follows instructions, has a persona, has refusal behavior.
- May have lost some raw-completion capability.

### Which to start from?

| Situation                                          | Start from   | Why                                                              |
|----------------------------------------------------|--------------|------------------------------------------------------------------|
| New chat persona / new task format / large dataset | **Base**     | More headroom; you control the format end-to-end                 |
| Small dataset (<5K), task close to general chat    | **Instruct** | Inherits instruction-following for free; tiny data won't teach it|
| Domain-specific assistant (medical, legal, code)   | **Instruct** | Format wins inherited; you only have to teach the domain shift   |
| Format-only / extraction / strict structured output| Either; **base** if you have >20K examples                       |
| Need to remove existing safety/refusal             | **Base**     | Easier than fighting instruct conditioning                       |
| Distilling from a frontier model                   | **Base**     | Avoid double-training; you're recreating an instruct model anyway|

### The "instruct LR" trap

Fine-tuning *on top of* an instruct model demands a **much lower
learning rate** than fine-tuning a base model. Typical numbers:

| Starting point | Full-FT LR    | LoRA LR (≈10× full-FT) |
|----------------|---------------|------------------------|
| Base           | 2e-5 – 5e-5   | 2e-4 – 5e-4            |
| Instruct       | 5e-6 – 2e-5   | 5e-5 – 2e-4            |

Use the same LR you'd use on a base, and you'll happily overwrite the
instruct model's alignment in a few hundred steps. Symptoms:
hallucinated refusals, format collapse, sudden loss of "be helpful"
behavior on out-of-distribution prompts. Covered in detail in lessons
10 and 13.

---

## 3. Size — capacity, VRAM, and a useful heuristic

A rough rule of thumb that has held up across model generations:

| Size  | Good for                                                                      | QLoRA VRAM (24 GB card) |
|-------|-------------------------------------------------------------------------------|--------------------------|
| 0.5–1.5B | Single-task extraction, classification, format enforcement                 | Easily; bf16 works too   |
| 3B    | Lightweight task models; on-device                                            | Comfortable              |
| 7–8B  | Default workhorse; most domain assistants                                     | Standard QLoRA setup     |
| 13–14B| Better reasoning / multi-step; meaningful jump over 7B                        | Tight; reduce seq_len    |
| 30–34B| Strong general capability; harder to fine-tune well on small data             | Needs ≥2× 24 GB or A100  |
| 70B+  | Frontier-ish; full-FT impossible on commodity HW; QLoRA + FSDP                | Multi-GPU only           |

Heuristic: **the smallest model that solves your task is the right one**.
Bigger is more expensive to train, slower at inference, and is *not*
necessarily better when your training data is small — it overfits
faster.

Another heuristic: if you've never fine-tuned this model family before,
start with a 1.5B–3B version of it. Get your pipeline working there
(loss goes down, generations look right, eval improves). Then scale up.
This pays for itself in saved GPU hours within a week.

---

## 4. Tokenizer / vocab quality — the silent quality multiplier

Two different tokenizers will assign different token counts to the same
text, and the difference matters in three ways:

- **Cost.** More tokens per example → longer sequences → more VRAM and
  more wall-clock per step.
- **Information density.** A token that gets split into 4 pieces
  (e.g. domain-specific jargon or non-English) is harder for the model
  to learn than one that maps to a single token.
- **Effective context.** A 32K context window with a fat tokenizer can
  fit half the actual content of a 32K window with a lean one.

Quick check on your real data:

```python
from transformers import AutoTokenizer
sample = open("my_data_sample.txt").read()

for m in ["Qwen/Qwen2.5-7B", "meta-llama/Llama-3.1-8B",
          "mistralai/Mistral-7B-v0.3", "google/gemma-2-9b"]:
    tok = AutoTokenizer.from_pretrained(m)
    n = len(tok(sample).input_ids)
    print(f"{m:40s}  {n:>7} tokens   ({n / len(sample):.3f} tok/char)")
```

A 20% advantage on tok-per-char compounds into a real wall-clock
difference over a training run. Especially important for:

- Code (Llama-3 and Qwen 2.5 are significantly better than older
  tokenizers on most code).
- Non-English text (Gemma and Qwen often beat Llama; check your
  target language).
- Specialized notation (math symbols, chemistry, log lines).

---

## 5. Context length — both numbers and which one

Models advertise a context length (e.g. 8K, 32K, 128K), but two
practical points often get missed:

- **Training data context.** If your dataset has examples that are
  10K tokens long but your `SFTConfig` sets `max_seq_length=2048`,
  every example is silently truncated. You lose the tail of every long
  example.
- **Pre-trained vs RoPE-extended.** Some models are pre-trained at a
  short context and have their RoPE base rescaled to claim long context.
  Fine-tuning at the full extended context can *degrade* the long-context
  ability if you don't include enough long examples. Lesson 12 covers
  this.

Practical rule: pick a model whose pre-trained context comfortably
covers your 95th-percentile example length. If 95% of your data fits in
4K, an 8K-context model is fine — don't pay the VRAM premium for 128K.

---

## 6. License — what you can actually ship

This kills more projects post-hoc than any technical issue. Check the
license *before* you train, not before you deploy.

| Family / model               | License (simplified)                | Commercial use? | Notable strings attached                |
|------------------------------|--------------------------------------|-----------------|------------------------------------------|
| Llama 3.x                    | Llama Community License              | Yes, with MAU cap (>700M MAU needs Meta agreement) | Naming attribution, AUP |
| Qwen 2.5 (most sizes)        | Apache 2.0                            | Yes             | Very permissive                          |
| Qwen 2.5 72B                 | Custom (Qianwen License)              | Yes, with cap   | Read the actual license                  |
| Mistral / Mixtral (Apache)   | Apache 2.0                            | Yes             | Permissive                               |
| Mistral "Research" models    | MNPL / non-commercial                 | **No**          | Watch for these specifically             |
| Gemma 2 / 3                  | Gemma Terms of Use                    | Yes             | AUP applies; redistribution rules        |
| DeepSeek (most)              | DeepSeek License / MIT-ish            | Yes             | Check per-model                          |
| Phi-3 / Phi-4                | MIT                                   | Yes             | Permissive                               |

**Practical rules:**

- Apache 2.0 / MIT — safest. No-brainer for commercial work.
- Llama Community License — fine for most shops; check the MAU clause
  and naming attribution (you must include "Llama" in the derived model
  name).
- Anything called "Research" or "Non-Commercial" — do not use for
  anything you will ship.
- Always look at the actual `LICENSE` file on the model's HF repo;
  family-level summaries can be out of date.

---

## 7. A decision template you can copy

Fill this out on paper *before* downloading anything.

```
TASK:               [one sentence]
ESTIMATED DATA:     [N examples, M tokens each at p95]
SHIP TARGET:        [API? on-device? offline?]

Base vs instruct:   [base / instruct]   because: [...]
Family candidates:  [Llama 3.1 / Qwen 2.5 / Gemma 2 / Mistral / ...]
Size:               [1.5B / 7B / 13B / ...]
Tokenizer check:    [tok/char on sample = ...]
Context budget:     [p95 of my data = X; model native ctx = Y]
License OK to ship: [yes / no / needs review]

VRAM check:         [fits in QLoRA on my hardware: yes / no]
First-pass model:   [ORG/MODEL_ID]
Fallback if eval is poor: [next model up; next family]
```

If you can't fill in all the rows, you're not ready to train.

---

## 8. Minimal code — short-list comparison on your data

```python
# pip install "transformers>=4.45" datasets
from transformers import AutoTokenizer, AutoConfig

CANDIDATES = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
]

# Bring 50–200 real examples from your task here
samples = [
    "Extract the action items as JSON: 'Alice will ...'",
    "Patient presents with intermittent chest pain ...",
    # ...
]

for name in CANDIDATES:
    cfg = AutoConfig.from_pretrained(name)
    tok = AutoTokenizer.from_pretrained(name)
    has_template = tok.chat_template is not None

    lengths = [len(tok(s).input_ids) for s in samples]
    p95 = sorted(lengths)[int(0.95 * len(lengths))]
    avg = sum(lengths) / len(lengths)

    print(f"{name}")
    print(f"   params (config)   : {cfg.num_hidden_layers}L x {cfg.hidden_size}d")
    print(f"   max_position_emb  : {cfg.max_position_embeddings}")
    print(f"   chat template     : {'yes' if has_template else 'NO'}")
    print(f"   tokens avg / p95  : {avg:.0f} / {p95}")
    print()
```

What you're looking for:

- p95 token length comfortably below `max_position_embeddings`.
- Tokens-per-example differences across models — favor the leaner one
  if all else is equal.
- Presence of a chat template (or a plan to add one for the base
  variant).

---

## 9. Try it yourself (30 min)

1. Fill out the decision template (§7) for the task you actually plan
   to fine-tune. Don't skip a row.
2. Run §8 against 100 real examples from your task. Save the table; it's
   what justifies your model choice in the eventual model card (lesson
   15.8).
3. Pick **two** candidates, not one. Plan to train your first run on the
   smaller / cheaper one to validate the pipeline, then scale up.
4. Open the license file of your chosen model on HF Hub. Read it. If
   you're in a team, send the link to whoever signs off on ship
   decisions. Saves a painful conversation later.

---

## 10. Pitfalls

- **Defaulting to "biggest you can fit."** Bigger models overfit faster
  on small data and waste serving budget. Pick the smallest that solves
  the task.
- **Mixing instruct + high LR.** Detailed in §2. Always halve (or
  quarter) your LR vs a base-model recipe.
- **Forgetting the base/instruct distinction at data time.** A dataset
  written for a base model (raw prompt → raw response) does not
  trivially work on an instruct model; you need to wrap it in the chat
  template. And vice versa.
- **Tokenizer-blind comparison.** Two 7B models are not the same model;
  different tokenizers mean different effective capacity and different
  serving cost.
- **Trusting "context length" marketing.** A 128K-claimed model that
  was pre-trained at 8K and extended by RoPE rescaling is *not* the
  same as a model natively pre-trained on long sequences. Lesson 12.
- **License check at deploy time.** "We trained on a Mistral Research
  model" is a sentence you do not want to hear in a launch review.
- **Skipping the smaller-model dry run.** Going straight to a 70B
  fine-tune on day one means every pipeline bug costs 10× more to
  debug.
- **Picking based on benchmark scores.** Benchmark leaderboards mix
  bases and instructs, often test capabilities orthogonal to your task,
  and are gameable. Run §8 on *your* data instead.
- **Family lock-in.** Building all your prompts / few-shot examples /
  eval scaffolding around one family makes it expensive to switch later.
  Keep your data and eval **template-agnostic** so you can re-render
  for a different family in an afternoon.

---

## 11. Further reading

- Model cards on HF Hub for each candidate — the only ground truth for
  license, context, and recommended chat template.
- *Llama 3 Tech Report* — the chapter on context-length extension is a
  good companion for lesson 12.
- *Qwen 2.5 Tech Report* — the section on tokenizer design is useful
  context for §4.
- Top-level [`../04-sft.md`](../04-sft.md) — "Full FT vs PEFT" sets up
  the next lesson.
- Next: [`05-full-ft-vs-lora.md`](./05-full-ft-vs-lora.md) — given the
  model you just picked, *how* are you going to train it?

---

**TL;DR:** Pick along five axes: base-vs-instruct (drives your LR),
size (smallest that works), tokenizer (run §8 on your data),
context (p95 ≤ native pretrain ctx), license (Apache/MIT > Llama
Community > anything labeled "Research"). Decide before you build data.
