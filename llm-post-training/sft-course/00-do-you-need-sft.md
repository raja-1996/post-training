# 00 — Do you even need SFT?

> **Goal of this lesson:** before you spend a weekend (and a GPU budget) on
> a fine-tune, walk a decision tree that rules SFT in or out. Most "we need
> to fine-tune" requests are actually solved a rung lower on the ladder.

**Reading:** ~20 min · **Hands-on:** ~30 min · **Prereqs:** familiarity with
calling an LLM via API or `transformers`.

---

## 1. Concept — the capability ladder

When a model isn't doing what you want, climb this ladder **from the
bottom**. Stop the moment you have a solution that's good enough; each
step up is roughly an order of magnitude more expensive and more brittle.

```
                              cost / complexity
                                      ▲
   6. RL (PPO / GRPO / online DPO)    │  needs reward signal, infra, eval
   5. Preference optimization (DPO)   │  needs preference pairs
   4. SFT (LoRA → QLoRA → full FT)    │  needs labeled (prompt, response)
   3. RAG / tools                     │  needs a retriever or tool layer
   2. Few-shot + structured output    │  needs 2–10 worked examples
   1. Prompt engineering              │  free, iterate in minutes
   0. Pick a better/bigger model      │  often the whole answer
                                      │
                              cheap / fast
```

A useful reframe: SFT teaches **behavior** (format, style, task pattern,
tool-call shape). It is not a great way to teach **facts** — that's what
pre-training and RAG are for. If your failure mode is "the model doesn't
know X," fine-tuning is usually the wrong tool.

### What SFT is actually good at

| Problem                                                              | SFT helps? |
|----------------------------------------------------------------------|------------|
| Always output strict JSON with these 7 fields                        | Yes        |
| Match our brand voice / tone across 1000s of outputs                 | Yes        |
| Follow a domain-specific instruction schema (e.g. medical SOAP notes)| Yes        |
| Emit tool calls in our exact format                                  | Yes        |
| Long chain-of-thought in a specific shape                            | Yes        |
| Answer questions about *our* private docs                            | **No → RAG** |
| Know yesterday's news                                                | **No → tools/RAG** |
| Stop hallucinating in general                                        | **No / partial** |
| Be more helpful / harmless / honest in open-ended chat               | **Partial → SFT then DPO/RLHF** |

### When *not* to SFT (common traps)

- **Your eval is vibes.** No held-out set, no metric → you cannot tell
  whether the fine-tune helped, hurt, or just memorized. Build eval first.
- **You have <100 high-quality examples.** Either iterate on prompting,
  or invest in data first.
- **You haven't tried a bigger model.** Going from a 7B to a 70B (or
  swapping in a frontier API) often closes the gap with no training at all.
- **Your task changes weekly.** Retraining cadence will dominate cost.
- **You want recency / freshness.** That's retrieval, not weights.
- **The base model already does it 95% of the time.** Fine-tuning a
  near-solved task often makes things *worse* (regression on edge cases,
  format collapse, loss of generality).

---

## 2. The decision tree (use this)

```
START
  │
  ▼
Have you written down 5–10 concrete failing examples?
  │
  ├── No  ──► Do that first. You cannot improve what you cannot measure.
  │
  ▼ Yes
Is the failure mode about FACTS the model doesn't know?
  │
  ├── Yes ──► RAG / tools / better context. Not SFT.
  │
  ▼ No
Did a stronger prompt + 3 few-shot examples fix ≥80%?
  │
  ├── Yes ──► Ship the prompt. Stop here.
  │
  ▼ No
Did swapping to a bigger / better base model fix it?
  │
  ├── Yes ──► Use that model. Stop here.
  │
  ▼ No
Can you produce ≥500 (prompt, response) pairs at the quality you want?
(human-written, distilled from a stronger model, or hybrid)
  │
  ├── No  ──► Go build data. SFT without data is theatre.
  │
  ▼ Yes
Is the task primarily BEHAVIOR (format / style / structure / tool calls)?
  │
  ├── Yes ──► SFT is the right tool.   ──►  Continue the course.
  │
  ▼ No (it's a preference / quality / "be more X" problem)
SFT first to nail format, THEN DPO / RLHF for preferences.
```

Two heuristics that save weeks:

1. **"1K good > 100K meh."** A thousand carefully-curated examples will
   usually beat 100K scraped ones. Quality > quantity is a recurring
   finding (LIMA, Tulu-3, and most production reports agree).
2. **If GPT-class or Claude-class can do it zero-shot, you can almost
   certainly distill it into a small open model with SFT.** This is the
   single most common, highest-ROI use of SFT in 2025–2026.

---

## 3. Minimal code — actually walk the ladder

Before reaching for `SFTTrainer`, run this 15-minute experiment on your
own task. The point is to **falsify the cheaper options first**.

```python
# pip install "transformers>=4.45" accelerate
import os
from transformers import pipeline

MODEL = "Qwen/Qwen2.5-7B-Instruct"  # or any instruct model you can run

# 1. Your evaluation set — write these BY HAND. 10 is plenty to start.
eval_set = [
    {"input": "...", "expected_traits": ["valid JSON", "has field `foo`"]},
    # ... 9 more, covering easy / typical / hard / adversarial
]

gen = pipeline("text-generation", model=MODEL, device_map="auto",
               torch_dtype="bfloat16")

def score(output, traits):
    """Replace with whatever 'good' means for YOUR task.
    Start with simple checks: JSON-parses, contains keyword, regex match."""
    return sum(t_check(output, t) for t in traits) / len(traits)

# --- Rung 1: bare prompt ---
prompt_v1 = "Extract the action items as a JSON list: {input}"

# --- Rung 2: stronger instruction + schema ---
prompt_v2 = """You are an extraction system. Output ONLY valid JSON of the form:
{"action_items": [{"who": str, "what": str, "due": str|null}]}

Text:
{input}

JSON:"""

# --- Rung 3: + 3 few-shot examples ---
FEWSHOT = """Example 1:
Text: "Alice will send the report by Friday."
JSON: {"action_items":[{"who":"Alice","what":"send the report","due":"Friday"}]}
... (2 more) ..."""
prompt_v3 = FEWSHOT + "\n\nText:\n{input}\n\nJSON:"

for name, p in [("v1", prompt_v1), ("v2", prompt_v2), ("v3", prompt_v3)]:
    scores = []
    for ex in eval_set:
        out = gen(p.format(input=ex["input"]),
                  max_new_tokens=256, do_sample=False)[0]["generated_text"]
        scores.append(score(out, ex["expected_traits"]))
    print(name, sum(scores) / len(scores))
```

**Decision rule:** if `v3` clears your bar, you're done. Ship the prompt
and walk away. If it doesn't, you now have a concrete baseline number to
beat with SFT — which is also exactly what you'll need to justify the
training cost.

---

## 4. Try it yourself (30 min)

Pick one real task you actually care about and:

1. Write **10 inputs** and the **trait checklist** you'd use to grade an
   output (binary or 0–1).
2. Run the three prompt rungs above against an instruct model you can
   access (open or API).
3. Compute average score per rung. Eyeball the failures.
4. Answer, on paper:
   - Is the residual failure about *behavior* or about *knowledge*?
   - Could 500 well-written examples plausibly fix it?
   - What would your eval set need to grow to (50? 200?) before you'd
     trust a fine-tune was actually better?

If you walk away with a written "yes, SFT, because X" or "no, SFT,
because Y" — this lesson did its job.

---

## 5. Pitfalls

- **Skipping eval.** "We'll figure out evaluation after training" is the
  single most common way SFT projects fail. Build eval first; the
  fine-tune is a function of the eval, not the other way around.
- **Confusing "model is bad at X" with "model doesn't know X".** The
  first is behavior (SFT-able). The second is facts (RAG / context).
- **Comparing fine-tuned-small to base-small.** Always also compare to
  prompted-large. If a 70B with a good prompt beats your fine-tuned 7B,
  the fine-tune has to justify itself on cost, latency, or privacy —
  not quality.
- **One-shot fine-tuning a moving target.** If the task definition will
  drift in 2 weeks, prompting / RAG buys you flexibility that weights
  don't.
- **Treating SFT as a knowledge-injection method.** It can memorize a
  small set of facts, but it does so unreliably and at the cost of
  general capability. Use retrieval.
- **Ignoring base-vs-instruct.** Fine-tuning on top of an already
  instruction-tuned model needs *much* lower LR than on a base model,
  or you'll nuke alignment. Covered in lesson 04.

---

## 6. Further reading

- LIMA: *Less Is More for Alignment* — 1K curated examples rivals much
  larger SFT corpora.
- Tulu-3 technical report — production-grade data-quality recipe.
- The top-level [`../04-sft.md`](../04-sft.md) overview — read after this.
- Next up: [`00-5-environment-setup.md`](./00-5-environment-setup.md) —
  get CUDA / torch / trl / peft / bnb / accelerate / flash-attn working
  and do the VRAM math before lesson 01.

---

**TL;DR:** SFT is the right answer when (a) you have an eval, (b) the
failure is behavior not facts, (c) prompting and a bigger model didn't
close it, and (d) you can produce a few hundred high-quality
`(prompt, response)` pairs. Otherwise climb back down the ladder.
