# 01 — What SFT actually is

> **Goal of this lesson:** internalize the one-line definition of SFT,
> the loss it optimizes, and *why* it differs from pre-training by
> exactly one detail (masking). By the end you should be able to
> hand-compute the loss for a 3-token example.

**Reading:** ~20 min · **Hands-on:** ~30 min · **Prereqs:** lessons
[`00`](./00-do-you-need-sft.md) and [`00.5`](./00-5-environment-setup.md).
Comfort with cross-entropy.

---

## 1. Concept — one line, then unpacked

> **SFT is next-token cross-entropy, computed only on the response tokens.**

That's it. Same loss as pre-training, same architecture, same optimizer
family. The only change is **which tokens contribute to the loss**.

### The pre-training loss

Given a sequence of tokens $x_1, x_2, \dots, x_T$, a causal LM is
trained to predict each next token from the previous ones:

$$
\mathcal{L}_{\text{pretrain}} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

Every token contributes. The model learns the joint distribution of
*all* text.

### The SFT loss

We now split the sequence into a **prompt** (tokens $1..P$) and a
**response** (tokens $P+1..T$). We compute loss **only on response
tokens**:

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{T - P}\sum_{t=P+1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

Mechanically, in HuggingFace land this is implemented by setting the
`labels` tensor to `-100` on the prompt positions. PyTorch's cross-entropy
ignores `-100` by default, so those positions contribute zero gradient.

```
tokens:  [<s> You are helpful. <usr> Capital of France? <asst> Paris </s>]
labels:  [-100 -100 -100 ... -100 -100 -100  ... -100 Paris </s>]
                  prompt (masked)                       response (loss here)
```

### Why mask the prompt

Two reasons, both practical:

1. **You don't want to teach the model to generate user-style text.**
   If the prompt contributes to the loss, gradient flows toward
   producing *prompts*, which is the opposite of what we want at
   inference.
2. **Gradient budget.** Prompts are often much longer than responses.
   Without masking, the model spends most of its capacity reconstructing
   inputs it will never need to generate.

A useful mental model: **SFT is imitation learning by teacher forcing**.
At every response position, the model sees the *correct* previous tokens
(from the dataset) and is asked to predict the next correct token. It
never sees its own (potentially wrong) outputs during training. That's
both the strength (stable, sample-efficient) and the weakness (exposure
bias — covered in lesson 14) of SFT.

---

## 2. A worked example you should be able to do by hand

Suppose your vocab is `{Paris=0, London=1, Rome=2}` and your model
outputs logits `[2.0, 1.0, 0.0]` for the position that should be
`Paris`.

Softmax probabilities:

```
p(Paris)  = e^2  / (e^2 + e^1 + e^0) = 7.389 / 11.107 ≈ 0.665
p(London) = e^1  / ...               ≈ 0.245
p(Rome)   = e^0  / ...               ≈ 0.090
```

Loss at that position: `-log(0.665) ≈ 0.408`.

If the dataset's response is just the single token `Paris`, **that's the
loss for the example.** The prompt tokens that came before contributed
nothing because their labels were `-100`.

Scale this up: per-token CE summed across all response positions, then
averaged. That's the entire training signal.

---

## 3. Minimal code — see the masking with your own eyes

```python
# pip install "transformers>=4.45" torch
import torch
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

messages = [
    {"role": "user", "content": "Capital of France?"},
    {"role": "assistant", "content": "Paris."},
]

# Render with the model's chat template
full = tok.apply_chat_template(messages, tokenize=False)
prompt_only = tok.apply_chat_template(messages[:-1], tokenize=False,
                                      add_generation_prompt=True)

full_ids   = tok(full,        add_special_tokens=False).input_ids
prompt_ids = tok(prompt_only, add_special_tokens=False).input_ids

# Build labels: copy of input_ids, with the prompt span masked to -100
labels = list(full_ids)
for i in range(len(prompt_ids)):
    labels[i] = -100

# Pretty-print so you can see what gets trained on
for tid, lab in zip(full_ids, labels):
    tok_str = tok.decode([tid])
    marker  = " <-- LOSS" if lab != -100 else ""
    print(f"{tid:>6}  {tok_str!r:<20s}  label={lab}{marker}")
```

What you should see: every token of the user turn (and the chat
template scaffolding leading up to the assistant turn) prints with
`label=-100`. The assistant's response tokens print with their real
token IDs and the `<-- LOSS` marker.

> **Heads-up:** the messy details of "where exactly does the prompt
> end?" — especially with multi-turn conversations and idiosyncratic
> templates — is the whole subject of lesson 02. For now: one user turn,
> one assistant turn. Get the picture clean before adding turns.

### Compute the loss for one batch

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.bfloat16, device_map="auto",
)

input_ids = torch.tensor([full_ids], device=model.device)
labels_t  = torch.tensor([labels],   device=model.device)

with torch.no_grad():
    out = model(input_ids=input_ids, labels=labels_t)

print("loss on response tokens only:", out.loss.item())

# Sanity check: if you UNMASK the prompt, the loss should change.
unmasked = torch.tensor([full_ids], device=model.device)
with torch.no_grad():
    out2 = model(input_ids=input_ids, labels=unmasked)
print("loss on ALL tokens (pre-training-style):", out2.loss.item())
```

These two numbers will differ, often substantially. The first is what
SFT optimizes. The second is what pre-training optimizes. **That single
difference is the entire conceptual content of SFT.**

---

## 4. Try it yourself (30 min)

1. Run the masking visualizer above on your own example with a longer
   prompt (e.g., a 3-sentence instruction + a short response). Count
   how many tokens get a real label vs `-100`. Note the ratio —
   that's your gradient-efficiency.
2. Pick a tokenizer for a different family (e.g., Llama-3 or Mistral
   instruct). Re-run the visualizer. Look at *where* the prompt/response
   boundary lands — different templates use different special tokens.
   This is the seed of lesson 03.
3. Try this experiment: take a single (prompt, response) pair, train for
   200 steps with a very high LR (e.g. `1e-3`), and watch the loss go to
   ~0. That's overfitting one example — a great way to confirm your
   training loop is wired up correctly. **If the loss does *not* go to
   zero, something is wrong with your masking, template, or label
   shifting.** Memorize this sanity check; you'll come back to it.

---

## 5. Pitfalls

- **"SFT is a different algorithm than pre-training."** It isn't. Same
  loss, same optimizer, same architecture. *Only the labels differ.*
  Internalizing this prevents a lot of mystical thinking about what SFT
  "does to the model."
- **Confusing teacher forcing with inference.** At training time the
  model always sees the *correct* previous tokens. At inference it sees
  its *own* previous tokens, which may be wrong. The gap between these
  is exposure bias — a real failure mode, and the structural reason
  on-policy methods (RL, online DPO) exist.
- **Masking off-by-one.** HF computes the loss internally on positions
  shifted by one (`logits[..., :-1, :]` vs `labels[..., 1:]`). You set
  `labels` aligned to `input_ids`; the shift happens inside the model.
  Don't shift it yourself — you'll double-shift and get gibberish loss.
- **Masking the wrong end.** Masking the *response* and keeping the
  prompt unmasked trains the model to generate prompts, not responses.
  Always sanity-check by printing the masked sequence (step 1 above).
- **Including the assistant's `<|im_start|>`-style header in the loss.**
  Subtle but real: do you want loss on the literal tokens that mark
  "now the assistant speaks"? Most people: no, those are scaffolding,
  not content. Lesson 02 has the canonical recipe.
- **Loss averaged differently than you think.** Most trainers average
  per-token across the *batch*. Long examples therefore contribute more
  total gradient than short ones — which matters when you mix tasks.
  See lesson 7.5.

---

## 6. Further reading

- The original *InstructGPT* paper (Ouyang et al., 2022) — the canonical
  SFT recipe for instruction-following.
- HuggingFace `transformers` docs on `CausalLMOutputWithPast` — short,
  worth reading once to confirm how `labels` flows through.
- Top-level [`../04-sft.md`](../04-sft.md) — the wider context (LoRA,
  forgetting, failure modes). Skim it again now that you have the loss
  in your head.
- Next: [`02-loss-masking.md`](./02-loss-masking.md) — the gory details
  of multi-turn masking, `-100`, and the templates that lie to you.

---

**TL;DR:** SFT = next-token cross-entropy with `labels = -100` on
prompt tokens. Same optimizer, same model, same loss as pre-training —
only the mask changes. Everything else in this course is consequences
of that one fact.
