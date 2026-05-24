# 02 — Loss masking

> **Goal of this lesson:** master the mechanics of `labels = -100`,
> handle multi-turn conversations correctly, and avoid the silent
> failure modes (off-by-one shifts, leaked prompt loss, wrong-template
> masking) that destroy SFT quality without changing the loss curve in
> any obvious way.

**Reading:** ~20 min · **Hands-on:** ~30 min · **Prereqs:** lesson
[`01-what-sft-is.md`](./01-what-sft-is.md). You should be able to state
the SFT loss equation from memory.

---

## 1. Concept — `-100` is the whole API

In HuggingFace `transformers`, the contract is:

- `input_ids[i]` — what the model *sees* at position `i`.
- `labels[i]`   — what the model is *graded against* at position `i`.
- `labels[i] == -100` → position `i` is ignored in the loss.

That's it. Every masking strategy below is a recipe for filling in the
`labels` tensor.

### The internal shift (read this once, never confuse it again)

Inside the model's `forward`, the loss is computed as:

```python
shift_logits = logits[..., :-1, :].contiguous()   # predicts position 1..T-1
shift_labels = labels[..., 1:].contiguous()       # target at position 1..T-1
loss = CrossEntropyLoss(ignore_index=-100)(
    shift_logits.view(-1, V), shift_labels.view(-1)
)
```

**Consequence:** you align `labels` with `input_ids` (same length, same
positions). The shift happens *inside*. If you pre-shift labels
yourself, the loss is computed against the wrong target — usually
visible as "loss is suspiciously high and never goes down."

A useful check: `len(input_ids) == len(labels)`. Always.

---

## 2. The three masking patterns you actually use

### Pattern A — single-turn (user → assistant)

```
input_ids: [BOS, prompt_tokens..., assistant_tokens..., EOS]
labels:    [-100, -100,  ...,      assistant_tokens..., EOS]
```

The assistant tokens *and the EOS* are supervised. Including EOS is
important: it's how the model learns to stop. Forget it and you get
runaway generations at inference.

### Pattern B — multi-turn, assistant-only loss (the default)

A conversation has multiple `(user, assistant)` turns. You typically
supervise **every assistant turn** and mask every user turn (and every
system message, and all chat-template scaffolding).

```
[system] [user_1] [asst_1] [user_2] [asst_2] [user_3] [asst_3]
 mask     mask     LOSS     mask     LOSS     mask     LOSS
```

This is the right default because every assistant turn is a thing you
want the model to learn to produce. Throwing them away (e.g.
supervising only the final turn) is a common waste of gradient.

### Pattern C — completion-only (prompt → completion, no chat template)

Some tasks aren't chat (classification, extraction, code completion).
The template is just `prompt + response`. Mask the prompt, supervise the
response.

```
input_ids: [prompt..., response..., EOS]
labels:    [-100, ...,  response..., EOS]
```

In TRL, this is what `DataCollatorForCompletionOnlyLM` is for: you give
it a response-delimiter string (e.g. `"### Answer:"`) and it finds the
boundary in each tokenized example.

---

## 3. Multi-turn masking, done right

The naïve approach — "mask everything until the last assistant turn" —
loses gradient on all but the final response. Don't do that.

The correct algorithm:

```
for each assistant turn t in conversation:
    locate the token span [start_t, end_t] in input_ids
    set labels[start_t:end_t] = input_ids[start_t:end_t]
mark everything else as -100
```

The hard part is **locating the spans** after the chat template has
rendered everything into one string. Two robust strategies:

### Strategy 1 — render twice and diff

For each turn, render the conversation **up to and including** that
turn, then **up to but excluding** the assistant content. The difference
in token length is the assistant span.

```python
def assistant_token_mask(tokenizer, messages):
    """Return a 0/1 mask aligned with the tokenized full conversation,
    where 1 = assistant-content position (supervise), 0 = mask out."""
    full = tokenizer.apply_chat_template(messages, tokenize=True)
    mask = [0] * len(full)

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        # Prefix without the assistant content (= up to generation prompt)
        prefix = tokenizer.apply_chat_template(
            messages[:i], tokenize=True, add_generation_prompt=True
        )
        # Prefix WITH the assistant content
        upto = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=True
        )
        for k in range(len(prefix), len(upto)):
            mask[k] = 1
    return full, mask
```

Then `labels[i] = full[i] if mask[i] else -100`.

This works for every chat template that is a prefix-extending render
(ChatML, Llama-3, Qwen, Gemma, most modern ones). It is the
single-most-portable masking recipe.

### Strategy 2 — TRL's built-in `assistant_only_loss`

TRL ≥ 0.12's `SFTTrainer` accepts an `assistant_only_loss=True` flag on
`SFTConfig` that does the above for you, when the tokenizer's chat
template defines `{% generation %}` markers. For Qwen 2.5 / Llama-3.x /
recent Gemma this Just Works; for older templates it silently
falls back to "no masking beyond what the data collator does," which is
not what you want.

**Always print one masked example after construction.** Visual
inspection is the only reliable sign your masking is correct.

---

## 4. Minimal code — assistant-only masking, end-to-end

```python
# pip install "transformers>=4.45" torch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)

messages = [
    {"role": "system",    "content": "You are concise."},
    {"role": "user",      "content": "Capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user",      "content": "Of Germany?"},
    {"role": "assistant", "content": "Berlin."},
]

def assistant_only_labels(tok, messages):
    full = tok.apply_chat_template(messages, tokenize=True)
    labels = [-100] * len(full)
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = tok.apply_chat_template(messages[:i], tokenize=True,
                                         add_generation_prompt=True)
        upto   = tok.apply_chat_template(messages[: i + 1], tokenize=True)
        for k in range(len(prefix), len(upto)):
            labels[k] = full[k]
    return full, labels

input_ids, labels = assistant_only_labels(tok, messages)

# --- Visualize ---
print(f"{'idx':>4} {'tok_id':>7}  {'token':<20s}  label")
for i, (tid, lab) in enumerate(zip(input_ids, labels)):
    marker = "LOSS" if lab != -100 else "----"
    print(f"{i:>4} {tid:>7}  {tok.decode([tid])!r:<20s}  {marker}")

# --- Sanity: forward pass ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
ids = torch.tensor([input_ids], device=model.device)
lab = torch.tensor([labels],    device=model.device)
print("loss:", model(input_ids=ids, labels=lab).loss.item())
```

**What you want to see in the printout:**

- The system message tokens: `----`
- The first user turn: `----`
- "Paris." and its EOS / `<|im_end|>`-style closer: `LOSS`
- The second user turn: `----`
- "Berlin." and its closer: `LOSS`

If *anything* in a user turn shows `LOSS`, or anything in an assistant
content shows `----`, your masking is wrong. Fix it before training.

---

## 5. Try it yourself (30 min)

1. Run the visualizer above on a 3-turn conversation of your own.
   Confirm by eye that every assistant content token is supervised and
   nothing else is.
2. Construct a deliberately-wrong version (mask the assistant, train on
   the user). Run for 50 steps at high LR on a single example. Generate
   from the model — you should get *user-style continuations*. This is
   the unmistakable signature of inverted masking; recognize it once and
   you'll never miss it again.
3. Compute the "supervised ratio" — `(# label != -100) / len(input_ids)` —
   for ~20 examples from your dataset. Plot the histogram. If the median
   is below ~10%, your gradient is dominated by a small slice of each
   sequence and you should consider:
   - Packing short examples together (lesson 07)
   - Filtering out examples with very long prompts and tiny responses
4. Repeat the visualizer with `tok = AutoTokenizer.from_pretrained(
   "meta-llama/Llama-3.1-8B-Instruct")` (if available). Note how the
   chat-template scaffolding tokens differ — and confirm none of *those*
   scaffolding tokens end up supervised either.

---

## 6. Pitfalls

- **Off-by-one shift, applied twice.** You manually shift labels because
  "obviously the model predicts the *next* token." Then the model shifts
  again internally. Loss never converges. **Fix:** never shift labels;
  align them 1-to-1 with `input_ids`.
- **EOS not supervised.** The response ends at content, but the model
  needs to learn to *emit* the EOS / end-of-turn token to stop. If you
  mask EOS, inference runs on. **Fix:** include the closing token in the
  supervised span.
- **Wrong-template masking.** You compute the prompt boundary using one
  template (e.g. `"### Response:"`) but the tokenizer renders a
  different one. Half the prompt slips into the loss. **Fix:** always
  derive the boundary from `apply_chat_template`, not from raw strings.
- **Special tokens supervised by accident.** Some templates wrap the
  assistant content in `<|im_start|>assistant\n ... <|im_end|>`. The
  header tokens are scaffolding; opinions differ on whether to mask
  them. The safe default: supervise content + the terminating token,
  mask the opening header. Pick a policy and stick to it across all
  data.
- **Only supervising the last turn.** In a 5-turn conversation that's
  4 wasted turns of gradient. Use multi-turn masking.
- **Padding tokens counted in the loss.** `tokenizer.pad_token_id`
  positions must be `-100` in labels, otherwise the model learns to
  predict pads. The standard collators handle this; custom collators
  often forget.
- **Mask correct, but loss averaged wrong.** Per-token mean over the
  batch up-weights long examples. For most SFT this is fine; for tasks
  with wildly varying response lengths, consider per-example mean.
- **System prompt supervised.** A surprising number of pipelines train
  on the system message because it's "early in the sequence and surely
  intentional." It's prompt — mask it.

---

## 7. Further reading

- TRL docs: *Train on completions only* — the `SFTConfig` knobs
  (`assistant_only_loss`, `completion_only_loss`) and what they need
  from the chat template.
- HF `transformers`: `LlamaForCausalLM.forward` source — read the
  ~10-line loss block once, then you understand the shift forever.
- Top-level [`../04-sft.md`](../04-sft.md) — the failure-modes section
  is more meaningful now that you've seen masking up close.
- Next: [`03-chat-templates.md`](./03-chat-templates.md) — where the
  prompt/response boundary actually lives, by template family.

---

**TL;DR:** Set `labels[i] = -100` for every token you don't want graded.
Supervise every assistant turn's content + closing token; mask
everything else. Print one masked example before every training run. If
loss won't go down or generations look like user turns, suspect masking
first.
