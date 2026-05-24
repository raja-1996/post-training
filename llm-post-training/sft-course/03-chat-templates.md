# 03 — Chat templates & tokenizer mechanics

> **Goal of this lesson:** know what a chat template *is* (a Jinja
> string on the tokenizer), what the major families look like, how to
> debug them, and the tokenizer-level details (special tokens, BOS,
> embedding resize) that silently destroy fine-tunes when you get them
> wrong.

**Reading:** ~20 min · **Hands-on:** ~30 min · **Prereqs:** lessons
[`01`](./01-what-sft-is.md) and [`02`](./02-loss-masking.md).

---

## 1. Concept — a chat template is a Jinja string

Modern HuggingFace tokenizers ship a `chat_template` attribute: a Jinja2
template that converts a list of `{"role": ..., "content": ...}` dicts
into one flat string of tokens. That string is what the model was
trained on, so it's the **only** correct way to render a conversation
for that model.

```python
tok = AutoTokenizer.from_pretrained("...")
print(tok.chat_template)        # the Jinja source
tok.apply_chat_template(msgs)   # renders it
```

Two crucial features:

- `add_generation_prompt=True` — appends the "now it's the assistant's
  turn" header but *not* an assistant response. This is what you do at
  **inference** to get the model to start generating.
- `tokenize=True/False` — return token IDs vs the rendered string.

If `tok.chat_template is None`, the tokenizer has no opinion and
`apply_chat_template` will error. You either supply a template (set
`tok.chat_template = "..."`) or you're using a base model that was
never trained on chat-formatted text — go back to lesson 04 (choosing a
base model) before fine-tuning.

---

## 2. The major families — what they actually look like

You will see these four formats in 95% of open-weight SFT work.

### ChatML (Qwen, many newer models)

```
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Capital of France?<|im_end|>
<|im_start|>assistant
Paris.<|im_end|>
```

- Special tokens: `<|im_start|>`, `<|im_end|>`.
- Clean, easy to mask: assistant content lives between
  `<|im_start|>assistant\n` and `<|im_end|>`.

### Llama-3 / Llama-3.1 / Llama-3.x

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Paris.<|eot_id|>
```

- Special tokens: `<|begin_of_text|>`, `<|start_header_id|>`,
  `<|end_header_id|>`, `<|eot_id|>`.
- `<|eot_id|>` ends an assistant turn — supervise this token or
  inference will run on. (See lesson 02 pitfalls.)

### Mistral / Mixtral (legacy)

```
<s>[INST] You are helpful.

Capital of France? [/INST] Paris.</s>
```

- Special tokens: `<s>`, `</s>`, `[INST]`, `[/INST]`.
- System prompt is fused into the first user turn — not a separate role.
- Newer Mistral models (Magistral, Mistral Small 3) use a v3/v7 tokenizer
  with proper system support; check `tok.chat_template` to be sure.

### Gemma 2 / Gemma 3

```
<start_of_turn>user
You are helpful. Capital of France?<end_of_turn>
<start_of_turn>model
Paris.<end_of_turn>
```

- Note the role is `"model"`, not `"assistant"`, and Gemma 2 has no
  system role at all (Gemma 3 added one).
- Special tokens: `<start_of_turn>`, `<end_of_turn>`.

### Quick comparison

| Family    | Role names              | Turn separator     | System role? | EOS-of-turn token |
|-----------|--------------------------|--------------------|--------------|-------------------|
| ChatML    | system, user, assistant | `<\|im_end\|>`     | yes          | `<\|im_end\|>`    |
| Llama-3   | system, user, assistant | `<\|eot_id\|>`     | yes          | `<\|eot_id\|>`    |
| Mistral   | (user only; sys inline) | `</s>`             | inline       | `</s>`            |
| Gemma     | user, model             | `<end_of_turn>`    | Gemma 3 only | `<end_of_turn>`   |

The **wrong template at fine-tune or inference time** is one of the top
silent failures in practice. The model will still output something —
often something that *looks* right — but quality is consistently worse,
and you'll spend a week thinking it's a data problem.

---

## 3. The BOS gotcha (one of the top-3 SFT bugs)

Most tokenizers add a Beginning-Of-Sequence token (`<s>`, `<|begin_of_text|>`,
`<bos>`) automatically when you call them on raw text:

```python
tok("hello").input_ids  # [BOS, hello]
```

But `apply_chat_template` **also** typically prepends BOS, because the
template starts with the BOS token literal. If you then pass the
rendered string back through the tokenizer with `add_special_tokens=True`,
you get **two BOS tokens** in a row.

```python
text = tok.apply_chat_template(msgs, tokenize=False)
ids  = tok(text).input_ids        # [BOS, BOS, ...]   ← bug
ids  = tok(text, add_special_tokens=False).input_ids  # [BOS, ...] ← correct
```

Two BOS tokens at the start of every sequence will not break the model
catastrophically — but it does shift the distribution off what the base
model expects, and it costs you a measurable amount of quality.

**Default rule:** when you've already used `apply_chat_template`, set
`add_special_tokens=False` on subsequent tokenizer calls. Or pass
`tokenize=True` to `apply_chat_template` and never re-tokenize.

---

## 4. Special tokens, vocab, and embedding resize

### Adding a new special token

Sometimes you want to add a new token (e.g. `<|tool_call|>` for a
custom tool-calling format, or a control token for structured output).

```python
tok.add_special_tokens({"additional_special_tokens": ["<|tool_call|>"]})
model.resize_token_embeddings(len(tok))
```

Two things must happen, in this order:

1. **Add to the tokenizer.** Now `tok.encode("<|tool_call|>")` returns
   a single ID (the new one) instead of being split into pieces.
2. **Resize the embedding matrix.** The model's input and output
   embeddings must grow to cover the new vocab index, or you get a
   shape mismatch on forward.

The new embedding row is **random-initialized** by default. For a
single SFT run this is usually fine — the model learns it. For high
stakes, initialize it as the mean of related existing embeddings (e.g.
the mean of all tokens in the word "tool" or "call") to give training a
warmer start.

### When you don't need to resize

If the token is already in vocab (many templates reserve "unused" slots
for exactly this), just promote it to special status — no resize.

```python
tok.add_tokens(["<|tool_call|>"], special_tokens=True)
```

### LoRA + embedding resize — a real gotcha

If you resize embeddings *and* train with LoRA on `target_modules` that
*don't* include `embed_tokens` / `lm_head`, the new token's embedding
never gets updated, and the LM head can't predict it either. Two fixes:

- Add `embed_tokens` and `lm_head` to `modules_to_save` in `LoraConfig`
  (full-rank training of those two, no LoRA there).
- Or don't resize: pick an existing unused vocab slot.

Lesson 09 covers `LoraConfig` knobs in detail.

---

## 5. Minimal code — inspect, render, and debug a template

```python
# pip install "transformers>=4.45"
from transformers import AutoTokenizer

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-2-2b-it",
]

msgs = [
    {"role": "system",    "content": "You are concise."},
    {"role": "user",      "content": "Capital of France?"},
    {"role": "assistant", "content": "Paris."},
]

for name in MODELS:
    print(f"\n=== {name} ===")
    try:
        tok = AutoTokenizer.from_pretrained(name)
    except Exception as e:
        print("skip:", e); continue

    # 1. What does the template render?
    print("--- inference render (add_generation_prompt=True) ---")
    print(tok.apply_chat_template(msgs[:-1], tokenize=False,
                                  add_generation_prompt=True))

    print("--- training render (full conversation) ---")
    print(tok.apply_chat_template(msgs, tokenize=False))

    # 2. Which tokens are special?
    print("special tokens:", tok.all_special_tokens)
    print("BOS / EOS / PAD:", tok.bos_token, tok.eos_token, tok.pad_token)

    # 3. BOS-double-up check
    rendered = tok.apply_chat_template(msgs, tokenize=False)
    naive    = tok(rendered).input_ids
    correct  = tok(rendered, add_special_tokens=False).input_ids
    bos = tok.bos_token_id
    print(f"BOS in naive[:3] = {naive[:3]}   "
          f"BOS in correct[:3] = {correct[:3]}")
    if bos is not None and naive[:2] == [bos, bos]:
        print("WARNING: naïve tokenization produces double-BOS")
```

Run this once per model family you intend to use. Save the rendered
output to a file — it's the ground truth for what your training data
must look like after templating.

### Build a brand-new template (when you have to)

Rare, but happens with research base models. Minimal ChatML-style:

```python
tok.chat_template = """{% for m in messages %}<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""
```

Add `<|im_start|>` and `<|im_end|>` to the tokenizer as special tokens
(see §4), resize embeddings, and — critically — make sure the model has
actually been mid-trained or warmed up on text using those tokens before
you call SFT, or you're asking it to learn a new structural format
from scratch with too little data.

---

## 6. Try it yourself (30 min)

1. Pick the model family you'll fine-tune with. Run the inspector above.
   Manually count the tokens in the assistant content vs the template
   scaffolding for one example. Note the ratio — short assistant turns
   on a verbose template waste gradient.
2. **The double-BOS test on your real pipeline.** Tokenize a sample as
   you would in training (via `apply_chat_template` → through whatever
   collator you use). Print the first 5 token IDs. Compare to
   `tok.bos_token_id`. If you see two BOS in a row, find the bug
   *now*.
3. **Inference / training mismatch test.** Render the *prompt half* of
   a conversation with `add_generation_prompt=True`, generate, and
   compare the resulting string structure to your training data. They
   should be byte-identical up to the model's response. Any drift here
   is the most common cause of "fine-tune works in eval, fails in
   serving."
4. Try `tok.apply_chat_template(msgs[:-1], add_generation_prompt=False)`
   — note it does *not* prompt the model to respond. Confirm you know
   when to use which flag.

---

## 7. Pitfalls

- **Wrong template, silently.** You apply a generic
  `"### Instruction:\n... ### Response:\n..."` to a Llama-3 model.
  Loss looks fine. Quality is 20% worse than it should be. **Fix:**
  always use `apply_chat_template` from the model's own tokenizer.
- **Double BOS.** Detailed in §3. The single most-frequent
  template-related bug in TRL pipelines that pre-render text and then
  tokenize.
- **Missing PAD.** Many base/instruct tokenizers have no PAD token.
  Setting `tok.pad_token = tok.eos_token` is the conventional fix, but
  *only if* your data collator masks PAD positions in labels (lesson
  02). Otherwise the model is trained to predict EOS everywhere.
- **`add_generation_prompt` flipped.** Use `True` at inference (you
  want the assistant header so the model continues from there); use
  `False` at training when the assistant turn is already in `messages`.
  Getting this wrong at inference means the model doesn't know it's
  supposed to talk.
- **Embedding resize on a quantized base.** Resizing the embedding
  matrix of a 4-bit base is fiddly; you generally need to either
  resize *before* quantizing, or use `modules_to_save=["embed_tokens","lm_head"]`
  in `LoraConfig` and accept the memory cost. Lesson 09 covers this.
- **Different system-prompt position at train vs inference.** Some
  serving stacks (vLLM, TGI) prepend their own system prompt unless
  told otherwise. Your fine-tune trained without one. Result: train/
  serve distribution mismatch.
- **Trusting a community-uploaded "chat template" file.** Always
  diff the rendered output against the original model card's example.
  Templates get re-written incorrectly more often than you'd hope.
- **Role names don't match.** Passing `{"role": "assistant", ...}` to a
  Gemma model that expects `"model"` will either render wrong content
  or, if the template uses strict Jinja, raise.

---

## 8. Further reading

- HF docs: *Templates for chat models* — the canonical reference for
  `apply_chat_template`, `add_generation_prompt`, and writing your own.
- Each model's card on HF Hub — copy the example chat template
  invocation from there, not from blog posts.
- Top-level [`../04-sft.md`](../04-sft.md), the "Chat templates"
  section — short context for why this matters at all.
- Next: [`04-choosing-a-base-model.md`](./04-choosing-a-base-model.md) —
  base vs instruct, sizes, licenses, and how the tokenizer quality
  affects your fine-tune before training starts.

---

**TL;DR:** A chat template is a Jinja string on the tokenizer; always
render with `apply_chat_template` and never re-tokenize without
`add_special_tokens=False`. Pick your model family, render one example,
inspect it. Mismatched templates and double BOS are silent killers —
both are detected in 30 seconds with the inspector above.
