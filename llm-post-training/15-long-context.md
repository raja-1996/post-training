# 15 — Long-Context Post-Training

Pre-training gives a model a context window. Post-training teaches it to
**actually use** that window — to recall, reason over, and stay coherent
across hundreds of thousands of tokens.

## Why this needs its own stage

A model with a 200K window is not automatically good at 200K. Failure modes:
- **Lost-in-the-middle** — strong at the start and end, weak in the middle
- **Distractor sensitivity** — gets pulled off-topic by irrelevant content
- **Stale planning** — forgets its own plan once the buffer fills
- **Degraded reasoning** — accuracy drops sharply beyond the trained length

Long-context post-training fixes the gap between *can attend* and *does
attend usefully*.

## Where long context shows up

- **Document QA** — read a 100-page contract, answer specific questions
- **Codebase tasks** — load a whole repo, edit across many files
- **Agent state** — accumulated tool outputs, prior turns, scratchpads
- **Multi-document synthesis** — research, legal review, due diligence

Modern agents are the dominant consumer — every tool call adds context.

## Training data shapes

- **Needle-in-a-haystack SFT** — plant a fact deep in a long document,
  ask about it. Forces attention to non-positional cues.
- **Multi-needle / multi-hop** — multiple planted facts that must be
  combined. Harder than single-needle.
- **Retrieval-style SFT** — packed documents with answers grounded in
  specific passages. Teaches citation behavior.
- **Long-trajectory agent traces** — full agent runs with all
  observations, training the model to ignore noise and re-read.
- **Long-form generation** — long structured outputs (reports, repos)
  not just long inputs.

## Mid-training vs post-training

The line is fuzzy here:
- **Mid-training**: continued pre-training on long documents (books, code
  repos, concatenated docs) with extended positional encodings.
- **Post-training**: instruction data, preference data, and RL **at long
  length** — teaches the model to follow instructions across the window.

You need both. Long mid-training without long SFT gives you a model that
can attend but won't follow instructions at length.

## Positional / architectural enablers

Post-training rides on architectural choices made earlier:
- **RoPE scaling** (NTK-aware, YaRN, LongRoPE) extends the usable window
- **Sliding-window / sparse attention** for very long contexts
- **Position interpolation** as a cheap extension trick

These are technically pre-/mid-training concerns but constrain what
post-training can achieve.

## Evaluation

- **Needle-in-a-Haystack** — single-fact recall across depth & length
- **RULER** — multi-needle, variable-format synthetic suite
- **LongBench / LongBench-v2** — realistic long-doc tasks
- **InfiniteBench** — extreme lengths (1M+)
- **Agentic long-context** — task success across long tool-use sessions

Recall benchmarks are necessary but not sufficient — passing NIAH does
not mean reasoning over the same content works.

## Open challenges

- Reasoning at length degrades faster than recall
- Long-context RL is expensive — rollouts get slow
- KV-cache cost grows with length, biasing training toward shorter samples
- Evaluations for **agentic** long context are still immature

## What to learn next

- "Lost in the Middle" (Liu et al., 2023)
- LongRoPE, YaRN, NTK-aware scaling
- RULER and Needle-in-a-Haystack
- Anthropic's 200K/1M context work
- Gemini long-context technical reports
