# 11 — Evaluation

"Did post-training actually help?" is harder to answer than it sounds.
Evaluation is its own deep problem.

## Why eval is hard

- Models can score well on benchmarks but feel worse in real use
- Benchmarks **leak** into training data (contamination)
- LLM-as-judge has its own biases (length, style, sycophancy)
- The thing you really want — "is this a good assistant?" — has no clean metric

## Categories of evals

### Capability benchmarks
- **MMLU** — broad knowledge, multiple choice
- **GSM8K** — grade-school math
- **MATH** — competition math
- **HumanEval, MBPP** — code generation
- **GPQA** — graduate-level science
- **BBH** — diverse hard reasoning

### Instruction following
- **IFEval** — verifiable instructions ("respond in exactly 3 sentences")
- **MT-Bench** — multi-turn chat, LLM-judged
- **AlpacaEval 2** — pairwise vs reference model, LLM-judged

### Human preference
- **LMSYS Chatbot Arena** — pairwise vote by real users in the wild
- **Arena-Hard** — automated approximation of Arena

### Reasoning
- **AIME** — competition math
- **MATH-500** — curated MATH subset
- **LiveCodeBench** — contamination-resistant code eval
- **SWE-Bench (Verified)** — real GitHub issues

### Safety / robustness
- **HarmBench** — harm refusal
- **XSTest** — over-refusal
- **JailbreakBench** — adversarial prompts
- **TruthfulQA** — honesty under misleading framing

## Common pitfalls

- **Contamination** — test data in training data, inflates scores
- **Length bias** — judges (human and LLM) prefer longer answers
- **Style bias** — judges prefer confident, formatted answers
- **Single-eval overfitting** — recipes tuned to one benchmark
- **Reward-hacked benchmarks** — high score, real-world worse

## Best practices

- Use **multiple** evals; never rely on one number
- **Hold out** evals from any training data pipeline
- Periodically **rotate** benchmarks (use new contamination-resistant ones)
- Pair automated evals with **human spot-checks**
- Track **regressions** on prior capabilities, not just gains

## The role of LLM-as-judge

Using a strong model to grade other models. Useful but biased:
- Length bias
- Self-preference (judges prefer outputs that look like their own)
- Style over substance

Mitigations: rubrics, pairwise instead of pointwise, swap positions, use
multiple judges, calibrate against humans.

## TRL support (judges)

TRL ships several judge utilities used inside online preference methods
and as evaluation callbacks:

| Judge / Feature                              | Added in   |
|----------------------------------------------|------------|
| WinRate LLM-as-judge callback                | v0.10      |
| Pairwise judges for online methods           | v0.12      |
| Soft judges for PairRM                       | v0.12      |
| `AllTrueJudge` (mixture of judges)           | v0.13      |

## What to learn next

- HELM (Holistic Eval of Language Models)
- "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
- LMSYS leaderboard methodology
- Contamination studies in recent benchmark papers
