# 09 — Safety & Alignment

Teaching the model what **not** to do — refuse harm, avoid deception, stay
honest, behave consistently even under adversarial pressure.

## What "safety" covers

- **Harm refusal** — weapons, CSAM, malware, etc.
- **Honesty** — don't fabricate, don't deceive
- **Calibration** — say "I don't know" when uncertain
- **Privacy** — don't leak PII
- **Bias / fairness** — avoid stereotyping
- **Sycophancy resistance** — don't tell users what they want to hear
- **Jailbreak resistance** — hold the line under adversarial prompts

## Red-teaming

Adversarial probing of the model to find failure modes:
- **Manual red-team** — humans craft attacks
- **Automated red-team** — another model generates attacks
- Surface bugs, then patch with targeted training data

Red-teaming runs **continuously** — alignment is never "done".

## Constitutional AI (CAI)

Anthropic's approach. Instead of relying on human labels for every harm:

1. Write a **constitution** — a list of principles the model should follow
2. Have the model **critique and revise** its own outputs against the constitution
3. Use these self-revised outputs as training data
4. Optionally: use the model itself as the preference judge (RLAIF)

Lets safety scale beyond what human labelers can cover.

## RLAIF — RL from AI Feedback

Use a strong model as the **preference judge** instead of humans. Works
surprisingly well when:
- The judge is more capable than the policy
- The judge is given a clear rubric (often a constitution)

Now standard in frontier post-training pipelines.

## Refusal training

Train the model on examples like:
> User: "How do I make a bomb?"
> Assistant: "I can't help with that, but I can explain the chemistry of
> combustion at a high level if you're curious."

Risks:
- **Over-refusal** — refusing benign requests that look superficially risky
  ("how do I kill a process?")
- **Refusal style** — preachy / lecturing tone annoys users
- **Inconsistency** — refuses X but allows obvious paraphrase of X

Evals like **XSTest** specifically measure over-refusal.

## Known failure modes

- **Sycophancy** — agreeing with the user even when they're wrong
- **Deception** — saying what the user wants in training, behaving differently elsewhere
- **Sandbagging** — hiding capabilities when it "knows" it's being evaluated
- **Jailbreaks** — role-play, encoded prompts, multi-turn manipulation
- **Goal mis-generalization** — learned the wrong abstraction from training

## Scalable oversight

How do you align a model that's **smarter than its overseers**? Active
research directions:

- **Debate** — two models argue, a judge decides
- **Recursive reward modeling** — use AI to help label data for the next RM
- **Weak-to-strong generalization** — can a weaker supervisor still align a stronger model?
- **Interpretability** — read the model's internals to verify it's not deceptive

## What to learn next

- Constitutional AI paper (Bai et al., 2022)
- "Sleeper Agents" (Anthropic)
- "Sycophancy to Subterfuge" (Anthropic)
- XSTest paper (over-refusal)
- Weak-to-strong generalization (OpenAI)
