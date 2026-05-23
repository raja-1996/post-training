# 02 — Data: The Real Bottleneck

Post-training results are dominated by **data quality**, not algorithms.
A mediocre algorithm with great data beats a great algorithm with mediocre
data, consistently.

## Two main types of post-training data

### 1. Instruction / demonstration data (for SFT)
Pairs of `(prompt, ideal response)`, often multi-turn conversations.

### 2. Preference data (for RM / DPO / RLHF)
Triples of `(prompt, chosen response, rejected response)` — humans (or AI)
picked which of two responses is better.

## Sources of instruction data

| Source           | Origin                          | Notes                       |
|------------------|----------------------------------|----------------------------|
| **FLAN**         | Academic NLP tasks reformatted   | Broad task coverage         |
| **Alpaca**       | Self-Instruct from GPT-3         | Sparked the synthetic era   |
| **Dolly**        | Databricks employee-written      | Fully human, permissive     |
| **OpenAssistant**| Crowdsourced conversations       | High quality, multi-turn    |
| **ShareGPT**     | Scraped ChatGPT conversations    | Real user distribution      |
| **UltraChat**    | Synthetic multi-turn dialogues   | Large scale                 |
| **Tulu mixes**   | Curated blends (AI2)             | Strong open recipes         |

## Synthetic data generation

Frontier labs increasingly generate their own training data with
**stronger models or self-distillation**.

- **Self-Instruct** — model generates instructions for itself
- **Evol-Instruct** — iteratively make prompts harder
- **Magpie** — extract instructions latent in the base model
- **Persona-driven generation** — diversity via fictional personas
- **Rejection sampling** — generate many, keep only the best

## Quality > quantity

The **LIMA** paper showed that **1,000 carefully curated examples** can
produce a competitive assistant. This is the "superficial alignment hypothesis":
the model already knows things — fine-tuning just teaches it *how to respond*.

Implications:
- A small, diverse, high-quality SFT set often beats a giant noisy one
- Manual curation pays off
- Diversity matters more than raw token count

## Data hygiene

- **Deduplication** — near-duplicates waste compute and hurt eval
- **Decontamination** — strip test-benchmark questions from training data
- **Length balancing** — RMs and DPO are notoriously length-biased
- **Format consistency** — all examples in the same chat template
- **Toxicity / PII filtering** — for safety and legal reasons

## Multi-turn vs single-turn

Real assistants are multi-turn. Training data must reflect this:
- Conversation memory
- Reference resolution ("the last one")
- Tool calls interleaved across turns
- Recovery from user corrections

## What to learn next

- The LIMA paper
- Self-Instruct and Evol-Instruct papers
- Tulu-3 data recipe
- Magpie paper
