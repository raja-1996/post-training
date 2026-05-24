# 16 — Agentic Coding Post-Training

A specialization of agentic training (file 08) for the dominant workload
of 2025–2026: **models that edit real codebases**. Cursor, Claude Code,
Aider, Devin, Windsurf, Codex — all of them lean on models specifically
post-trained for this loop.

## Why coding deserves its own treatment

Coding is uniquely well-suited to RL:
- **Verifiable** — tests pass or fail
- **Compositional** — many small skills (read, edit, run, debug)
- **Long-horizon** — real tasks span many tool calls
- **Abundant data** — public repos, issues, PRs, commits

But it's also harder than synthetic tool use:
- Edits must be **precise** across multi-file repos
- Build/test environments are slow and flaky
- Partial credit is real — a fix that breaks one test but solves three is non-trivial

## The agentic coding loop

```
Read files  →  Plan  →  Edit  →  Run tests  →  Read errors  →  Edit  →  ...
```

Training data must capture all of it — not just the final diff but the
investigation, the failed attempts, the recovery.

## Training data sources

- **Real PRs / commits** — issue + diff pairs, mined from GitHub
- **Synthetic agent trajectories** — a strong model solves tasks in a
  sandbox; winning trajectories become SFT data
- **Trajectory rewriting** — clean up messy real-world traces into ideal
  ones (remove dead ends, add reflections)
- **Self-generated** — the model solves problems, verifier filters by
  test pass; survivors train the next iteration

General tool-use data pipelines like **APIGen** (verifiable function-call
data) and **Toolformer** (self-supervised tool-call annotation via
loss-reduction) inform the coding-agent variants — see
[13-agentic-tool-use](./13-agentic-tool-use.md) and the survey at
[arxiv 2604.00835](../papers/2604.00835-agentic-tool-use.md).

## SWE-Bench and friends

The benchmark suite that drove the field:
- **SWE-Bench** — resolve real GitHub issues from popular repos
- **SWE-Bench Verified** — human-validated subset
- **SWE-Bench Multimodal** — UI/visual bugs
- **Aider polyglot** — multi-language edit accuracy
- **LiveCodeBench** — contamination-resistant competitive programming
- **Terminal-Bench / TerminalArena** — shell-using agents

These shape what "good at coding" means in training reward design.

## RL recipes for coding

Outcome rewards:
- Did all tests pass?
- Did the agent stop within budget?
- Were the right files modified?

Process / shaping rewards:
- Step-level: did this edit move the test pass-rate up?
- Format: are tool calls valid?
- Cost: how many tokens / tool calls were used?

GRPO and PPO variants both used; outcome reward with **rejection
sampling** is the simplest recipe that works.

## Multi-file editing

The hard part. Models must:
- Build a mental map of unfamiliar code before editing
- Avoid edits that compile but break tests
- Keep edits minimal — sprawl breaks PR review
- Apply consistent style and naming

Trained via:
- SFT on real human PRs (diff-conditioned generation)
- Edit-format losses (search/replace, unified diff, fenced blocks)
- Reward signals tied to diff size, test pass, and reviewer approval

## Tool design as a training choice

The same model behaves very differently with different toolsets:
- **Coarse tools** (`run_bash`, `edit_file`) — flexible but error-prone
- **Fine tools** (`grep`, `read_lines`, `apply_patch`) — easier to verify
- **Agent-native** (`Read`, `Edit`, `Bash`) — Claude Code, Cursor style

Training data must match the tools the deployed agent will see.

## Sandbox & infrastructure

Agentic coding RL is expensive because each rollout runs real code:
- Containerized sandboxes per rollout
- Test runners with strict timeouts
- Caching of build artifacts to amortize cost
- Parallel rollouts at large scale (often thousands concurrently)

Cost per training step is orders of magnitude higher than text-only SFT.

## Open challenges

- **Reward hacking** — model patches the test, not the bug (see file 18)
- **Test sufficiency** — passing tests ≠ correct fix
- **Repo-scale reasoning** — context windows still smaller than big codebases
- **Style and review quality** — passes tests but is unmergeable
- **Generalization** — strong on Python, weaker on niche languages

## What to learn next

- SWE-Bench and SWE-Agent papers
- "SWE-RL" (Meta) — RL on real PRs
- Aider's edit-format research
- DeepSeek-Coder, Qwen-Coder, Codestral training reports
- Anthropic's Claude Code and OpenAI's Codex/Operator engineering posts
