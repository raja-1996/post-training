# 08 — Agentic & Tool-Use Training

Making models that can **act**, not just chat — call functions, search the
web, run code, control a browser, navigate a filesystem.

## Why this needs its own post-training

Tool use looks like text but has hidden structure:
- Strict JSON / schema formats
- Multi-turn loops (call → observe → think → call again)
- Error recovery (the tool returned nothing useful)
- Long horizons (dozens of steps to finish a task)

A model that's been SFT'd only on chitchat will not robustly use tools.

## Function-calling training

The minimum viable version:
- Define tools with name, description, and JSON schema
- Generate / collect training data where the assistant **outputs a tool call**
  in the expected format
- SFT on those traces

This is now table-stakes for any "instruct" model.

## ReAct-style traces

```
Thought:  I need to look up X.
Action:   search("X")
Observation: ...result...
Thought:  Now I know X. I should compute Y.
Action:   python("...")
Observation: ...result...
Thought:  Answer is Z.
Final:    Z
```

Training data follows this interleaved pattern. Strong models learn to
**plan, observe, replan**.

## Multi-turn tool use

Real agents don't get one shot. They:
- Maintain state across turns
- Re-read documents they fetched earlier
- Recover when a tool fails
- Decide when to stop

Training data must reflect long, messy trajectories — not just clean
single-shot calls.

## Trajectory-level RL

The natural next step: treat the **whole trajectory** (multiple tool calls,
observations, final answer) as one episode and apply RL to it.

- Reward at the end (did the agent solve the task?)
- Credit assignment over many tokens and turns
- Often uses GRPO or PPO variants

Open challenges:
- Sparse rewards over very long horizons
- Sandboxing tools safely during training rollouts
- Cost — agentic rollouts are slow

## Domains where agentic training matters

- **Code agents** — SWE-Bench, Aider, Devin-style systems
- **Web agents** — browser navigation, form filling
- **Computer use** — screenshot → action loops
- **Research agents** — iterative search and synthesis
- **Tool-using assistants** — calendars, email, APIs

## TRL support

| Feature                                          | Added in   |
|--------------------------------------------------|------------|
| **Text Environments** (early agent infra)        | v0.7       |
| Tool support in data preprocessing utilities     | v0.13      |
| Tools in `SFTTrainer` (JSON schema)              | v0.19      |
| GRPO agent training with tools                   | v0.26      |
| Tool-calling iteration limits in GRPO            | v0.27      |
| Multimodal tool responses (VLMs)                 | v1.1       |
| Tool-calling for LLaMA 3.1/3.2, DeepSeek-V3      | v1.2       |
| Tool-calling for GPT-OSS, GLM-4-MoE, Qwen3-VL    | v1.1       |
| Tool calls in `VLLMClient.chat()`                | v1.0       |
| OpenEnv / OpenReward integrations                | v0.25 / v1.4|

## What to learn next

- ReAct paper (Yao et al., 2022)
- Toolformer and Gorilla
- SWE-Bench and Aider
- Anthropic's computer-use research
- "WebGPT", "WebArena", "Mind2Web"
