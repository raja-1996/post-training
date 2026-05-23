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

## The three-paradigm taxonomy (Hu et al., 2026)

Hu et al.'s 2026 survey [arxiv 2604.00835](../papers/2604.00835-agentic-tool-use.md)
organizes the field by **where the optimization signal comes from**:

| Paradigm                        | Signal                          | Weights touched | Core challenge                             |
|---------------------------------|----------------------------------|------------------|---------------------------------------------|
| **I. Prompting (plug-and-play)** | in-context (instructions, demos, observations) | No | brittle on long tasks; high token cost     |
| **II. Supervised tool learning** | labeled/synthetic SFT data       | Yes (SFT)        | building large, high-quality tool datasets |
| **III. Reward-driven policy**    | environmental reward / RL        | Yes (RL)         | credit assignment across long tool chains  |

Evaluation is a fourth, orthogonal dimension (see §Eval below). The three
paradigms are **complementary** — production systems combine all three.

## Paradigm I: Prompting plug-and-play

Frozen LLM coordinates tools via prompts and observation feedback. Three
sub-styles:

- **Interleaved reasoning + action** — ReAct (Yao et al., 2022), Reflexion,
  Self-Refine, Chain-of-Verification, CRITIC, LATS (ReAct + tree search)
- **Decoupled plan + execute** — ReWOO (planner/worker), LLMCompiler
  (dependency-aware parallel calls), AdaPlanner, Plan-and-Solve,
  HuggingGPT, TaskMatrix.AI
- **Program-aided reasoning** — PAL, PoT, Logic-LM, ViperGPT,
  Code-as-Policies, MathPrompter, LATM ("LLMs as tool makers"), CREATOR

Cheap, no fine-tuning. Brittle on multi-step or long-horizon tasks.

## Function-calling training (Paradigm II)

The minimum viable version:
- Define tools with name, description, and JSON schema
- Generate / collect training data where the assistant **outputs a tool call**
  in the expected format
- SFT on those traces

This is now table-stakes for any "instruct" model.

### Synthetic data pipelines for tool training

Most open tool-trained models depend on one of these data sources:

| Pipeline | Key idea |
|---|---|
| **Toolformer** (Schick et al., 2023) | Self-supervised: keep a tool call iff inserting its result *reduces* next-token loss |
| **ToolAlpaca** | Teacher-LLM-generated synthetic demonstrations |
| **ToolLLM / ToolBench** | Instruction tuning over 16,000+ real REST APIs |
| **APIGen** | Multi-stage pipeline producing **executable, verified** function-call data |
| **ToolACE** | Self-evolving API augmentation + dialogue synthesis + hierarchical validation |
| **AutoAct** | Auto-synthesized planning trajectories for tool-augmented QA |
| **Confucius** | Introspective-feedback synthesis with easy-to-difficult curriculum |
| **Agent-FLAN** | Decomposes agent capability into format-following + reasoning, uses negatives to suppress hallucinated actions |
| **AgentTuning** | Mixes agent supervision with general data to preserve generalist competence |

Trend across these: trajectory generation → verifiable function-call quality
→ curriculum / introspection → fully self-evolving agents.

### Tool-call formats and schemas

- **Schema-constrained function calling** — JSON-with-signature; the canonical
  evaluation is **BFCL** (Berkeley Function-Calling Leaderboard)
- **REST APIs** — ToolLLM/ToolBench's primary format
- **Code as action space** — PAL/PoT/CodeAct treat Python as the universal tool
- **Token-level tool embeddings** — ToolkenGPT represents tools as vocabulary
  tokens instead of textual schemas (avoids context blow-up)
- **Compressed docs** — EasyTool transforms verbose tool documentation into
  token-efficient form
- **MCP (Model Context Protocol)** — the emerging cross-vendor standard for
  agent ↔ tool integration; reduces N×M custom integrations to N+M

### Process-oriented and alignment SFT

Internalize the *reasoning process*, not just call correctness:

- **FireAct** — SFT on full ReAct-style trajectories rather than IO pairs
- **ToRA** — interleaved rationale + tool-call training for math
- **Masked Thought** — recover masked spans of reasoning chains; reduces
  surface-pattern matching
- **RAFT** — train to distinguish relevant from distractor retrieved docs
- **MetaTool** — train models to decide *whether* a tool is needed at all
- **ToolAlign** — H2A principle (helpfulness, harmlessness, autonomy)
- **ToolSword** — adversarial scenarios (prompt injection)
- **RoT** — robustness under noisy or imperfect tool documentation
- **Lemur / AgentTuning** — preserve generalist competence alongside
  agentic SFT

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

## Paradigm III: Reward-driven tool policy learning

Treat the **whole trajectory** (multiple tool calls, observations, final
answer) as one RL episode.

- Reward at the end (did the agent solve the task?)
- Credit assignment over many tokens and turns
- Often uses GRPO or PPO variants

### Reward designs that have worked

| Reward shape | Representative method |
|---|---|
| Outcome-only (correct answer) | **ToolRL** ("Reward is all tool learning needs"); **Search-R1** |
| Outcome + selectivity penalty | **ReTool** — reward correctness *and* discourage unnecessary tool calls |
| Process reward modeling | **AgentPRM** — score promise/progress of intermediate steps |
| Process + outcome blend | **LeTS** — joint reasoning + retrieval |
| Trajectory-level reward infra | **RLFactory**, **VerlTool** |
| Specialized tool-outcome RM | **ToolRM** — reward models for tool-call efficiency/correctness |
| Curriculum / hard-sampling | **ToolExpander**, **Agent0** (ADPO) — auto-curriculum |
| Role-distributed rewards | **MATPO** — multi-agent tool-integrated PO with planner/worker roles |
| Reflection-as-action | **Agent-R** — make critique an explicit RL action |

### End-to-end multi-turn policy learning

- **Search-R1** — RL induces query reformulation and synthesis from sparse
  outcome rewards (search agent)
- **SimpleTIR** — unified end-to-end RL loop over reasoning + tool execution
- **Retroformer / TRICE** — retrospective learning and credit assignment
  across long tool chains
- **RAGEN** — self-evolving from reflective revision of past trajectories

### Holistic agentic frameworks

- **VerlTool** — agent system (memory + tool selector + LLM) updated as a
  unified policy
- **DeepAgent** — autonomous memory folding compresses past interactions
  into structured episodic memory
- **AgentRL** — async generation-training pipeline, unified function-call
  API, centralized environment controller for scalable multi-task training
- **Agent Q** — guided MCTS + self-critique; learns from successful *and*
  failed trajectories via self-play
- **GEPO** — builds a state-transition graph from experience, uses graph
  centrality to guide exploration

Open challenges:
- Sparse rewards over very long horizons
- Sandboxing tools safely during training rollouts
- Cost — agentic rollouts are slow

## Credit assignment in agentic RL

The defining technical challenge of multi-turn agents. With $T\sim 100$
turns and only a terminal reward, GRPO's episode-level credit produces
the **"echo trap"** — agents converge to repetitive safe behavior because
the gradient signal cannot distinguish productive from redundant actions.
Zhang's 2026 survey [arxiv 2604.09459](../papers/2604.09459-credit-assignment-rl-llms.md)
argues this isn't a quantitative extension of reasoning-RL: stochastic
environments, partial observability, non-verifiable intermediate states,
and heterogeneous action types genuinely *reshape* the CA problem.

### Turn-level PRMs
- **AgentPRM** — replaces MC step labelling with **TD+GAE** at turn
  granularity; 8× more sample-efficient than MC-based PRM training; +19pp
  over ORM on WebShop.
- **SWEET-RL** (Meta/FAIR) — **privileged (asymmetric) critic** that sees
  ground-truth answers and full future trajectories at training time only;
  enables turn-level rewards on non-verifiable tasks like collaborative
  coding (ColBench).
- **Turn-PPO** — reformulates multi-turn RL as a turn-level MDP; each turn
  is a macro-action with its own value function and importance ratio.
- **SORL** — turn-level importance sampling + clipping-triggered
  normalization for long-horizon off-policy stability.
- **TARL** — LLM-as-judge turn-level adjudication; +6pp on τ-bench.
- **ITPO** — implicit turn-level rewards from log-prob ratios
  (no separate RM).

### Hindsight / counterfactual (the distinctively agentic family)

Three independent papers in **one week** of March 2026 — flagged in the
survey as a community convergence:

- **HCAPO** — post-trajectory hindsight: LLM critic generates
  counterfactual continuations per turn and compares expected outcomes,
  entirely in "imagination" (no environment re-execution).
- **C3** — leave-one-out credit: $c_t = R(\tau) - R(\tau_{\setminus t})$;
  also extends to multi-agent ($c_k$ across agents).
- **CCPO** — structural causal model view; each turn is a treatment,
  credit = average treatment effect via do-calculus.
- **CriticSearch** — hindsight specifically for search agents with a
  frozen asymmetric critic using gold answers + full trajectory.

### Critic-free step-level
- **GiGPO** (Group-in-Group PO) — nests step-level groups within
  trajectory-level groups by anchor-state matching; critic-free.
  **+12.6pp on ALFWorld, +9.1pp on WebShop vs GRPO.**
- **POAD** (Policy Optimization with Action Decomposition) — Bellman with
  Action Decomposition integrates intra-action (token) + inter-action
  (turn) credit.
- **CARL** — identifies bifurcation points by **action entropy**;
  optimizes only highest-entropy actions. **72% fewer gradient updates,
  no performance loss.**
- **iStar** — implicit step rewards from trajectory-level DPO + multi-level
  fusion (turn + token).
- **IGPO** — turn credit as **information gain** about task success:
  $c_t = \log P(\text{success}\mid h_{1:t}) - \log P(\text{success}\mid h_{1:t-1})$.

### Hierarchical
- **ArCHer** — pioneer two-level architecture: off-policy TD critic over
  turns + on-policy actor over tokens-within-a-turn.
- **PilotRL** — progressive coarse-to-fine: plan-level → step-level →
  token-level RL stages.
- **StepAgent** — IRL on expert demos to infer step rewards; novice-to-
  expert curriculum.

### Infrastructure
- **Agent Lightning** (Microsoft) — LightningRL algorithm + execution/
  training decoupling; drop-in for LangChain/AutoGen.
- **SPA-RL** — lightweight MLP "progress estimator"; RUDDER-style
  return decomposition.
- **RAGEN/StarPO** — uncertainty-based filtering to escape the echo trap.

### Multi-agent CA
- **M-GRPO** — two-level: inter-agent (compare team compositions) +
  intra-agent (standard GRPO).
- **SHARP** — Shapley-based marginal credit across agents + global +
  tool-process reward. **+23.7% over single-agent baselines.**
- **MAPPA** — per-action AI-judge process rewards; **+16.7pp on data
  analysis tasks.**
- **Dr. MAS** — agent-wise advantage normalization (fixes the gradient-spike
  divergence of naïve multi-agent GRPO).
- **LLM-MCA** / **QLLM** — LLM-based centralized critics; QLLM has the
  LLM *generate Python code* that computes credit (training-free).

### Practical selection (from the survey's decision tree)

| Setting | Suggested methods |
|---|---|
| Short reasoning (≤5K tokens) | GRPO, PURE, SPO, SPRO |
| Long reasoning (>5K), limited compute | HICRA, CAPO, SPRO |
| Long reasoning, generous compute | VinePPO, SCAR, CAPO |
| Agentic, ≤30 turns, no aux model | GiGPO, CARL, iStar, POAD |
| Agentic, ≤30 turns, with aux | AgentPRM, SWEET-RL |
| Agentic, >30 turns, limited compute | CARL, HCAPO, ArCHer |
| Agentic, >30 turns, generous compute | C3/CCPO, HCAPO, IGPO |
| Multi-agent | M-GRPO, SHARP, MAPPA, Dr. MAS |

## Multi-environment trajectory RL

Rather than one task at a time, train RL **across many tool-using
environments in parallel**. NVIDIA's NeMo-Gym (used for Nemotron 3) is
one published instance: 21 environment configurations, ~1.2M rollouts,
trajectory-based reinforcement. The hypothesis is that diversity of
environments produces more robust, transferable agent behavior than
deeper single-environment RL.

## Domains where agentic training matters

- **Code agents** — SWE-Bench, Aider, Devin-style systems
- **Web agents** — browser navigation, form filling
- **Computer use** — screenshot → action loops
- **Research agents** — iterative search and synthesis
- **Tool-using assistants** — calendars, email, APIs

## Evaluation (three layers)

Hu et al.'s survey organizes agent benchmarks into three layers:

1. **Tool-usage correctness** — does the model produce a valid, well-typed
   tool call?
   - **BFCL** (function-call validity), **ToolEyes**, **T-Eval** (step-wise)
   - **API-Bank**, **StableToolBench**, **Gorilla** (selection/retrieval)
   - **When2Call**, **WTU-Eval** (decide *whether* to invoke)
   - **NESTFUL**, **NexusRaven** (nested/compositional calls)
2. **Task completion** — math/QA (HotpotQA, ToolQA, FreshQA), programming
   (HumanEval, MBPP, SWE-Bench, BigCodeBench), scientific/professional
   (SciBench, MedAgent-Bench, ChemCrow)
3. **Tool-driven interaction**:
   - **Web**: WebArena, VisualWebArena, WebVoyager, GAIA, Mind2Web,
     TravelPlanner
   - **Computer/OS/mobile**: OSWorld, AndroidWorld, AppWorld, OSCopilot,
     **τ-bench** (tool-agent-user with policy compliance),
     TheAgentCompany, WorkArena, OfficeBench
   - **Safety/reliability**: ToolEmu (sandboxed risky-action emulation),
     ToolSword, InjEcT-Agent (indirect prompt injection), R-Judge

## Future directions (from the survey)

1. **MCP / standardization** — universal protocol so each tool integrates
   once, agents reuse it everywhere
2. **Agentic foundation models** — Vision-Language-Action models pretrained
   for action (Magma, ShowUI, OmniParser, Agent-S) replacing patched LLM+UI
3. **Continuous evolution** — Titans-style test-time memory, evolving tool
   libraries (Trove), self-curated training (SEAgent, Agent0)
4. **Safety against indirect prompt injection** — programmable guardrails
   (NeMo Guardrails), adversarial input detection (PromptArmor),
   sandboxed execution
5. **Human–agent symbiosis** — multi-agent orchestration (MetaGPT, AutoGen,
   OpenDevin); counter-trend: **Agentless** shows simple pipelines can beat
   complex long-horizon agents when task structure is clear

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
