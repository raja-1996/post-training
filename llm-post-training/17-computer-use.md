# 17 — Computer-Use Post-Training

Training models to **operate a computer the way a human does** — looking
at the screen, moving the mouse, typing, clicking. The hardest version
of agentic training because the interface is pixels, not APIs.

## What "computer use" means

The agent receives **screenshots** and outputs **actions**:
- `click(x, y)`
- `type("hello")`
- `scroll(direction, amount)`
- `key("cmd+t")`
- `screenshot()`

A full task is dozens to hundreds of these.

Anthropic's "computer use" (Claude 3.5 Sonnet), OpenAI's Operator, and
Google's Project Mariner are the public landmarks.

## Why this is different from tool use

| Tool use            | Computer use               |
|---------------------|----------------------------|
| Structured JSON     | Pixels + coordinates       |
| Discrete actions    | Continuous (x, y) outputs  |
| Clean errors        | Silent failures            |
| Short trajectories  | Very long trajectories     |
| Deterministic       | UI changes, popups, ads    |

The model must **see**, **ground**, and **act** — all three are
post-training problems.

## Training data shapes

- **Screenshot → action SFT** — paired data of what the screen looked
  like and what action a human (or strong model) took next
- **Trajectories with reasoning** — multi-step traces with explicit
  plans between actions
- **Synthetic web/UI tasks** — auto-generated form-filling, navigation,
  shopping, scheduling
- **Replay-style data** — human demonstrations on real apps,
  re-annotated with thoughts

## Grounding: the core skill

Click accuracy is the bottleneck. "Click the **Submit** button" requires:
- **Recognize** the button visually
- **Locate** its bounding box / center
- **Translate** to (x, y) pixels at the current resolution

Approaches:
- Predict raw (x, y) coordinates
- Predict element IDs / DOM references, then resolve
- Predict bounding boxes, then center-click
- Set-of-Marks (numbered overlays) to reduce coordinate regression

## Vision-language post-training

Computer-use is a special case of VLM post-training:
- Standard VLM SFT for general visual understanding
- UI-specific SFT for layouts, icons, widgets
- Resolution and aspect-ratio robustness
- OCR fidelity for text-heavy screens

Many recipes start from a strong VLM, then add UI-specific stages.

## Trajectory-level RL

Same idea as agentic RL but harder:
- **Reward**: task success (form submitted, item purchased, file saved)
- **Sparsity**: success only at the end, after 50+ actions
- **Cost**: real browser/OS rollouts are slow, often serial
- **Safety**: rollouts touch real systems — sandboxing is critical

## Failure modes

- **Mis-clicks** — off-by-a-few-pixels
- **Hallucinated UI** — describes a button that isn't there
- **Loop / stuck** — same action repeatedly when the page didn't update
- **Popup blindness** — ignores modals, dialogs, cookie banners
- **Distractor susceptibility** — ads, related items, autocomplete

Each failure mode maps to a specific training data gap.

## Evaluation

- **OSWorld** — desktop OS tasks
- **WebArena, VisualWebArena** — web navigation
- **Mind2Web** — generalist web agent
- **ScreenSpot** — pure grounding accuracy
- **AndroidWorld** — mobile UI

Grounding accuracy and end-to-end task success diverge — both matter.

## Safety considerations specific to computer use

- The agent can **buy things, send messages, delete files**
- Prompt injection from web content can hijack the agent
- Screenshots may capture **secrets, PII**
- Consent and confirmation step training is its own subproblem

Training must include refusal and confirmation behaviors — not just
task success.

## Open challenges

- Long-horizon credit assignment over 100+ actions
- Real-time UI changes between observation and action
- Cross-platform generalization (web vs desktop vs mobile)
- Cost of running real OS/browser rollouts at training scale
- Adversarial web content (prompt injection via DOM)

## What to learn next

- Anthropic's computer-use launch + research notes
- WebGPT, WebArena, Mind2Web, VisualWebArena
- OS-Atlas, SeeClick, CogAgent, ShowUI
- Set-of-Marks prompting (Yang et al.)
- OSWorld and ScreenSpot benchmarks
