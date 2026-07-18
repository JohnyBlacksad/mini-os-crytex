# Agent Swarm, Clean Sessions, LoRA, and Real E2E Proof

Date: 2026-07-16

## Core Model

Crytex must not treat each agent as a separate loaded LLM. The correct model is:

```text
one loaded base model
  -> many short-lived clean agent sessions
  -> each session has its own role system prompt
  -> each session has its own role-selected LoRA adapter
  -> each session receives only explicit handoff artifacts plus fresh RAG context
```

An agent session is a clean inference unit, not a continuation of the previous
agent chat.

## Agent Session Contract

Each agent execution must be created as a new clean session:

```text
AgentSession {
  session_id,
  trace_id,
  task_id,
  agent_role,
  base_model_id,
  system_prompt_id,
  lora_adapter_id,
  upstream_artifact,
  assembled_rag_context,
  output_schema
}
```

The previous agent must not pass its scratchpad, hidden reasoning, full prompt,
or raw message history. Only the explicit artifact is passed forward.

## Artifact Handoff

The workflow chain should look like:

```text
Architect clean session
  -> design_artifact
Coder clean session
  <- design_artifact
  -> patch_artifact
QA clean session
  <- patch_artifact
  -> test_report_artifact
Critic clean session
  <- design_artifact + patch_artifact + test_report_artifact
  -> review_decision
Human review
  -> reward / rejection / remediation
```

The handoff artifact must be structured JSON whenever possible. It is the only
piece of previous-agent context that is automatically carried into the next
session.

## Role LoRA Selection

LoRA adapters are selected by role first:

1. Explicit task override, if present.
2. Role adapter: `architect`, `coder`, `qa`, `security`, `critic`, etc.
3. Role + task kind adapter: for example `coder-rust-bugfix`.
4. Project/domain adapter.
5. No LoRA fallback.

The selected adapter must be written into `InferenceRequest.lora_adapter_id`.
Global LoRA state is unsafe for concurrent swarms unless the backend proves
request isolation.

## Runtime Rules

The runtime must distinguish:

- `request_scoped_adapter`: the adapter is attached for this request only.
- `true_hot_swap`: the base model stays loaded and adapter switch does not reload.
- `reload_swap`: adapter switch invalidates/reloads backend state.
- `unsupported`: backend cannot use LoRA.

Diagnostics must expose the actual mode. We must not call reload-based behavior
"true hot-swap".

## Real E2E Proof Plan

The real proof must run on a concrete coding task, not on a toy prompt.

Example task:

```text
Implement a small but non-trivial Rust feature:
- add a parser or validator
- update an existing service boundary
- add tests
- preserve existing behavior
```

Required runs:

1. Baseline: no evolved prompt, no LoRA.
2. Prompt only: role system prompts active, no LoRA.
3. Prompt + role LoRA: role prompts plus selected role adapters.

Each run must record:

- trace id;
- session id per agent;
- role;
- prompt version id;
- LoRA adapter id;
- adapter mode;
- handoff artifact received;
- generated artifact;
- tests run;
- critic result;
- human/reward result;
- latency and token usage.

## A/B Rules

The benchmark must compare held-out cases that were not used for training.

Guardrails:

- no duplicate benchmark case ids;
- no low-information cases;
- no training/benchmark leakage;
- baseline and challenger run on the same held-out cases;
- challenger is promoted only if benchmark and critic/human signals improve;
- regression or inconclusive result rolls back the LoRA.

Promotion must prove improvement. It is not enough that training finished or
validation loss looked good.

## Current Executable Proofs

Implemented tests now prove:

- workflow agent tasks receive a clean payload with explicit `upstream_artifact`;
- unrelated workflow state does not leak into the agent task payload;
- workflow executor selects LoRA by agent role before calling the agent service;
- LoRA evolution rejects low-information golden examples;
- LoRA evolution rejects oversized adapter artifacts;
- LoRA evolution rejects overfit train/validation loss gaps;
- benchmark harness validates golden sets and detects training leakage;
- Candle trainer writes adapter-only safetensors, not full base model weights.

## Remaining Work

1. Implement concrete `crytex-bench` backed `LoraBenchmarkGate`.
2. Add real corpus fixture and held-out golden set.
3. Add real E2E run: baseline vs prompt vs prompt+LoRA.
4. Add real or optional mistral.rs hot-swap proof with diagnostics.
5. Add diagnostics export for every session boundary and adapter decision.
