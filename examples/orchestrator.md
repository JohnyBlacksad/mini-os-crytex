# Example: Orchestrator Role

This example shows how the `orchestrator` role decomposes a user goal into a
task graph that Kanban can project as backend truth.

## Goal

Turn a broad product request into ordered tasks with roles, dependencies,
acceptance criteria, and artifact expectations.

## CLI

```powershell
crytex goal submit "Add CSV import with validation, tests, docs, and security review"
crytex plan show --json
crytex plan approve
crytex kanban show --json
crytex kanban watch --json
```

Development proof:

```powershell
cargo run -p crytex-kernel -- prove-kanban-projection --report-path reports\kanban-p5-proof.json
cargo run -p crytex-kernel -- prove-role-quality-contracts --report-path reports\role-quality-p6-proof.json
```

## Role Contract

Role: `orchestrator`.

Expected artifact:

- decomposed task list;
- role assignment;
- task kind;
- dependency chain;
- queue position;
- acceptance criteria;
- handoff artifact contract for every downstream role.

The orchestrator should create clean sessions for each role and pass artifacts,
not chat history. A bad decomposition failure routes to orchestrator quality
improvement, and repeated failures can become role-specific LoRA examples.

## Diagnostics

Inspect plan output, task dependencies, Kanban columns, queue positions, trace
ids, and task movement diagnostics.

## Troubleshooting

- If tasks are too large, check orchestrator benchmark fixtures.
- If dependency order is wrong, inspect Kanban history and dependency graph.
- If a disabled module is required, the plan should show typed degraded status
  instead of failing at runtime.
