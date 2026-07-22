# Crytex Kanban Backend Projection

This document defines the P5 Kanban backend contract. Kanban is a projection of
task workflow state, not a UI-only board.

## Canonical Statuses

The product Kanban statuses are:

- `backlog`
- `ready`
- `in_progress`
- `review`
- `remediation`
- `done`
- `failed`
- `blocked`

Legacy persisted statuses remain readable:

- `pending` is projected as `ready`.
- `completed` is projected as `done`.
- `cancelled` is projected as `blocked`.

This keeps old data usable while exposing one stable backend truth to CLI, UI,
diagnostics, and tests.

## Task Card Fields

Every Kanban task projection includes:

- `id`
- `title`
- `goal`
- `agent_role`
- `task_kind`
- `dependency_chain`
- `queue_position`
- `status`
- `critic_comment`
- `remediation_link`
- `trace_id`

Returned or remediation tasks expose critic feedback through `critic_comment`
and the linked remediation task through `remediation_link` when present in task
payload.

## Commands

Production contract:

```powershell
crytex kanban show --json
crytex kanban watch
crytex kanban history --run latest
```

Current development binary:

```powershell
cargo run -p crytex-kernel -- kanban show --project-id <project-id> --json
cargo run -p crytex-kernel -- kanban watch --project-id <project-id> --json --duration-seconds 30
cargo run -p crytex-kernel -- kanban history --project-id <project-id> --run latest --json
cargo run -p crytex-kernel -- prove-kanban-projection --report-path reports\kanban-p5-proof.json
```

When `--project-id` is omitted in the development binary, the latest updated
project is selected.

## Diagnostic Events

Every task status movement emits:

```json
{
  "task_id": "task-id",
  "project_id": "project-id",
  "from": "ready",
  "to": "in_progress",
  "trace_id": "run-id",
  "timestamp": 1710000000000
}
```

The event type is `TaskMoved`. `kanban watch --json` streams these events as
NDJSON.

## SOLID Boundary

Implementation lives in `crates/crytex-core/src/services/kanban_projection.rs`.
The projection depends only on repository traits:

- `ProjectRepository`
- `TaskRepository`

It does not depend on CLI, storage implementation, inference, RAG, agents,
sandbox, or UI. The kernel CLI only resolves project/run selectors and
serializes the projection.

The proof command runs before full `AppContext` initialization and emits a
deterministic JSON artifact proving canonical columns, required card fields,
returned-task remediation metadata, latest-run history, and `TaskMoved`
diagnostic serialization.
