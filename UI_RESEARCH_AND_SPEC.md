# Crytex Tauri UI Research and MVP Spec

> Historical note: this file contains early UI research notes. The canonical product UI specification is now `UI_PRODUCT_REQUIREMENTS.md`.

Date: 2026-07-04

## Status

Detailed implementation requirements now live in `UI_REQUIREMENTS.md`. Treat this file as the research and direction brief, and treat `UI_REQUIREMENTS.md` as the build contract for the first Tauri UI.

Lazyweb MCP and skill pack are installed, but the current Codex session has not reloaded the new MCP server yet. The first proper Lazyweb pass after reload should be:

1. `lazyweb_search` for `agent workflow dashboard`, platform `desktop`.
2. `lazyweb_search` for `developer task board`, platform `desktop`.
3. `lazyweb_generate_report` with objective `create`, product context from this document, and a screenshot once the first UI shell exists.

Until then, this spec uses current product references and the existing Crytex architecture/code.

## Product Definition

Crytex is a local agentic development OS: it creates projects, decomposes work into tasks, routes tasks through specialized agents, tracks execution and review state, stores memory/artifacts, and exposes system health.

The UI must make the invisible agent system inspectable and controllable. The first desktop UI is not a landing page, not a chat-only app, and not a generic admin dashboard. It is an operator console for running and debugging local agent workflows.

## Reference Direction

### Linear

Borrow:

- Dense issue list and board ergonomics.
- Fast project/task switching.
- Status-first workflow language.
- Minimal task cards with enough metadata to scan.

Avoid:

- Pure issue tracker scope. Crytex tasks also have agent traces, artifacts, prompts, and execution logs.

### LangGraph Studio

Borrow:

- Graph/debugging mental model for agent workflows.
- Inspectable runs, state, and transitions.
- A separate graph/trace view for understanding why a task is blocked or ready.

Avoid:

- Making graph editing the first screen. For Crytex MVP, graph inspection is secondary to task execution.

### Retool / Internal Ops Dashboards

Borrow:

- Dense workbench layout.
- Stable sidebars, tables, filters, and detail panels.
- Clear operational states and fast repeated actions.

Avoid:

- Card-heavy marketing dashboards and decorative hero sections.

### Dify / Flowise

Borrow:

- Workflow node vocabulary for future orchestration visualization.
- Human-in-the-loop checkpoints and run histories.

Avoid:

- Visual builder-first experience in the MVP. Crytex already has a backend orchestration model; manual workflow authoring comes later.

## MVP User Goals

The first UI must let the user manually test the backend in one loop:

1. Create or select a project.
2. Create a task with kind, priority, assigned agent, and payload.
3. See task distribution by status.
4. Move task status through valid transitions.
5. See whether a task is ready or blocked.
6. Inspect a task's details: payload, result, trace id, scores, timestamps.
7. See project state, latest audit/log information, and metrics snapshot.
8. Refresh state without restarting the app.

Non-goals for the first UI:

- Full code editor.
- Visual workflow builder.
- Model download manager.
- Drag-and-drop between every status.
- Live streaming event channel UI if the Tauri runtime wrapper is not ready yet.
- Authentication, teams, billing, cloud sync.

## Information Architecture

Use a single workbench screen with persistent regions:

1. Left sidebar: projects and global navigation.
2. Main area: active project board/list.
3. Right inspector: selected task/project details.
4. Bottom drawer or right-tab panel: logs, events, raw JSON, metrics.

Primary navigation:

- `Projects`
- `Tasks`
- `Graph`
- `Runs`
- `Memory`
- `Models`
- `Settings`

MVP implements `Projects`, `Tasks`, and a lightweight `System` panel inside the same screen. Other sections can be visible but disabled only if that improves orientation; avoid dead navigation if it feels fake.

## First Screen Layout

### Header

Purpose: current context and global actions.

Contents:

- Current project name and root path.
- Environment badge: `local`.
- Refresh button.
- New task button.
- New project button.

### Left Sidebar

Purpose: project selection.

Contents:

- Project list.
- Project search/filter.
- Compact project metadata: path, task count if available.

### Board Area

Purpose: status overview and manual workflow testing.

Columns:

- `Backlog`
- `Pending`
- `In Progress`
- `Review`
- `Completed`
- `Failed`
- `Cancelled`

Card fields:

- Title.
- Kind.
- Assigned agent.
- Priority.
- Status.
- Readiness marker: `ready`, `blocked`, or `unknown`.
- Trace id short suffix.
- Timestamps where useful.

MVP can use click actions instead of drag-and-drop:

- Select card.
- Move status via segmented control or status menu in inspector.
- Refresh board.

### Inspector

Purpose: answer "what is this task and why is it here?"

Task tabs:

- `Overview`: title, status, kind, agent, priority, parent, trace id.
- `Payload`: formatted JSON.
- `Result`: formatted JSON or empty state.
- `Scores`: critic score, human score, priority score.
- `Timing`: created, started, finished, iteration count.
- `Debug`: raw task JSON.

Project tabs:

- `State`: project metadata and latest state export.
- `Metrics`: CPU/RAM/cache/task counters from `MetricsSnapshot`.
- `Logs`: audit logs when available.

### Bottom/Side Debug Panel

Purpose: backend trust and troubleshooting.

MVP:

- Last command result.
- Last error.
- Raw JSON toggle for selected object.
- Manual refresh timestamp.

Later:

- Live event stream from `subscribe_to_events`.
- Worker logs.
- Sandbox output.

## UI Behavior

### Empty State

If there are no projects:

- Show project creation form directly in the main area.
- Required fields: name, root path.
- Root path should be a text input first; native folder picker can come later through Tauri dialog plugin.

If a project has no tasks:

- Show task creation form and explain nothing in-app beyond labels/placeholders.

### Task Creation

Fields:

- Title.
- Description.
- Kind: `codegen`, `research`, `summarization`, `qa`, `security`, `review`, `generic`.
- Assigned agent: `coder`, `architect`, `researcher`, `qa`, `security`, `critic`, or empty.
- Priority: numeric stepper/input.
- Payload JSON editor textarea.

Validation:

- Title cannot be empty.
- Kind cannot be empty.
- Payload must parse as JSON; default `{}`.

### Status Change

Use status menu or segmented control in inspector. Backend remains source of truth; invalid transitions must show the returned error.

Do not encode all transition rules only in frontend. Frontend may disable obvious impossible actions, but backend validation is authoritative.

## Tauri / Backend Contract

Already available in `crates/crytex-tauri/src/commands.rs`:

- `list_projects`
- `create_project`
- `kanban_state`
- `list_tasks`
- `submit_task`
- `set_task_status`
- `get_project_state`
- `subscribe_to_events` as plain async function scaffold

Needed before or during first real Tauri shell:

- Runtime wrapper with `#[tauri::command]` functions that call the existing plain async command functions.
- Managed application state that stores `Arc<dyn ProjectService>`, `Arc<dyn TaskService>`, audit, snapshots, metrics, and events.
- Serializable error conversion for frontend display.
- A simple in-memory or SQLite-backed bootstrap path for local manual testing.

Potential next IPC commands:

- `list_ready_tasks(project_id?)`
- `get_task(task_id)`
- `get_metrics_snapshot`
- `list_audit_logs(project_id, task_id?)`
- `cancel_task(task_id)`
- `retry_task(task_id, feedback?)`

## Frontend Stack

Recommended:

- Tauri 2.
- Vite.
- React + TypeScript.
- CSS modules or plain CSS first; do not introduce a large UI framework until patterns stabilize.
- Lucide icons for buttons.
- TanStack Query or a small local query wrapper only if state refresh starts duplicating.

Avoid in MVP:

- Heavy charting until metrics are actually useful.
- Complex drag-and-drop library before status changes are stable.
- Monaco/CodeMirror until artifacts/code editing are in scope.
- Nested cards and decorative dashboard chrome.

## Visual Direction

The app should feel like a compact developer operations console:

- Neutral background.
- High contrast text.
- Subtle borders.
- Status colors used sparingly.
- Dense but readable spacing.
- Cards only for task items and modal/panel content.
- No landing page, hero section, decorative gradients, or oversized marketing typography.

Use stable dimensions:

- Sidebar width: 260-300px.
- Inspector width: 360-440px.
- Board columns: fixed min width with horizontal scroll.
- Task cards: consistent height bands, no layout jumps on hover.

## Implementation Phases

### Phase 1: Shell and Read Model

- Create Tauri/Vite shell.
- Show static workbench layout.
- Wire `list_projects`, `list_tasks`, `kanban_state`.
- Add selected project/task state.
- Show raw JSON inspector.

### Phase 2: Write Actions

- Wire `create_project`.
- Wire `submit_task`.
- Wire `set_task_status`.
- Add error display and loading states.

### Phase 3: Debuggability

- Wire `get_project_state`.
- Show metrics snapshot.
- Show audit/log panel if available.
- Add readiness/blocked indicators.

### Phase 4: Live Updates

- Wrap `subscribe_to_events` with Tauri Channel.
- Append event feed.
- Auto-refresh affected project/task state.

### Phase 5: Workflow/Graph

- Add graph view using React Flow.
- Show dependencies and status coloring.
- Show selected edge/dependency metadata.

## Acceptance Criteria for First Manual UI

The user can:

- Start the desktop app.
- Create/select a project.
- Create a task.
- See the task in a status column.
- Select the task and inspect raw/details.
- Move the task status and see the board update.
- Refresh project state.
- See backend errors without the UI crashing.

Technical checks:

- `npm run dev` starts frontend.
- `cargo tauri dev` or equivalent starts the desktop shell.
- Rust commands are unit tested at the command wrapper boundary.
- TypeScript build passes.
- UI works at 1366x768 and 1920x1080 without overlapping text.

## Open Questions

- Should first persistence bootstrap use real `Storage` or in-memory repositories seeded from UI?
- Should project root creation require existing path only, or allow future path?
- Should status changes be click-only first, or include drag-and-drop immediately?
- Which task kinds are truly supported end-to-end today versus only modeled?
- Do we want the first UI to expose raw JSON payload editing, or a safer structured form per task kind?

## External Sources to Re-check With Lazyweb

- Linear issue/board workflows.
- LangGraph Studio graph/run debugging.
- Dify workflow and run inspection.
- Retool internal dashboard density.
- Tauri 2 command/state docs.
