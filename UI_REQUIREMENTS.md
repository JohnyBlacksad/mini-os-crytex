# Crytex Tauri UI Requirements

> Historical note: this file describes the temporary low-level Tauri IPC test bench. The canonical product UI specification is now `UI_PRODUCT_REQUIREMENTS.md`.

Date: 2026-07-04

## Purpose

Crytex UI is a local operator console for an agentic development OS. Its job is to make projects, tasks, agents, memory, runs, logs, metrics, and failures visible enough that a human can test and debug the core by hand.

The first version must not be a landing page, marketing dashboard, or chat-only app. It must be a dense dark desktop workbench for controlling the existing Rust core through Tauri IPC.

## Current Backend Contract

The first UI must be built around commands that already exist in `crates/crytex-tauri/src/commands.rs`:

- `list_projects`
- `create_project`
- `kanban_state`
- `list_tasks`
- `submit_task`
- `set_task_status`
- `get_project_state`
- `subscribe_to_events` as a plain async scaffold, not yet a real frontend stream wrapper

The UI may show future tabs only when they are clearly marked as unavailable or read-only. It must not pretend that missing backend functions are implemented.

Needed soon:

- `get_task(task_id)`
- `list_ready_tasks(project_id?)`
- `get_metrics_snapshot`
- `list_audit_logs(project_id, task_id?)`
- `cancel_task(task_id)`
- `retry_task(task_id, feedback?)`
- Tauri `#[tauri::command]` wrappers around the current plain async functions
- Tauri event/channel wrapper around `subscribe_to_events`

## Visual Direction

The default theme is dark and technical:

- Background: near-black graphite, not pure black.
- Surfaces: slightly raised dark panels with thin borders.
- Text: high contrast for primary data, muted gray for metadata.
- Accent: cold cyan/blue for primary actions, amber for pending/review, green for success, red for failures.
- Typography: compact UI font for labels, monospace for IDs, logs, JSON, prompts, and traces.
- Density: operational and scannable, closer to Linear, OpenAI logs, Better Stack, Port, and n8n than to a marketing SaaS homepage.

Avoid:

- Hero sections.
- Decorative gradients or blobs.
- Oversized cards.
- Nested cards.
- Large empty dashboard chrome.
- Fake analytics charts before the backend exposes useful metrics.

## App Shell

The app uses one persistent desktop workbench.

Layout:

- Left rail: global navigation icons.
- Project sidebar: project list, search, create project.
- Header: current project, health/status badges, refresh, global create task.
- Main workspace: selected top-level section.
- Right inspector: selected task/run/project details.
- Bottom observability drawer: logs, events, traces, raw command output.

Recommended dimensions:

- Global nav rail: 56 px.
- Project sidebar: 280 px.
- Header: 52-64 px.
- Inspector: 400-460 px.
- Bottom drawer: 260-420 px, resizable later.
- Board columns: min 260 px each, horizontal scroll.

The layout must work at 1366x768 and 1920x1080 without text overlap.

## Global Navigation

### Projects

Default entry point. Shows available projects and the active project workbench.

Enabled in MVP.

### Tasks

Kanban board and task list for the selected project.

Enabled in MVP.

### Runs

Execution/run history grouped by task, trace id, agent, model, and status.

MVP behavior: visible as a tab inside Observability or disabled global nav item until run history IPC exists.

### Observe

Logs, events, traces, metrics, raw JSON, errors.

High priority. MVP must include the panel even if some feeds are initially populated from `get_project_state` and command responses rather than live streaming.

### Memory

Project memory, snapshots, RAG chunks, semantic index state.

MVP behavior: read-only placeholder or a project state tab if `ProjectState` already exposes snapshots. Full search/edit comes later.

### Models

Inference backend, model registry, LoRA adapter status, embedding/rerank backend.

MVP behavior: disabled or read-only "not wired" section unless current config APIs are exposed.

### Graph

Task dependency graph and future workflow visualization.

MVP behavior: optional read-only graph from task dependencies if data is available. Do not make it the first screen.

### Settings

Local storage path, theme, refresh interval, debug toggles.

MVP behavior: local frontend settings only, no unsupported backend mutation.

## Header

The header is always visible.

Left side:

- Project name.
- Root path as muted secondary text.
- Environment badge: `local`.
- Storage badge: `memory`, `sqlite`, or `unknown`.

Center:

- Active section title.
- Optional quick filters for the current section.

Right side:

- Refresh button.
- New task button.
- New project button.
- Observe drawer toggle.

Refresh button:

- Calls `list_projects`.
- If a project is selected, calls `kanban_state`, `list_tasks`, and `get_project_state`.
- Updates "last refreshed" timestamp in Observe drawer.
- Shows loading state, then success or error.

New task button:

- Opens task creation modal/drawer.
- Requires active project.
- Disabled with tooltip if no project is selected.

New project button:

- Opens project creation modal/drawer.

## Project Sidebar

Contents:

- Search input.
- Project list.
- Create project icon button.
- Compact project metadata.

Project item fields:

- Project name.
- Root path suffix.
- Task count if available.
- Last updated if available.
- Status dot from latest project state: ok, warning, error, unknown.

Click behavior:

- Selects project.
- Loads `kanban_state(project_id)`.
- Loads `list_tasks(project_id)`.
- Loads `get_project_state(project_id)`.
- Clears selected task unless that task belongs to the newly selected project.

Empty project state:

- Main workspace shows create project form directly.
- Fields: name, root path.
- Root path is a text input in MVP. Native folder picker can come later.

## Tasks: Kanban Board

The kanban board is the primary manual testing surface.

Columns must match `TaskStatus`:

- Backlog: `backlog`
- Pending: `pending`
- In Progress: `in_progress`
- Review: `review`
- Completed: `completed`
- Failed: `failed`
- Cancelled: `cancelled`

Column header:

- Human label.
- Status key in small monospace text.
- Count.
- Optional ready count when `list_ready_tasks` exists.

Task card fields:

- Title.
- Kind.
- Assigned agent.
- Priority.
- Status.
- Trace id suffix.
- Parent marker when `parent_id` exists.
- Readiness marker: `ready`, `blocked`, or `unknown`.
- Small timestamps if room allows.

Card click:

- Selects task.
- Opens or updates right inspector.
- Keeps board scroll position.

Card double click:

- Opens task inspector full-height.

Drag and drop:

- Not required in MVP.
- MVP status changes happen through inspector controls, because backend validation is authoritative.

Status changes:

- UI sends `set_task_status({ task_id, status })`.
- UI must show backend error if transition is invalid.
- UI may disable obvious invalid terminal transitions, but it must not duplicate all domain rules as the only validation layer.

Known transition rules from current core behavior:

- `pending -> in_progress`
- `in_progress -> review`
- `review -> completed`
- `in_progress -> completed`
- `pending/in_progress/review -> failed`
- `pending/in_progress/review -> cancelled`
- Retry can return eligible failed/review task to pending when retry IPC exists.

The UI should display terminal statuses as visually final:

- Completed: green.
- Failed: red.
- Cancelled: gray.

## Task Creation

Task creation is a modal or right-side drawer.

Fields:

- Title, required.
- Description, optional.
- Kind, required.
- Assigned agent, optional.
- Priority, numeric.
- Parent task, optional.
- Trace id, optional.
- Payload JSON, default `{}`.

Allowed initial kind options:

- `codegen`
- `research`
- `summarization`
- `qa`
- `security`
- `review`
- `generic`
- `sandbox`

Allowed initial agent options:

- `architect`
- `coder`
- `researcher`
- `qa`
- `security`
- `critic`
- `summarizer`

Validation:

- Title cannot be empty.
- Kind cannot be empty.
- Payload must be valid JSON.
- Priority must be an integer.

Submit behavior:

- Calls `submit_task`.
- On success, closes drawer, selects created task, refreshes board and task list.
- On failure, keeps form open and shows error in form and Observe drawer.

## Right Inspector

The inspector explains the selected object.

When no task is selected:

- Show active project overview.
- Show project metadata and recent system state.
- Show a short "select a task" empty state.

When a task is selected, tabs are:

### Overview

Fields:

- Task id.
- Title.
- Description.
- Status.
- Kind.
- Assigned agent.
- Priority.
- Project id.
- Parent id.
- Trace id.
- Prompt version id.
- LoRA adapter id.
- Iteration count.

Actions:

- Status segmented control or status menu.
- Refresh task/project.
- Copy id.
- Copy trace id.
- Cancel task when IPC exists.
- Retry task when IPC exists.

### Payload

Shows formatted JSON from `task.payload`.

MVP:

- Read-only JSON viewer.
- Copy JSON button.

Later:

- Edit and resubmit only through explicit task update IPC, not local mutation.

### Result

Shows formatted JSON from `task.result`.

Empty state:

- "No result yet" with no long explanatory prose.

### Scores

Fields:

- Priority score.
- Critic score.
- Human score.
- Iteration count.

If scores are absent, show `not scored`.

### Timing

Fields:

- Created at.
- Started at.
- Finished at.
- Duration when both timestamps exist.

### Dependencies

Fields:

- Parent task.
- Blocking dependencies.
- Dependent tasks.

MVP:

- Show parent id and dependency ids if available.
- Full dependency graph waits for Graph tab or extra IPC.

### Debug

Shows raw task JSON exactly as returned by backend.

Actions:

- Copy raw JSON.
- Pin raw JSON to Observe drawer.

## Observe Section And Drawer

Observability is a first-class part of the UI, not an afterthought.

There are two surfaces:

- Bottom Observe drawer available from every section.
- Full Observe section for deeper inspection.

MVP drawer tabs:

- Events
- Logs
- Traces
- Metrics
- Errors
- Raw

### Events

Purpose:

- Show lifecycle events and backend notifications.

MVP data source:

- Last command results and synthetic UI events until `subscribe_to_events` is wrapped for Tauri.

Future data source:

- Tauri Channel from `subscribe_to_events`.

Event row fields:

- Timestamp.
- Level.
- Event type.
- Project id.
- Task id.
- Agent.
- Short message.
- Trace id.

Click behavior:

- Selects related task when task id exists.
- Opens event detail in inspector or drawer side pane.

### Logs

Purpose:

- Show agent action audit logs and human-readable backend logs.

Expected log event types from architecture:

- `task_started`
- `prompt_sent`
- `response_received`
- `tool_called`
- `file_read`
- `file_written`
- `test_run`
- `status_changed`
- `thinking`
- `error`
- `human_intervention`

Log row fields:

- Timestamp.
- Level.
- Agent.
- Task id.
- Event type.
- Message.
- Duration.
- Model.
- Tool name when applicable.

Filters:

- Level.
- Agent.
- Task.
- Event type.
- Text search.
- Trace id.

Important behavior:

- Prompt and response payloads must be expandable.
- Tool args and results must be formatted as JSON.
- File writes must show path and diff summary when available.
- Errors must be sticky until acknowledged or fixed.

MVP:

- Populate from `get_project_state` if audit logs are present.
- Also show frontend command errors and command results.

Needed IPC:

- `list_audit_logs(project_id, task_id?)`
- `replay_task(task_id)`

### Traces

Purpose:

- Correlate task, agent, model, prompts, tool calls, file changes, and status transitions by `trace_id`.

Trace view layout:

- Left: trace list grouped by trace id.
- Center: chronological timeline.
- Right: selected span/event details.

Span fields:

- Trace id.
- Span id when available.
- Parent span id when available.
- Task id.
- Agent.
- Operation.
- Started at.
- Duration.
- Status.
- Error.

MVP:

- Show trace id groups from task fields and logs if present.
- Full span tree is future work.

### Metrics

Purpose:

- Show system and agent health.

MVP fields from `MetricsSnapshot` where available:

- Tasks completed.
- Tasks failed.
- Average latency.
- Cache hits.
- Cache misses.
- Success rate.

Future architecture metrics:

- CPU.
- GPU.
- RAM.
- Disk.
- Network.
- Power/fan.
- Tokens in/out.
- Average tokens per task.
- Success rate by agent.
- Model latency.
- Queue depth.

UI behavior:

- Use compact metric strips and small tables first.
- Avoid heavy charts until history is available.
- Use charts only when `metrics.history` is wired.

Needed IPC:

- `get_metrics_snapshot`
- `get_metrics_history(from, to)`

### Errors

Purpose:

- Make backend and UI failures impossible to miss.

Contents:

- Last command error.
- Backend validation errors.
- Event stream disconnect.
- JSON parse errors.
- Task transition errors.

Behavior:

- Error rows include timestamp, command, target id, message, and raw error.
- Clicking an error selects related project/task when possible.
- Copy raw error button.

### Raw

Purpose:

- Debug what the UI receives from the backend.

Contents:

- Selected task JSON.
- Selected project JSON.
- Last command request.
- Last command response.
- Last `ProjectState`.

Behavior:

- Pretty JSON by default.
- Copy button.
- Collapse large payloads.

## Full Observe Section

The full Observe section gives more room than the drawer.

Tabs:

- Overview
- Events
- Agent Logs
- LLM
- Tools
- Files
- Tests
- Metrics
- Raw

### Overview

Shows:

- Current project health.
- Queue depth.
- Running tasks.
- Failed tasks.
- Latest errors.
- Model/backend status when available.
- Event stream connection status.

### Agent Logs

Shows the audit log table with advanced filters.

Must support:

- Filter by task.
- Filter by agent.
- Filter by event type.
- Filter by trace id.
- Expand row details.
- Jump to task.

### LLM

Shows AI-specific observability:

- Prompt sent.
- System prompt version.
- Model.
- LoRA adapter.
- Token counts.
- Latency.
- Finish reason.
- Raw response.

Important:

- Raw prompts and responses are debug data. They should be visible because Crytex is local and observability-first, but they should live in an explicit debug tab, not pollute the kanban board.

### Tools

Shows tool calls:

- Tool name.
- Args.
- Result.
- Duration.
- Success/failure.
- Related file paths.

For file tools:

- Show path.
- Bytes read/written.
- Diff summary if available.
- Lock/write status if available.

For shell/test tools:

- Show command.
- Exit code.
- Stdout.
- Stderr.
- Duration.

### Files

Shows file operations from logs:

- Reads.
- Writes.
- Deletes if ever supported.
- Generated artifacts.

MVP:

- Table from logs only.

Later:

- File tree and diff viewer.

### Tests

Shows validation runs:

- Command.
- Status.
- Exit code.
- Duration.
- stdout/stderr.
- Related task.

### Metrics

Shows system and task metrics.

MVP:

- Snapshot.

Later:

- History chart.
- Agent comparison table.
- Model/backend comparison.

## Project State Section

Project state can be shown either under Projects or as an inspector tab.

Tabs:

- Summary
- Tasks
- Snapshots
- Memory
- Metrics
- Raw

Summary:

- Project metadata.
- Root path.
- Task counts by status.
- Latest snapshot id.
- Latest error.

Tasks:

- Table alternative to kanban.
- Sort by priority, status, agent, created time.

Snapshots:

- List `ProjectSnapshot` data if available.
- Show raw snapshot JSON.

Memory:

- Placeholder until RAG/memory IPC exists.

Raw:

- Full `ProjectState`.

## Graph Section

Graph is not the MVP primary surface, but the navigation model must reserve space for it.

Purpose:

- Show dependencies between tasks.
- Show workflow progression.
- Show blocked tasks and why they are blocked.

MVP:

- If dependencies are available in loaded tasks/state, render simple read-only dependency graph.
- Otherwise show "Graph data unavailable" with no fake nodes.

Later:

- React Flow graph.
- Color nodes by status.
- Click node selects task.
- Click edge shows dependency details.
- Filter by agent/status.

## Memory Section

Purpose:

- Inspect project memory, semantic chunks, snapshots, context compression, and retrieval inputs.

MVP:

- Read-only placeholder or raw project snapshot view.

Later tabs:

- Snapshots.
- RAG Chunks.
- Search.
- Compression.
- Raw.

Needed IPC:

- `list_project_snapshots`
- `search_memory`
- `list_indexed_files`
- `get_compressed_context`

## Models Section

Purpose:

- Inspect inference, embedding, reranking, and LoRA configuration.

MVP:

- Disabled/read-only until config IPC exists.

Later tabs:

- Backends.
- Models.
- LoRA.
- Embeddings.
- Rerankers.
- Health.

Key fields:

- Backend id.
- Backend kind.
- Model.
- Loaded/unloaded.
- Supports LoRA.
- Current adapter.
- Latency.
- Error state.

## Settings Section

MVP:

- Theme mode: dark only, light disabled or absent.
- Refresh interval: manual/off initially.
- Debug mode toggle for showing Raw tab by default.
- Observe drawer default open/closed.

Later:

- Storage path.
- Inference config.
- Model config.
- Project indexing settings.
- Safety/sandbox settings.

## Command Feedback

Every backend command must update a common command state:

- Command name.
- Started at.
- Finished at.
- Duration.
- Request payload.
- Response payload.
- Error.

This state feeds:

- Header loading indicators.
- Observe drawer Raw tab.
- Observe drawer Errors tab.
- Toasts or inline errors.

Toasts:

- Use sparingly.
- Errors should also persist in Observe.
- Success toasts should be short and optional.

## Loading States

Rules:

- The UI must remain interactive during refresh.
- Boards and tables show subtle loading overlays, not full-screen spinners.
- Buttons show busy state and prevent duplicate submits.
- Inspector keeps last selected object visible while refreshing.

## Empty States

No projects:

- Show create project form.
- No decorative illustration.

No tasks:

- Show create task action in the board area.

No logs:

- Show empty log table with filters disabled or inactive.

No selected task:

- Show project summary in inspector.

## Accessibility And Interaction

Required:

- Keyboard focus states.
- Buttons with icons plus accessible labels.
- Tooltips for icon-only controls.
- Copy buttons for IDs and JSON.
- No text overflow inside controls.
- Status color must not be the only status indicator; include labels.

Keyboard shortcuts can come later, but the layout should not block them:

- New task.
- Refresh.
- Open Observe.
- Search projects.

## MVP Acceptance Criteria

The first manual UI is acceptable when the user can:

- Start the Tauri app.
- Create a project.
- Select a project.
- Create a task.
- See the task in the correct kanban status column.
- Select the task.
- Inspect task overview, payload, result, scores, timing, and raw JSON.
- Change task status through backend IPC and see the board update.
- See invalid backend transitions as visible errors.
- Refresh project state.
- Open Observe drawer.
- See command history, last error, raw backend responses, and available logs/metrics.

Technical acceptance:

- Tauri command wrappers are tested.
- TypeScript build passes.
- Rust tests pass for touched crates.
- The UI fits 1366x768 and 1920x1080.
- Dark theme is the default and only polished theme.

## Implementation Order

### Phase 1: Static Shell

- App shell.
- Dark theme.
- Left nav.
- Project sidebar.
- Header.
- Empty board.
- Inspector.
- Observe drawer.

### Phase 2: Read Wiring

- `list_projects`
- `kanban_state`
- `list_tasks`
- `get_project_state`
- Raw JSON inspector.
- Command state in Observe drawer.

### Phase 3: Write Wiring

- `create_project`
- `submit_task`
- `set_task_status`
- Error handling.
- Loading states.

### Phase 4: Observability First Pass

- Logs table from `ProjectState`/audit data when present.
- Event list from command state and available backend events.
- Metrics snapshot.
- Errors tab.
- Trace grouping by trace id.

### Phase 5: Live Events

- Tauri Channel wrapper for `subscribe_to_events`.
- Live event feed.
- Auto-refresh affected task/project.

### Phase 6: Deeper Debug Tools

- Audit log IPC.
- Replay task IPC.
- Metrics history.
- Tool call details.
- Test run details.

### Phase 7: Graph And Memory

- Read-only task dependency graph.
- Memory/snapshot explorer.
- Semantic search.

## Reference Mapping From Lazyweb Research

Use these references as product direction:

- Port: agentic work management dashboard, queue/approval metrics, agent management.
- n8n: workflow dashboard and execution overview.
- Relevance AI: agent workflow builder and checklist.
- Attio/Tray/Customer.io: workflow canvas patterns for future graph mode.
- OpenAI logs: AI logs with prompt/response style inspection.
- Better Stack/Sentry/Okta: logs, traces, filters, and timeline-oriented debugging.
- ClickUp/Airtable/Shortcut: kanban board density and task card scanning.

Design synthesis:

- Crytex board borrows task/status scanning from Linear/ClickUp.
- Crytex Observe borrows logs/traces/debug structure from OpenAI logs and Better Stack.
- Crytex future Graph borrows workflow/canvas structure from n8n/Relevance AI/Attio.
- Crytex shell borrows operational density from Port and internal tools.
