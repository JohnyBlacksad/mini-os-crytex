# Crytex UI Product Requirements

Status: canonical product UI requirements.

This document is the source of truth for product-level UI work. `UI_REQUIREMENTS.md` and `UI_RESEARCH_AND_SPEC.md` are historical notes for the temporary Tauri IPC test bench and early research.

## Product Vision

Crytex is a local self-improving agentic IDE and development OS.

The main user action is not "create task". The main user action is "state goal". The AI architect/teamlead decomposes the goal into a task graph, assigns agents, runs work through local or configured models, uses automatic RAG over the project, and learns from approved/rejected outcomes through Prompt Evolution and LoRA Evolution.

The UI must make this system testable by hand without hiding the important machinery.

## Non-Negotiables

1. Humans create goals. AI creates tasks.
2. The built-in IDE is first-class and always present in the product model.
3. Automatic project indexing is first-class. The watcher starts when a project is opened.
4. RAG on Qdrant Edge is first-class, with selectable embedding and reranker models.
5. Hugging Face model download and local model runtime management are first-class.
6. Prompt Evolution and LoRA Evolution are the core moat, not a stats page.
7. The knowledge base is filled automatically from project files, code, docs, PDFs, logs, approvals, rejections, and outcomes.
8. Human work is mostly approve/reject/comment/manual edit, not manually maintaining a task board.
9. Observability must expose prompts, retrieval, model calls, tool calls, file operations, tests, approvals, failures, retries, benchmarks, and evolution decisions.
10. Dark, technical UI is the default visual direction.

## Reference Direction

Research fallback was used because Lazyweb MCP tools were not visible in the current tool list.

External reference patterns:

- Cursor and Windsurf: IDE-centered AI workflow with agent/chat side panels, file context, diffs, and coding loop.
- LangSmith and Opik: trace-first LLM observability with prompt, model call, tool call, latency, cost, errors, and evaluation detail.
- Qdrant Dashboard/Cloud UI: collections, points, vector search, snapshots, cluster/index status.
- Hugging Face Hub: searchable model catalog, model cards, downloads, local cache/runtime mapping.
- Weights & Biases/Comet-style experiment UIs: model registry, eval runs, version promotion, rollback, comparisons.

Design implication: Crytex should feel like an IDE plus AI operations console, not like a project-management SaaS.

Useful sources to re-check during implementation:

- Qdrant Web UI: collections, REST console, snapshots, and local/cloud dashboard patterns.
- Qdrant collections/search docs: collection status, vector schemas, points, dense/sparse vectors, and search behavior.
- Hugging Face Hub: model search, model cards, downloading models, and model metadata.
- Opik: LLM/agent observability with traces for LLM calls, retrieval steps, tool invocations, and agent steps.
- W&B Weave/Models: evaluation dashboards, model registry, artifacts, versioning, lineage, and RAG evaluation workflows.

## Primary Happy Path

1. User starts Crytex.
2. App shows readiness: GPU/VRAM, model runtime, Qdrant Edge, watcher, indexer, sandbox, active LoRA, active project.
3. User opens or creates a project.
4. Watcher starts automatically.
5. Indexer chunks project files and documents with overlap and writes vectors into Qdrant Edge.
6. User confirms or configures LLM, embedding model, reranker, backend, quantization, and LoRA adapter.
7. User writes a high-level goal in the Goal panel.
8. Architect creates a plan and generated task graph.
9. User reviews plan, dependencies, risks, expected file changes, acceptance criteria, and required approvals.
10. User approves the plan.
11. Agents execute tasks using RAG, tools, models, sandbox, IDE/project context, and LoRA when available.
12. Observe shows the live trace: prompts, retrieval, model calls, tools, diffs, tests, failures, retries.
13. User reviews output in IDE/diff view.
14. User approves or rejects with a comment.
15. Rejected work returns to revision.
16. Approved/rejected outcomes become experience data.
17. Prompt Evolution and LoRA Evolution evaluate whether new versions should be proposed, benchmarked, promoted, or rolled back.
18. User finishes session and sees summary: changed files, unresolved risks, approvals, learned experience, model/evolution state.

## Main App Shell

### Activity Rail

Left vertical rail with stable sections:

- Workspace
- Goals
- Runs
- Generated Tasks
- Index/RAG
- Models
- Evolution
- Observe
- Settings

### Status Bar

Always visible. Required indicators:

- Current project
- Watcher status
- Indexing status
- Qdrant Edge status
- Active LLM
- Embedding model
- Reranker
- Active LoRA adapter
- GPU/VRAM
- Sandbox
- Current run
- Pending approvals

### Bottom Panel

Tabs:

- Terminal
- Problems
- Output
- Tests
- Tool Calls
- Retrieval
- Logs

### Right Inspector

Context-sensitive details for the selected entity:

- File metadata
- Indexed chunks
- Retrieval trace
- Task details
- Run step
- Model call
- Tool call
- Prompt version
- LoRA adapter
- Evaluation result

## Core Screens

### Workspace

Default screen. It must combine IDE, project cockpit, and goal entry.

Required areas:

- File explorer
- Editor tabs
- Code editor
- Diff review
- Agent goal panel
- Run/plan preview
- Right inspector
- Bottom panel
- Status bar

Required actions:

- Open project
- Save file
- Create file
- Rename file
- Delete file with confirmation
- Create goal from project
- Create goal from active file
- Create goal from selected text
- Apply/reject agent diff
- Start/rebuild indexing

### Built-In IDE

The IDE is mandatory because users may want to write code manually.

Required capabilities:

- Open files from project tree
- Edit and save files
- Dirty state
- Multiple tabs
- Syntax highlighting
- Diagnostics
- Go to definition
- Find references
- Symbols/outline
- Inline AI suggestions
- Agent diff hunks
- Apply/reject hunk
- Read-only generated artifacts where needed
- Link selected code to a goal, task, retrieval trace, or run

Architecture mapping:

- `crates/crytex-ide` is the IDE/LSP bridge.
- `ide_service.rs` maps to LSP client management and definition/reference operations.
- `protocol.rs` maps to inline suggestions, diff hunks, suggestion actions, and request/response protocol.

### Goals

Purpose: user intent and architect planning.

Required UI:

- Goal composer
- Context chips: project, file, selection, diagnostics, terminal output, retrieval result
- Goal list
- Goal detail
- Architect plan
- Generated task graph
- Dependency graph
- Risk list
- Acceptance criteria
- Approve/reject/edit plan
- Start run
- Pause/cancel

Rules:

- `New goal` is primary.
- `New task` is debug/admin only.
- Human approval happens before execution when the plan has file, shell, model, or sandbox impact.

### Runs

Purpose: live and historical execution sessions.

Required UI:

- Active run timeline
- Agent chain
- Current step
- Queue
- Blockers
- Retry/revision state
- Run summary
- Pause/cancel/retry controls
- Links to Observe, tasks, files, tests, and approvals

### Generated Tasks

Purpose: inspect AI-created task graph, not maintain a human Kanban.

Required UI:

- Columns or grouped list by task state
- Dependency view
- Assigned agent
- Assigned model
- Assigned LoRA adapter
- Inputs
- Outputs
- Acceptance criteria
- Current evidence
- Approval status
- Retry/revision loop

Allowed:

- Manual task creation only under debug/admin mode.

### Index/RAG

Purpose: make automatic knowledge ingestion visible and controllable.

Required UI:

- Watcher status
- Indexer queue
- Indexed files
- Skipped files
- Failed files
- Chunk count
- Chunk overlap settings
- Qdrant Edge status
- Collection status
- Embedding model selector
- Reranker selector
- Hybrid/vector search test box
- Retrieval trace viewer
- Chunk viewer
- Rebuild/pause/resume controls

Critical behavior:

- Opening a project starts watcher and indexing automatically.
- Code and non-code documents, including PDFs, are supported.
- User can inspect why a retrieval result was used.

### Models

Purpose: make local model operations testable from the UI.

Required UI:

- Hugging Face search
- Model detail/model card summary
- Download queue
- Download progress
- Local model inventory
- Load/unload controls
- Backend selection
- Quantization/optimization options
- GPU/VRAM fit estimate
- Active runtime status
- Default LLM selector
- Default embedding model selector
- Default reranker selector
- Runtime errors and recovery actions

Critical behavior:

- User can download an LLM from Hugging Face.
- Backend downloads, registers, deploys, and optimizes the model for the user's GPU where supported.

### Evolution

Purpose: expose the self-improvement loop.

Required UI:

- Experience dataset summary
- Success/failure signals
- Prompt versions
- Prompt mutation proposals
- Prompt A/B tests
- Prompt benchmark results
- Promote/rollback prompt
- LoRA adapters
- LoRA training queue
- LoRA benchmark results
- LoRA promote/rollback
- "Not enough data yet" state

Critical behavior:

- Evolution is driven by approved/rejected task outcomes.
- New prompts/adapters must be evaluated before promotion.
- UI must show whether a proposed evolution improved or degraded results.

### Observe

Purpose: understand exactly what AI did and why.

Required UI:

- Live event stream
- Trace tree
- Prompt logs
- Model calls
- Token counts
- Latency
- Retrieval traces
- Tool calls
- File reads/writes
- Diffs
- Test runs
- Sandbox events
- Human approval/rejection events
- Errors and retries
- Benchmark/evolution decisions
- Replay selected task/run

Critical behavior:

- Observe must be useful during execution, not only after the run.
- Every major user-visible agent decision must be traceable to prompt, retrieval, tool output, or human feedback.

## Required IPC Surface

### Project and IDE

- `open_project`
- `create_project`
- `list_project_files`
- `read_file`
- `write_file`
- `save_file`
- `rename_file`
- `delete_file`
- `start_language_server`
- `goto_definition`
- `find_references`
- `list_diagnostics`
- `request_inline_suggestion`
- `apply_diff`
- `reject_diff`

### Goals and Orchestration

- `submit_goal`
- `get_goal`
- `list_goals`
- `approve_plan`
- `reject_plan`
- `start_run`
- `pause_run`
- `cancel_run`
- `retry_task`
- `approve_task`
- `reject_task`

### Index and RAG

- `start_project_index`
- `pause_project_index`
- `rebuild_project_index`
- `get_index_status`
- `list_indexed_files`
- `list_failed_index_files`
- `search_rag`
- `get_retrieval_trace`
- `list_chunks`
- `get_chunk`
- `set_embedding_model`
- `set_reranker_model`

### Models

- `search_huggingface_models`
- `download_model`
- `get_model_download_status`
- `list_local_models`
- `load_model`
- `unload_model`
- `get_model_runtime_status`
- `set_default_backend`
- `set_embedding_backend`
- `set_rerank_backend`
- `estimate_gpu_fit`

### Evolution

- `get_experience_dataset_summary`
- `list_prompt_versions`
- `create_prompt_evolution_proposal`
- `list_prompt_ab_tests`
- `run_prompt_benchmark`
- `promote_prompt_version`
- `rollback_prompt_version`
- `list_lora_adapters`
- `queue_lora_training`
- `get_lora_training_status`
- `run_lora_benchmark`
- `promote_lora_adapter`
- `rollback_lora_adapter`

### Observe

- `subscribe_to_events`
- `list_audit_logs`
- `get_run_trace`
- `get_task_replay`
- `list_tool_calls`
- `list_file_operations`
- `list_test_runs`
- `get_metrics_snapshot`
- `get_metrics_history`

## MVP Acceptance Criteria

First serious UI milestone must prove:

1. User can open/create a project.
2. Watcher starts automatically.
3. Indexing status is visible.
4. RAG/Qdrant status is visible.
5. User can open and edit files in the built-in IDE.
6. User can submit a goal from project/file/selection context.
7. Architect creates tasks automatically.
8. User can approve or reject generated plan.
9. At least one real or stubbed run can execute from the UI.
10. Observe shows prompt, retrieval, tool, file, test, and approval events.
11. User can approve or reject task output.
12. Feedback is stored as experience data.
13. Evolution screen shows whether there is enough data to evolve prompts or LoRA.
14. Status bar reflects model, Qdrant, indexer, watcher, LoRA, GPU, sandbox, and run state.

## First UI Rebuild Plan

1. Make Workspace the default screen.
2. Replace primary `New task` with `New goal`.
3. Add IDE shell: file tree, editor tabs, code editor placeholder or Monaco/CodeMirror integration.
4. Add goal composer with context chips.
5. Rename Kanban to Generated Tasks.
6. Move manual task creation into debug/admin.
7. Add Index/RAG, Models, Evolution as first-class nav items.
8. Add Observe as a real trace/log surface, not just command history.
9. Add persistent status bar for runtime readiness.
10. Wire current IPC where available and stub missing commands explicitly.

## Visual Direction

Dark technical interface. Dense but readable. No marketing hero page.

Layout should favor:

- IDE-like spatial stability
- Small, strong icon buttons
- Clear tabs
- Split panes
- Trace timelines
- Tables for operational data
- Graph view for plans/dependencies
- Diff-first review for code changes
- Compact cards only for repeated entities, not whole page sections

Avoid:

- Landing-page composition
- Oversized decorative panels
- Manual-task-board-first UX
- One-note purple/blue gradient theme
- Hiding agent internals behind vague chat bubbles
