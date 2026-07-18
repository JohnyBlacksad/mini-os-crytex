# Crytex Product Roadmap

Дата: 2026-07-16

Назначение: единый рабочий план от текущего состояния до product-ready системы. Этот документ фиксирует скоуп, порядок фаз, критерии готовности и запрет на бесконечный "фундамент" без пользовательского результата.

## 0. Product Vision

Crytex - локальная self-improving agentic IDE и операционная среда для работы над проектами.

Главная пользовательская петля:

1. Пользователь открывает проект.
2. Система автоматически индексирует проект: код, документы, PDF, заметки.
3. Пользователь выбирает/скачивает модель, embedding model, reranker, backend и LoRA policy.
4. Пользователь формулирует цель, а не создает задачи руками.
5. Architect/orchestrator декомпозирует цель в task graph.
6. Агентная цепочка выполняет задачи с RAG, инструментами, sandbox, IDE context и выбранной моделью.
7. Critic/reviewer проверяет результат, дает причину reject или пропускает на human review.
8. Пользователь approve/reject с комментарием.
9. Reward, experience, prompt evolution и LoRA evolution обновляют систему.
10. UI показывает весь процесс: цепочку, trace, логи, RAG evidence, critic feedback, artifacts, diff, reward.

## 1. Current State

Текущий статус: ранний working vertical slice, не product-ready.

Что уже доказано:

- Real Ollama smoke проходит на `qwen3.5:9b`.
- Цепочка работает: goal -> generated tasks -> agents -> critic -> optional remediation -> human review -> reward.
- RAG marker реально попадает в LLM request.
- Watcher/indexer участвуют в создании проекта.
- Audit/trace слой есть.
- `export_run_diagnostics` собирает структурированный отчет по run/trace.
- JSON parser стал устойчивее к реальным LLM-ошибкам: trailing commas и missing commas между полями.

Критические пробелы:

- UI пока не является основным продуктовым workflow.
- Observe UI еще не использует `export_run_diagnostics` как основной источник правды.
- Модельный UX не замкнут: download -> optimize -> select -> run -> diagnostics.
- Prompt Evolution и LoRA Evolution имеют backend-каркас, но не полноценную product loop.
- IDE есть как обязательный продуктовый скоуп, но пока не является рабочей средой.
- Нет стабильного release/e2e matrix: локальная модель, managed model, cloud model, RAG-only documents, code project.

## 2. Product-Ready Definition

Crytex считается product-ready alpha, когда пользователь без разработчика может:

1. Установить и открыть desktop app.
2. Создать или открыть проект.
3. Увидеть, что watcher/indexer/RAG/model runtime готовы.
4. Скачать или выбрать модель через UI.
5. Настроить cloud API keys при необходимости.
6. Написать цель в goal-first интерфейсе.
7. Получить task graph от architect/orchestrator.
8. Approve/reject plan.
9. Запустить agent chain.
10. Увидеть Observe report: trace, agent steps, logs, RAG evidence, model requests, critic feedback, remediation, reward.
11. Открыть результат в IDE/diff view.
12. Approve/reject result с комментарием.
13. Увидеть, что experience записан и доступен Evolution layer.
14. Увидеть состояние Prompt Evolution/LoRA Evolution: not enough data, queued, training, benchmarked, promoted, rolled back.

## 3. Roadmap Phases

### Phase 1 - Observable Working Product Slice

Цель: пользователь руками запускает реальный goal-first workflow и видит доказательство работы в UI.

Scope:

- Observe screen consumes `export_run_diagnostics`.
- Runs screen has "Open diagnostics" action for last/current run.
- Workspace is goal-first: central area is agent console + run status, Kanban is secondary.
- Runtime status bar shows model/backend, RAG/indexer/watcher, sandbox, active project.
- UI exposes happy path:
  - create/open project
  - submit goal
  - approve plan
  - start run
  - approve/reject human review
  - inspect diagnostics

Exit criteria:

- Manual UI run with Ollama completes the same path as real smoke.
- UI shows critic feedback and remediation when reject occurs.
- UI shows RAG evidence and reward evidence from diagnostics report.
- No hidden terminal-only requirement for the main happy path.

### Phase 2 - Model Management Product Loop

Цель: человек управляет моделями из UI, а не ENV/терминалом.

Scope:

- Ollama inventory and active model selection polished.
- Managed model registry UI:
  - add HuggingFace model
  - download
  - show progress/status
  - select as active runtime
- Hardware detection visible:
  - GPU/VRAM
  - recommended quantization
  - context size
  - GPU layers
- Runtime selection emits diagnostics events.
- Cloud provider settings:
  - OpenAI-compatible endpoint
  - API key
  - model id
  - token optimizer route

Exit criteria:

- User can switch between Ollama and managed model without restart.
- Download failure/success visible in UI.
- Active runtime is visible in diagnostics and status bar.

### Phase 3 - RAG and Project Intelligence

Цель: индексирование проекта становится first-class system, а не скрытой side effect.

Scope:

- Index/RAG screen:
  - watcher status
  - last indexed files
  - chunk count
  - Qdrant Edge status
  - embedding model
  - reranker model
  - manual search/test query
- File/document parsing:
  - code
  - markdown
  - PDF
  - plain text
- Incremental reindex evidence in Observe.
- Retrieval preview attached to agent tasks.

Exit criteria:

- User sees why a model received specific context.
- RAG search is testable from UI.
- Index failures surface as actionable UI errors.

### Phase 4 - Agent Chain Hardening

Цель: chain работает не только в smoke, а устойчиво на разных задачах.

Scope:

- Task graph viewer:
  - dependencies
  - current agent
  - artifact handoff
  - status transitions
- Critic feedback is structured:
  - decision
  - reason
  - blocking issues
  - suggested remediation
- Orchestrator creates debug/remediation tasks from critic rejection.
- Retry loop has max iterations, failure reasons, and human intervention path.
- Artifact model:
  - files changed
  - commands run
  - test output
  - generated report

Exit criteria:

- E2E tests cover approve path, reject path, remediation path, failed path.
- UI shows why a task moved forward or backward.
- No silent "completed" without real artifact/evidence.

### Phase 5 - Built-In IDE Minimum

Цель: IDE перестает быть placeholder и становится рабочей поверхностью.

Scope:

- File tree.
- Editor tabs.
- Read/write files.
- Diff viewer for agent changes.
- Basic diagnostics list.
- Terminal/output panel.
- Link task artifacts to files/diffs.

Exit criteria:

- User can inspect and edit files without leaving Crytex.
- Human review happens against actual artifacts/diffs.
- Agent output can be opened from task/diagnostics into IDE.

### Phase 6 - Evolution MVP

Цель: главная фича проекта становится реальной product loop.

Scope:

- Experience dataset UI:
  - approved/rejected tasks
  - reward
  - critic score
  - human score
  - prompt version
  - LoRA adapter
- Prompt Evolution:
  - baseline prompt
  - mutation proposal
  - A/B test run
  - benchmark result
  - promote/rollback
- LoRA Evolution:
  - training queue
  - dataset threshold
  - trainer backend status
  - adapter registry
  - benchmark result
  - promote/rollback
- Clear "not enough data yet" states.

Exit criteria:

- At least one prompt mutation can be generated, evaluated, and promoted/rolled back.
- At least one LoRA training job can be queued or explicitly marked unsupported by current hardware/backend with a clear reason.
- Evolution is driven by real approved/rejected task outcomes.

### Phase 7 - Product Hardening

Цель: alpha release, который можно давать человеку.

Scope:

- Error boundaries in UI.
- Persistent settings.
- Project reopen.
- Data migration strategy.
- Crash-safe watcher/indexer shutdown.
- Logs export.
- Smoke/e2e matrix:
  - Ollama
  - managed local model
  - cloud model
  - code project
  - docs/PDF project
- Performance budget:
  - indexing time
  - memory
  - vector DB size
  - model load time
  - UI responsiveness

Exit criteria:

- Fresh install works.
- Main happy path works without terminal.
- Failures are visible and recoverable.
- Release build can be produced.

## 4. Immediate Next Milestone

Milestone: "Hands-On Alpha Slice"

Goal: UI lets the user run and inspect the same successful path currently proven by real smoke.

Implementation order:

1. Add diagnostics API call to frontend.
2. Add Run Diagnostics panel to Observe.
3. Wire last run -> diagnostics.
4. Show runtime/model/trace summary.
5. Show agent chain and task rows.
6. Show critic feedback/remediation.
7. Show RAG evidence and reward evidence.
8. Move Workspace center toward goal console + run status.
9. Keep Kanban as secondary inspector.
10. Add manual test checklist for this path.

This is the next "meat" milestone. Until it is done, avoid expanding IDE, LoRA trainer, or extra model backends unless the work directly supports this slice.

## 5. Scope Control Rules

- No new subsystem without a UI or e2e proof path.
- No "completed" status without evidence: artifact, log, diagnostic event, reward, or test.
- No UI screen that hides critical product state behind placeholders.
- Every major backend feature must have:
  - unit test
  - integration/e2e test
  - Observe evidence
  - UI entry point
- Kanban is not the product center. Goal console, IDE, and Observe are the center.
- Evolution is not a metrics page. It is the product moat and must be wired to real outcomes.

## 6. Completion Scoreboard

Current rough readiness:

| Area | Status | Readiness |
| --- | --- | --- |
| Goal orchestration | Working vertical slice | 50% |
| Agent execution | Ollama smoke works | 45% |
| Critic/remediation | Proven in smoke | 40% |
| RAG/indexing/watcher | Works but UI thin | 45% |
| Observe/diagnostics | Backend ready, UI pending | 40% |
| UI workflow | Early shell | 15% |
| Model management | Partial | 25% |
| Cloud model config | Not productized | 10% |
| IDE | Placeholder | 10% |
| Prompt Evolution | Backend pieces | 20% |
| LoRA Evolution | Backend pieces | 15% |
| Release hardening | Not started | 5% |

## 7. Definition of Done for Next Work Session

The next implementation session is complete only when:

- UI can call `export_run_diagnostics`.
- Observe displays diagnostics for the last run.
- The displayed data includes model/backend, trace, tasks, critic feedback, remediation, RAG evidence, reward evidence.
- Frontend tests cover the API path.
- Rust tests still pass.
- `cargo clean` is run after Rust verification.

