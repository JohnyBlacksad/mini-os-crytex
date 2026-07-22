# Example: Python Coder Role

This example shows how the `coder-python` role should be used from the Crytex
CLI when the goal is to add production Python code with tests and evidence.

## Goal

Add CSV import validation to a Python service. The agent must read project
documentation through RAG, implement code, run tests in the sandbox, and leave a
typed artifact for QA and critic roles.

## CLI

```powershell
crytex project open A:\Projects\billing-service
crytex index run --project billing-service
crytex rag search "CSV import validation rules" --rerank --explain --json
crytex goal submit "Add Python CSV import validation with tests and docs"
crytex plan show --json
crytex run start --json
crytex kanban show --role coder-python --json
crytex diag export --run latest --out reports\python-coder-run.json
```

Development binary:

```powershell
cargo run -p crytex-kernel -- rag search "CSV import validation rules" --project-id billing-service --path A:\Projects\billing-service --rerank --explain --json --diagnostics-path reports\python-coder-rag.json
```

## Role Contract

Role: `coder-python`.

Expected artifact:

- files changed;
- implementation summary;
- tests added or updated;
- commands run;
- RAG evidence ids used;
- prompt version id;
- model id;
- LoRA adapter id when active.

The coder must not mark the task complete without a typed artifact. If tests are
missing, critic feedback should route the task to remediation and create a
negative training example for the coder role.

## Diagnostics

Inspect:

- `selected_chunks` in RAG diagnostics;
- tool audit log for file and process calls;
- Kanban card status and dependency chain;
- benchmark or test command output;
- task `prompt_version_id` and `lora_adapter_id`.

## Troubleshooting

- If the role writes code that ignores project rules, inspect RAG context first.
- If the output format is wrong, route to Prompt Evolution.
- If the same Python quality failure repeats with good context and valid schema,
  route to LoRA dataset/training for `coder-python`.
