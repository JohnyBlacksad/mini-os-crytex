# Example: QA Role

This example shows how the `qa` role verifies implementation quality through
the Crytex CLI. QA is not a UI decoration; it is a backend role with artifacts,
diagnostics, and benchmark evidence.

## Goal

Validate that a completed feature has meaningful tests, covers edge cases, and
does not regress existing behavior.

## CLI

```powershell
crytex kanban show --status review --role qa --json
crytex rag search "test strategy and edge cases for CSV import" --rerank --explain --json
crytex run start --json
crytex review show --json
crytex diag export --run latest --out reports\qa-run.json
```

Optional proof:

```powershell
crytex backend-acceptance --full --json --deterministic --report-path reports\backend-acceptance.json
```

## Role Contract

Role: `qa`.

Expected artifact:

- test plan;
- test files or commands executed;
- pass/fail evidence;
- uncovered edge cases;
- risk summary;
- recommendation: approve, reject, or request remediation.

QA must preserve evidence ids and command output references. If the QA answer is
too vague, critic feedback routes to `critic-etc` or `qa` improvement depending
on whether the weakness is critique quality or test strategy.

## Diagnostics

QA diagnostics include task id, dependency chain, test commands, sandbox result,
RAG evidence, prompt version, model id, and any LoRA adapter used by the role.

## Troubleshooting

- If tests cannot run, check sandbox doctor and tool permissions.
- If QA misses obvious edge cases repeatedly, build negative examples for `qa`.
- If QA output is malformed, fix prompt schema through Prompt Evolution first.
