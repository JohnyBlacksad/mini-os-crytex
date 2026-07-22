# Example: Critic Role

This example shows how a critic role returns structured feedback that can drive
remediation, reward, Prompt Evolution, and LoRA learning.

## Goal

Review a Python coder artifact and decide whether it is acceptable. The critic
must provide blocking issues and a remediation proposal when rejecting.

## CLI

```powershell
crytex kanban show --status review --json
crytex review show --json
crytex review reject <task-id> --failure-type missing-tests --comment "Missing malformed CSV and empty-file tests"
crytex kanban history --run latest --json
crytex diag export --run latest --out reports\critic-review.json
```

Autonomous mode uses the same backend contract without manual `review reject`:
the critic emits structured decision data and Crytex moves the task to
`remediation`.

## Role Contract

Role: `critic-coder`, `critic-analyst`, `critic-researcher`, or `critic-etc`.

Expected artifact:

- decision: accept, reject, or request changes;
- reason;
- blocking issues;
- target task id;
- remediation proposal;
- failure type;
- evidence ids.

Completed review without typed artifact is invalid. Rejected outputs become
negative examples for the relevant producing role; weak or shallow criticism
becomes an evolution signal for the critic role.

## Diagnostics

Inspect Kanban movement events, audit logs, task result JSON, critic feedback,
failure taxonomy, reward, and LoRA dataset rows.

## Troubleshooting

- If a critic rejects without blocking issues, the artifact contract should fail.
- If feedback is too generic, route to critic role evolution.
- If the critic follows malicious project instructions, run
  `crytex security prove --malicious-rag-fixture`.
