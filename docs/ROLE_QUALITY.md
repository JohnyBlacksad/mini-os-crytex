# Role Quality Contracts

Crytex treats each agent role as a measurable backend capability. A role is not
complete because a prompt exists; it is complete only when the backend knows the
role's system prompt, output schema, typed artifact contract, metrics, failure
taxonomy, benchmark fixture, and LoRA adapter route.

## Roles

The production role catalog covers:

```text
orchestrator, architect,
coder-python, coder-rust, coder-ts, coder-etc,
analyst, researcher, qa, devops, security,
critic-analyst, critic-coder, critic-researcher, critic-etc,
summarizer
```

Legacy aliases `coder` and `critic` resolve to `coder-etc` and `critic-etc`
for catalog lookup while staying compatible with existing workflow nodes.

## Contract Fields

Every role contract contains:

- `role_id`: canonical role identifier.
- `system_prompt`: role-level behavioral instruction.
- `output_schema`: JSON schema for the role artifact.
- `artifact_contract`: artifact kind and required fields.
- `metrics`: quality measurements tracked per role.
- `failure_taxonomy`: typed failure reasons used for training and analysis.
- `prompt_sources`: source paths imported from `trash/agent-skills-main` and
  `trash/skills-main`.
- `benchmark_fixture`: deterministic role fixture for mocked and real smoke.

## Artifact Contracts

Role completion requires typed artifacts. `TaskService::set_result` validates
the artifact before mutating task state, so a role task cannot become
`Completed` from an unstructured string.

Examples:

- `coder-*` -> `patch_artifact`: `summary`, `files_changed`.
- `critic-*` -> `review_decision`: `decision`, `reason`, `blocking_issues`,
  `target_task`, `remediation_proposal`.
- `orchestrator` -> `task_graph_artifact`: `summary`, `tasks`,
  `dependency_edges`.
- `researcher` -> `research_artifact`: `summary`, `sources`.
- `qa` -> `test_report_artifact`: `summary` or `test_results`.
- `security` -> `security_report_artifact`: `summary` or `risk`.

## Critic Feedback

Critics produce structured remediation data:

```json
{
  "decision": "approve | request_changes | reject",
  "reason": "why this decision was made",
  "blocking_issues": ["issue that blocks completion"],
  "target_task": "task id or role being reviewed",
  "remediation_proposal": {
    "assigned_agent": "coder-rust",
    "goal": "concrete repair task"
  }
}
```

`request_changes` and `reject` require non-empty `blocking_issues`.

## Clean Sessions And LoRA Routing

Workflow agent nodes create a fresh `agent_session` per role. The session
contains:

- `session_id`
- `trace_id`
- `role`
- `role_contract_id`
- `artifact_kind`
- `node_id`
- `input_key`
- `clean_context`
- optional `lora_adapter_id`
- optional `lora_selection_reason`

Role-specific LoRA routing uses `AgentRole`, including specialized roles such
as `coder-python` and `critic-coder`, so switching agents can hot-swap adapters
without replacing the whole LLM runtime.

## Proof Command

Current development binary:

```powershell
cargo run -p crytex-kernel -- prove-role-quality-contracts --report-path reports\role-quality-p6-proof.json
```

The proof artifact includes:

- contract coverage for every production role;
- prompt source evidence from `trash/agent-skills-main` and `trash/skills-main`;
- deterministic mocked smoke per role;
- structured critic feedback schema;
- role-specific LoRA hot-swap evidence;
- pass/fail gates.

Exit codes follow the global CLI contract: `0` for pass, `1` for command or
write failure, and `2` for a proof gate failure.
