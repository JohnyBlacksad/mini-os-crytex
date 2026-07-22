# Autonomous Evolution Policy

Crytex must attribute a failure before changing the system. The policy prevents
LoRA from learning the wrong lesson when the real cause is bad context, schema
formatting, weak critic feedback, security policy, or missing benchmark coverage.

## Actions

- `rag_fix`: retrieval, indexing, rerank, prompt-injection scan, or context-budget issue.
- `prompt_evolution`: schema/format/instruction failure.
- `lora_training`: repeated role skill failure after context and prompt are ruled out.
- `critic_role_evolution`: critic feedback is too vague or lacks blocking issues.
- `security_policy`: prompt injection, unsafe tool use, or missing policy rule.
- `benchmark_expansion`: attribution is uncertain or regression coverage is missing.

## Routing Rules

- Bad or missing context routes to `rag_fix`, not LoRA.
- Schema/format failures route to `prompt_evolution` before LoRA.
- Repeated role skill failures route to role-scoped `lora_training`.
- Weak critic detail routes to `critic_role_evolution`.
- Security/tool misuse routes to `security_policy`.
- Unknown attribution routes to `benchmark_expansion`.

## Diagnostics

Every decision emits `autonomous_evolution_decision` diagnostics with:

- role;
- failure kind;
- selected action;
- reason;
- repeated count;
- source evidence.

## CLI

```powershell
crytex evolution run --all-roles --json
crytex prove evolution-policy --report-path reports\evolution-policy-p11-proof.json
```

Current development binary:

```powershell
cargo run -p crytex-kernel -- evolution run --all-roles --json
cargo run -p crytex-kernel -- prove-evolution-policy --report-path reports\evolution-policy-p11-proof.json
```

## Design References

- Failure attribution for multi-agent systems: https://arxiv.org/html/2604.22708v1
- Automated multi-agent failure attribution: https://openreview.net/forum?id=GazlTYxZss&noteId=cypPlShPMW
- Regression and capability eval discipline: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- Metric-based automated regression testing: https://developers.openai.com/api/docs/guides/evaluation-best-practices
- Preference training uses chosen/rejected pairs: https://papers.nips.cc/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html
