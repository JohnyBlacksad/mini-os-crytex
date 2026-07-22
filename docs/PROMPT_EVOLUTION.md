# Prompt Evolution

Prompt Evolution is the first remediation path for role behavior that is wrong
because of instructions, schemas, or output formatting. LoRA receives examples
only after the prompt contract is healthy enough that bad outputs are useful
training signal rather than prompt noise.

## Contract

- Every task stores the active `prompt_version_id` at submission time.
- `crytex prompts propose` creates an inactive challenger from the active
  baseline.
- A challenger cannot become active just because it was mutated.
- `crytex prompts benchmark` compares the active baseline and challenger with a
  held-out benchmark gate.
- Regression benchmark metadata is mandatory. If the gate omits
  `regression.passed=true`, the challenger is rejected even when held-out score
  improves.
- `crytex prompts promote` activates only a version that already has an accepted
  benchmark decision.
- `crytex prompts rollback` reactivates an earlier version and records a typed
  rollback decision.
- Diagnostics record the prompt decision, scores, regression status, baseline
  id, challenger id, reason, and gate metadata.
- Schema and format failures route to Prompt Evolution before LoRA.

## CLI

Production contract:

```powershell
crytex prompts status --agent coder-python --json
crytex prompts propose --agent coder-python --operator inject-example --json
crytex prompts benchmark --agent coder-python --challenger <version-id> --regression-suite fixtures\prompt-regression.jsonl --json
crytex prompts promote --agent coder-python --version <version-id> --json
crytex prompts rollback --agent coder-python --to <version-id> --json
crytex prove prompt-evolution --report-path reports\prompt-evolution-p7-proof.json
```

Current development binary:

```powershell
cargo run -p crytex-kernel -- prompts status --agent coder-python --json
cargo run -p crytex-kernel -- prompts propose --agent coder-python --operator inject-example --json
cargo run -p crytex-kernel -- prompts benchmark --agent coder-python --challenger <version-id> --regression-suite fixtures\prompt-regression.jsonl --json
cargo run -p crytex-kernel -- prompts promote --agent coder-python --version <version-id> --json
cargo run -p crytex-kernel -- prompts rollback --agent coder-python --to <version-id> --json
cargo run -p crytex-kernel -- prove-prompt-evolution --report-path reports\prompt-evolution-p7-proof.json
```

## Trait Boundaries

- `PromptVersionRepository` provides prompt storage, active-version lookup, and
  active-version switching.
- `ExperienceRepository` provides reward history for prompt fitness.
- `PromptBenchmarkGate` evaluates baseline versus challenger. Implementations
  can use deterministic CI, local runtime, cloud runtime, or a full benchmark
  harness without changing `PromptEvolutionService`.
- `PromptFailureRouter` classifies failure ownership before LoRA collection.

These boundaries keep Prompt Evolution open for new benchmark runners and
closed against direct dependency on a concrete model runtime or storage engine.

## JSON Decision Shape

```json
{
  "agent": "coder-python",
  "baseline_version_id": "01...",
  "challenger_version_id": "01...",
  "decision_kind": "promoted",
  "accepted": true,
  "reason": "winner=Challenger, delta_pass_rate=0.2500",
  "baseline_score": 0.5,
  "challenger_score": 0.9,
  "regression_passed": true,
  "diagnostics": {
    "kind": "prompt_evolution_decision",
    "decision": "promoted"
  }
}
```

## Failure Routing

| Failure kind | First route | Reason |
| --- | --- | --- |
| `schema` | Prompt Evolution | The role output contract is wrong or under-specified. |
| `format` | Prompt Evolution | The system prompt must constrain response shape first. |
| `quality` | LoRA | The prompt contract may be correct but behavior needs learned examples. |
| `safety`, `tool_use`, `other` | Critic | Needs policy/critic diagnosis before training signal is trusted. |

## Proof

`prove-prompt-evolution` emits a deterministic JSON artifact proving:

- challenger creation does not activate the mutation;
- missing regression metadata rejects a challenger;
- passing regression metadata allows promotion when score improves;
- diagnostics contain prompt decisions;
- rollback restores the previous baseline;
- schema/format failures route to Prompt Evolution before LoRA.
