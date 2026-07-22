# LoRA Quality Gate

Crytex promotes a LoRA adapter only when quality improves and the runtime proves
the adapter was actually applied.

## Required Gates

Every accepted LoRA benchmark decision must include all required typed gates:

- `positive_benchmark`: the agent solves correct held-out tasks better.
- `negative_benchmark`: the agent repeats known bad patterns less often.
- `regression_benchmark`: old skills do not regress.
- `safety_benchmark`: prompt-injection and tool-misuse behavior does not worsen.
- `runtime_application`: runtime diagnostics prove the adapter was loaded/applied.
- `output_changed`: adapted behavior differs from the baseline.

`accepted=true` is not enough. `LoraEvolutionService` rejects and rolls back any
challenger whose benchmark decision lacks one of these gates or marks it failed.

## Promotion Policy

Promotion happens only after:

- training succeeds;
- adapter artifact validates;
- metric thresholds pass;
- benchmark gate returns `accepted=true`;
- all six quality gates pass;
- runtime registration succeeds.

The promoted adapter metrics include the benchmark metadata and the typed
quality-gate evidence.

## Rollback Policy

Failed challengers are not registered as active adapters. Their artifact
directory is removed, the training job is marked `rolled_back`, and the previous
active adapter remains active.

## CLI

```powershell
crytex prove lora-quality-gate --report-path reports\lora-quality-gate-p10-proof.json
```

Current development binary:

```powershell
cargo run -p crytex-kernel -- prove-lora-quality-gate --report-path reports\lora-quality-gate-p10-proof.json
```

## Proof

`prove-lora-quality-gate` emits a deterministic JSON artifact proving:

- positive benchmark gate;
- negative benchmark gate;
- regression benchmark gate;
- safety benchmark gate;
- runtime application proof;
- output changed proof;
- promotion requires all gates;
- rollback removes failed challenger artifact;
- rollback preserves the active adapter.
