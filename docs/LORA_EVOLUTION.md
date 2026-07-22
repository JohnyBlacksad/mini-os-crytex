# Crytex LoRA Evolution

LoRA Evolution is the backend loop that teaches each role how to improve while
also teaching it what not to repeat. Crytex is autonomous: no human labels are
required during normal operation, so the system must preserve positive output,
negative output, critic feedback, failure type, RAG evidence, model id, prompt
version, reward, and benchmark evidence.

## CLI

```powershell
crytex lora status coder-python --json
crytex lora dataset build coder-python --preference --json
crytex lora dataset inspect coder-python --json
crytex lora dataset stats coder-python --json
crytex lora train coder-python --objective dpo --role coder-python --json
crytex lora benchmark coder-python --include-negative --json
crytex lora promote coder-python <adapter-id> --json
crytex lora rollback coder-python --json
```

Development proof commands:

```powershell
cargo run -p crytex-kernel -- prove-lora-dataset --report-path reports\lora-dataset-p8-proof.json
cargo run -p crytex-kernel -- prove-lora-training-objectives --report-path reports\lora-training-objectives-p9-proof.json
cargo run -p crytex-kernel -- prove-lora-quality-gate --report-path reports\lora-quality-gate-p10-proof.json
```

## Backend Loop

1. A role produces an artifact.
2. Critics evaluate it with structured feedback.
3. Accepted output becomes positive evidence.
4. Rejected output becomes negative preference evidence, never an SFT target.
5. Remediation creates a chosen/rejected pair.
6. Dataset builder filters leakage, low-information rows, duplicates, and
   imbalance by failure type.
7. Policy selects SFT, DPO, ORPO, or KTO-style objective when supported.
8. Trainer writes adapter metadata: role, base model, objective, dataset hash,
   metrics, and artifact path.
9. Quality gate runs positive, negative, regression, safety, runtime-application,
   and output-changed checks.
10. Promotion activates only when every gate passes; rollback removes failed
    challenger artifacts and restores the previous active adapter.

## Role Separation

Datasets are role-scoped. `coder-python` must not train from `qa` examples,
critics must not train from orchestrator artifacts, and analyst quality is
measured separately from code generation quality. This keeps the adapter useful
for the role that will actually load it.

Each task records `prompt_version_id`, `model_id`, `rag_evidence_ids`, and
`lora_adapter_id`. This makes quality attribution possible: if the failure came
from bad context, policy fixes RAG; if it came from schema drift, Prompt
Evolution runs first; if it is repeated role skill failure, LoRA training is the
right action.

## Diagnostics

LoRA diagnostics must include dataset hash, role, accepted/rejected pair counts,
failure-type balance, leakage filtering, low-information filtering, training
objective, backend support, adapter artifact validation, benchmark gates,
runtime adapter application proof, output-changed proof, promotion decision, and
rollback decision.

## Troubleshooting

- If a backend does not support a requested objective, the trainer returns typed
  `unsupported` instead of pretending training happened.
- If the adapter directory lacks `adapter_config.json`, `adapter_metadata.json`,
  or `adapter_model.safetensors`, the training job fails and cannot promote.
- If negative benchmark patterns worsen, promotion is rejected even if positive
  benchmark improves.
- If output does not change, require runtime diagnostics proving adapter
  application before accepting the gate.
- If a crash interrupts training, run `crytex diag storage-recovery --json`.

The design follows preference-learning practice: chosen/rejected pairs teach the
model comparative behavior, while regression and safety suites prevent local
improvement from damaging existing capability.
