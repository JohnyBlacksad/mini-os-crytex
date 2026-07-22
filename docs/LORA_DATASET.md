# LoRA Dataset

Crytex learns both what to do and what not to repeat. The LoRA dataset layer
therefore stores positive SFT targets and negative preference sides separately.

## Row Contract

Each `TrainingExample` preserves:

- role, such as `coder-python`, `qa`, `critic-coder`, or `orchestrator`;
- task kind;
- prompt version id;
- model id;
- RAG evidence ids;
- input text;
- accepted output;
- rejected output;
- critic feedback;
- failure type;
- reward.

`accepted_output` is the positive side and mirrors the SFT `output_text`.
`rejected_output` is the negative side of the preference pair and must never be
used as an SFT target.

## Remediation Pairing

When a remediation task completes, Crytex reads `payload.remediation_for` and
links the accepted remediation output with the rejected parent task output:

```json
{
  "accepted_output": "fixed implementation with tests",
  "rejected_output": "implementation without tests",
  "critic_feedback": "missing pytest coverage",
  "failure_type": "missing-tests"
}
```

This creates a chosen/rejected pair for preference-style training while keeping
the negative side out of supervised target rows.

## Role Scope

Datasets are built per role. `coder-python` examples do not train `qa`,
`critic-coder`, `orchestrator`, or another role adapter. This keeps each
adapter specialized for the behavior it is supposed to improve.

## Dataset Diagnostics

Dataset reports include:

- total examples;
- positive examples;
- negative examples;
- chosen/rejected pair count;
- examples per failure type;
- balancing targets per failure type;
- leakage diagnostics for duplicate tasks and duplicate accepted outputs;
- low-information filtering for tiny input/output/negative sides.

## CLI

Production contract:

```powershell
crytex lora dataset build coder-python --preference --json
crytex lora dataset inspect coder-python --json
crytex lora dataset stats coder-python --json
crytex prove lora-dataset --report-path reports\lora-dataset-p8-proof.json
```

Current development binary:

```powershell
cargo run -p crytex-kernel -- lora dataset build coder-python --preference --json
cargo run -p crytex-kernel -- lora dataset inspect coder-python --json
cargo run -p crytex-kernel -- lora dataset stats coder-python --json
cargo run -p crytex-kernel -- prove-lora-dataset --report-path reports\lora-dataset-p8-proof.json
cargo run -p crytex-kernel -- prove-lora-training-objectives --report-path reports\lora-training-objectives-p9-proof.json
```

## Proof

`prove-lora-dataset` emits a deterministic JSON artifact proving:

- accepted outputs are positive targets;
- rejected outputs are negative sides only;
- datasets are role-scoped;
- failure-type balancing is calculated;
- leakage is detected;
- low-information rows are reported.

The dataset is consumed by objective-aware trainers described in
[LORA_TRAINING_OBJECTIVES.md](LORA_TRAINING_OBJECTIVES.md).
