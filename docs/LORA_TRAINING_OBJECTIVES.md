# LoRA Training Objectives

Crytex trains role adapters against an explicit objective, not a vague
`train(kind)` operation.

## Objectives

- `sft`: supervised fine-tuning against `accepted_output` / `output_text`.
- `dpo`: pairwise preference optimization over `accepted_output` and `rejected_output`.
- `orpo`: reference-free odds-ratio preference optimization over chosen/rejected pairs.
- `kto`: utility-style optimization from positive and negative feedback.

Preference objectives validate that every training row contains both the chosen
positive side and the rejected negative side. Rejected output is never used as
the SFT target.

## Trainer Boundary

`LoraTrainer` provides:

- `backend_name()`;
- `supports_objective(objective)`;
- `train(examples, config, output_dir)`.

Backends that cannot train an objective return a typed
`LoraTrainingError::UnsupportedObjective` instead of silently falling back.

Current backend behavior:

- Candle: real SFT path, typed unsupported for DPO/ORPO/KTO.
- Mistral mock: deterministic SFT and preference objective path for CI/proofs.
- Kernel deterministic proof trainer: deterministic SFT/DPO/ORPO/KTO contract proof.

## Adapter Metadata

Every adapter directory must contain:

- `adapter_config.json`;
- `adapter_model.safetensors`;
- `adapter_metadata.json`.

`adapter_metadata.json` stores:

- role;
- base model;
- objective;
- dataset hash.

The dataset hash is calculated from stable training-example fields and allows a
promoted adapter to be traced back to the exact role dataset it learned from.

## Job State

Training jobs use typed states:

- `queued`;
- `running`;
- `failed`;
- `promoted`;
- `rolled_back`.

`promoted` means the adapter passed training, artifact validation, validation
metrics, optional benchmark gate, registration, and runtime registration.

## CLI

```powershell
crytex lora train coder-python --objective sft
crytex lora train coder-python --objective dpo --role coder-python
crytex lora train coder-python --objective orpo --role coder-python
crytex lora train coder-python --objective kto --role coder-python
crytex prove lora-training-objectives --report-path reports\lora-training-objectives-p9-proof.json
```

Current development binary:

```powershell
cargo run -p crytex-kernel -- lora train coder-python --objective dpo --role coder-python
cargo run -p crytex-kernel -- prove-lora-training-objectives --report-path reports\lora-training-objectives-p9-proof.json
```

## Proof

`prove-lora-training-objectives` emits a deterministic JSON artifact proving:

- objective-aware trainer trait;
- SFT/DPO/ORPO/KTO support reporting;
- typed unsupported objective error;
- adapter metadata with role, base model, objective, and dataset hash;
- adapter directory validation;
- job-state vocabulary for queued/running/failed/promoted/rolled_back.
