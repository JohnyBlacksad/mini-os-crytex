# Example: Analyst Role

This example shows how the `analyst` role uses RAG evidence to produce a
traceable analysis artifact.

## Goal

Analyze project telemetry and documentation to explain why a backend proof gate
failed, then propose the next highest-leverage fix.

## CLI

```powershell
crytex rag search "latest failed backend acceptance diagnostics" --rerank --explain --json
crytex goal submit "Analyze failing backend acceptance gate and propose remediation"
crytex plan show --json
crytex run start --json
crytex kanban show --role analyst --json
crytex diag export --run latest --out reports\analyst-run.json
```

## Role Contract

Role: `analyst`.

Expected artifact:

- question answered;
- evidence ids;
- assumptions;
- findings;
- confidence;
- recommended next tasks;
- risks and missing data.

Analyst output must cite RAG evidence ids and distinguish observation from
inference. Weak analysis becomes a negative example for `analyst`; missing
context routes to RAG rather than LoRA.

## Diagnostics

Inspect selected RAG chunks, token budget decisions, prompt version, model id,
task result, critic decision, and reward.

## Troubleshooting

- If the analysis hallucinates files or metrics, check RAG selection and
  evidence ids.
- If the artifact lacks confidence or assumptions, route to Prompt Evolution.
- If good context repeatedly produces shallow reasoning, route to LoRA training
  for `analyst`.
