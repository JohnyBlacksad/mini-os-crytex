# Runtime / Model Matrix

Crytex treats backend support as a typed contract, not a best-effort guess.
Every backend reports `supported`, `partial`, or `unsupported` per capability
with reasons that can be shown by CLI, doctor, diagnostics, and acceptance
proofs.

## Commands

```powershell
crytex models list --json
crytex models add --id qwen --repo owner/model --filename model.gguf --backend mistralrs
crytex models download --id qwen --activate --backend-id local
crytex models activate --id qwen --backend-id local
crytex models prove --id qwen --backend local --report-path reports\model-runtime.json
crytex diag probe-runtime-matrix --json --report-path reports\runtime-model-matrix-p12-proof.json
```

Current development binary:

```powershell
cargo run -p crytex-kernel -- diag probe-runtime-matrix --json --report-path reports\runtime-model-matrix-p12-proof.json
```

## Capability Truth

| Backend | Overall | Generation | Embeddings | Rerank | Runtime LoRA | LoRA training | CUDA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ollama | partial | supported | supported | unsupported | unsupported | unsupported | partial |
| Mistral GGUF CPU/CUDA | supported | supported | unsupported | unsupported | supported | supported | partial |
| ONNX | partial | unsupported | supported | supported | unsupported | unsupported | partial |
| OpenAI-compatible | partial | supported | supported | unsupported | unsupported | unsupported | unsupported |
| Anthropic | partial | supported | unsupported | unsupported | unsupported | unsupported | unsupported |

## Rules

- LoRA training and runtime adapter application are Crytex-local capabilities.
  They are supported through the local Mistral/GGUF path and reported
  unsupported for remote HTTP backends.
- ONNX is a RAG backend for embeddings and reranking. It is not a text
  generation backend.
- Ollama can generate, chat, embed, list, and pull models through its HTTP API,
  but Crytex does not hot-swap LoRA adapters inside Ollama.
- OpenAI-compatible providers may support chat, embeddings, and model listing
  when the provider implements compatible endpoints. Download, CUDA placement,
  and LoRA adapter application remain provider-side and are unsupported in
  Crytex.
- Anthropic is generation/chat only in Crytex. The configured model id is used
  as inventory when public model discovery is unavailable.
- `trash/crytex-inference-trtllm` remains a future optional module until it is
  promoted into `crates/` with CI, toolchain detection, and runtime probes.

## Doctor Preflight

Doctor includes CUDA/toolchain preflight as typed diagnostics:

- `nvidia-smi` availability;
- `nvcc --version` or CUDA runtime library visibility;
- GPU-required mode fails if no GPU is visible;
- optional GPU mode reports warning/CPU fallback instead of crashing.

## References

- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
- mistral.rs: https://github.com/EricLBuehler/mistral.rs
- ONNX Runtime CUDA execution provider: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- OpenAI API reference: https://platform.openai.com/docs/api-reference
- Anthropic Messages API: https://docs.anthropic.com/en/api/messages
