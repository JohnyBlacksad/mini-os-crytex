# Sandbox / Security

Crytex must stay autonomous without becoming unsafe. The sandbox/security
contract blocks untrusted project instructions, restricts tool capabilities,
keeps filesystem access inside allowed roots, audits tool calls, and converts
security failures into negative learning signals for the relevant role.

## Commands

```powershell
crytex sandbox doctor --json
crytex sandbox prove --json --report-path reports\sandbox-security-p13-proof.json
crytex security prove --malicious-rag-fixture --json --report-path reports\security-p13-proof.json
```

Current development binary:

```powershell
cargo run -p crytex-kernel -- sandbox prove --json --report-path reports\sandbox-security-p13-proof.json
cargo run -p crytex-kernel -- security prove --malicious-rag-fixture --json --report-path reports\security-p13-proof.json
```

## Tool Permissions

Tool execution is capability-scoped:

| Permission | Capability |
| --- | --- |
| file read/search | `READ` |
| file write | `WRITE` |
| process execution | `SHELL` |
| network access | `NETWORK` |
| git mutation/read tools | `GIT` |

Tools must fail with typed `Forbidden` errors when the caller lacks the required
capability. Search is intentionally read-scoped.

## Sandbox Matrix

| Backend | Status | Isolation |
| --- | --- | --- |
| Docker | supported/partial by daemon availability | ephemeral container, network denied by default, `cap_drop=ALL`, `no-new-privileges`, read-only rootfs, resource limits |
| WASI | supported | Wasmtime fuel, memory limits, closed stdin, bounded stdout/stderr, explicit preopened dirs |
| Host | partial | argv-only execution, cleared environment, timeout, path sandbox, capability guardrails |

Host mode is a fallback for trusted local development. Docker or WASI should be
preferred for untrusted execution.

## RAG Prompt Injection

Project documents are treated as untrusted data. The scanner detects prompt
injection patterns such as attempts to override previous/system instructions or
force tool use. Malicious document findings are preserved in diagnostics and the
content is blocked or wrapped according to security config before it reaches an
agent prompt.

## Audit And Learning

All agent tool calls are wrapped by `AuditedToolService`. The audit entry stores:

- task id;
- project id;
- agent;
- tool name;
- arguments;
- success/error result;
- duration;
- trace id.

Security failures are routed into role-scoped negative examples. The rejected
side is stored as `rejected_output`; it is not used as an SFT target.

## Proof Gates

The P13 proof artifact requires:

- file/process/network/git/search permissions are enforced;
- dot-dot and absolute path traversal are blocked;
- malicious RAG fixture is detected as prompt injection;
- Docker/WASI/host posture is reported;
- tool calls are audited;
- security failure becomes a negative example for the security role.

## References

- OWASP Top 10 for LLM Applications: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- NIST Generative AI Profile: https://airc.nist.gov/AI_RMF_Knowledge_Base/GenAI
- Docker security: https://docs.docker.com/engine/security/
