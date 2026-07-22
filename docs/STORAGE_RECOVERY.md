# Storage / Recovery Backend Contract

Crytex must survive crashes, updates, interrupted long-running work, partial
downloads, and Windows file locking without silently corrupting project state.

## Command

```powershell
crytex diag storage-recovery --json --report-path reports\storage-recovery-p14-proof.json
```

Human output summarizes the gate status. JSON output is the durable proof
artifact and includes:

- `schema_version`: current storage contract version.
- `migration_plan`: ordered migration versions from an old database fixture.
- `backup`: backup-before-migrate policy.
- `export_import`: portable export/import manifest and checksum policy.
- `run_resume`: task status recovery map.
- `training_resume`: interrupted LoRA training recovery map.
- `model_download_resume`: partial download resume decision.
- `index_rebuild`: staging plus atomic-swap rebuild plan.
- `windows_lock_policy`: exclusive writer and shared reader commands.
- `corrupt_adapter`: adapter promotion guard.
- `gates`: typed pass/fail evidence.

## Recovery Rules

Schema migrations are versioned with SQLite `PRAGMA user_version`. Older
databases are upgraded through ordered migration steps and receive a backup
before schema mutation.

Interrupted runs do not mark work as done. `InProgress` tasks return to `Ready`;
`Review` and `Remediation` remain visible for critic/remediation workflow; terminal
states remain terminal.

Interrupted training resumes only when the adapter artifact is structurally
valid. Missing or corrupt adapter output marks the training job `failed`; it is
never promoted.

Interrupted model downloads resume from the partial byte count when possible.
Partial files are not registered as downloaded models until size/checksum
validation succeeds.

Index rebuild uses staging state and atomic swap semantics, so agents never
consume a half-built RAG index.

Windows concurrent CLI policy uses `.crytex.lock`: mutating commands are
exclusive, read-only diagnostics and projections may run as shared readers, and
stale locks are typed diagnostics rather than panics.

## References

- [SQLite PRAGMA user_version](https://www.sqlite.org/pragma.html#pragma_user_version)
- [SQLite Backup API](https://www.sqlite.org/backup.html)
- [Windows file locking](https://learn.microsoft.com/windows/win32/fileio/locking-and-unlocking-byte-ranges-in-files)
