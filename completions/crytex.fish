# fish completion for crytex
complete -c crytex -f
complete -c crytex -n "__fish_use_subcommand" -a "doctor project index rag token-economy goal plan kanban run review diag models security prompts lora evolution bench sandbox backend-acceptance prove"
complete -c crytex -n "__fish_seen_subcommand_from prove" -a "kernel-e2e hf-model hf-runtime-matrix rag-full kanban-projection token-economy orchestrator-quality agent-swarm-lora-routing lora-live-e2e lora-evolution-loop lora-hot-swap lora-candle-learning lora-real-model lora-real-quality-gate release-gate"
complete -c crytex -n "__fish_seen_subcommand_from diag" -a "export probe-runtime-matrix storage-recovery"
complete -c crytex-kernel -w crytex
