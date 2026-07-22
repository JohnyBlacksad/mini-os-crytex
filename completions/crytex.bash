# bash completion for crytex
_crytex() {
    local cur prev commands
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    commands="doctor project index rag token-economy goal plan kanban run review diag models security prompts lora evolution bench sandbox backend-acceptance prove"
    case "$prev" in
        crytex|crytex-kernel)
            COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
            return 0
            ;;
        prove)
            COMPREPLY=( $(compgen -W "kernel-e2e hf-model hf-runtime-matrix rag-full kanban-projection token-economy orchestrator-quality agent-swarm-lora-routing lora-live-e2e lora-evolution-loop lora-hot-swap lora-candle-learning lora-real-model lora-real-quality-gate release-gate" -- "$cur") )
            return 0
            ;;
        diag)
            COMPREPLY=( $(compgen -W "export probe-runtime-matrix storage-recovery" -- "$cur") )
            return 0
            ;;
    esac
}
complete -F _crytex crytex
complete -F _crytex crytex-kernel
