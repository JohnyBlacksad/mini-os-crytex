use crate::crytex_cli_commands::Commands;

/// Proof commands are explicit, potentially expensive gates. Keeping this
/// classifier outside `main.rs` prevents proof-only behavior from leaking into
/// everyday command routing.
pub fn is_proof_command(command: &Commands) -> bool {
    matches!(
        command,
        Commands::Prove { .. }
            | Commands::ProveHfModel { .. }
            | Commands::ProveHfRuntimeMatrix { .. }
            | Commands::ProveKernelE2e { .. }
            | Commands::ProveLoraLiveE2e { .. }
            | Commands::ProveLoraEvolutionLoop { .. }
            | Commands::ProveLoraHotSwap { .. }
            | Commands::ProveLoraCandleLearning { .. }
            | Commands::ProveLoraRealModel { .. }
            | Commands::ProveLoraRealQualityGate { .. }
            | Commands::ProveAgentSwarmLoraRouting { .. }
            | Commands::ProveOrchestratorQualityGate { .. }
            | Commands::ProveRagFull { .. }
            | Commands::ProveKanbanProjection { .. }
            | Commands::ProveTokenEconomy { .. }
            | Commands::ProveRoleQualityContracts { .. }
            | Commands::ProvePromptEvolution { .. }
            | Commands::ProveLoraDataset { .. }
            | Commands::ProveLoraTrainingObjectives { .. }
            | Commands::ProveLoraQualityGate { .. }
            | Commands::ProveEvolutionPolicy { .. }
            | Commands::ProveReleaseGate { .. }
            | Commands::BackendAcceptance { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn classifier_separates_proof_from_everyday_commands() {
        let proof = Commands::ProveRagFull { report_path: None };
        let everyday = Commands::ListProjects;
        let acceptance = Commands::BackendAcceptance {
            full: true,
            json: true,
            deterministic: true,
            runtime: crate::crytex_cli_commands::AcceptanceRuntimeMode::Deterministic,
            path: None,
            name: "Backend Acceptance".into(),
            goal: "prove backend".into(),
            live_model: "qwen3.5:9b".into(),
            live_url: "http://localhost:11434".into(),
            report_path: None,
        };

        assert!(is_proof_command(&proof));
        assert!(is_proof_command(&acceptance));
        assert!(!is_proof_command(&everyday));
    }

    #[test]
    fn expensive_lora_hot_swap_is_proof_only() {
        let command = Commands::ProveLoraHotSwap {
            gguf_path: None,
            adapter_a_path: PathBuf::from("a"),
            adapter_b_path: PathBuf::from("b"),
            adapter_a_id: "a".into(),
            adapter_b_id: "b".into(),
            context_size: 64,
            gpu_layers: None,
            max_tokens: 8,
            generation_timeout_secs: 45,
            report_path: None,
        };

        assert!(is_proof_command(&command));
    }

    #[test]
    fn token_economy_is_proof_only() {
        let command = Commands::ProveTokenEconomy {
            backend: "ollama".into(),
            model: "qwen3.5:9b".into(),
            context_window: 32_768,
            expected_completion_tokens: 512,
            report_path: None,
        };

        assert!(is_proof_command(&command));
    }

    #[test]
    fn kanban_projection_is_proof_only() {
        let command = Commands::ProveKanbanProjection { report_path: None };

        assert!(is_proof_command(&command));
    }

    #[test]
    fn role_quality_contracts_is_proof_only() {
        let command = Commands::ProveRoleQualityContracts { report_path: None };

        assert!(is_proof_command(&command));
    }

    #[test]
    fn prompt_evolution_is_proof_only() {
        let command = Commands::ProvePromptEvolution { report_path: None };

        assert!(is_proof_command(&command));
    }

    #[test]
    fn lora_dataset_is_proof_only() {
        let command = Commands::ProveLoraDataset { report_path: None };

        assert!(is_proof_command(&command));
    }

    #[test]
    fn lora_training_objectives_is_proof_only() {
        let command = Commands::ProveLoraTrainingObjectives { report_path: None };

        assert!(is_proof_command(&command));
    }

    #[test]
    fn lora_quality_gate_is_proof_only() {
        let command = Commands::ProveLoraQualityGate { report_path: None };

        assert!(is_proof_command(&command));
    }

    #[test]
    fn evolution_policy_is_proof_only() {
        let command = Commands::ProveEvolutionPolicy { report_path: None };

        assert!(is_proof_command(&command));
    }

    #[test]
    fn release_gate_is_proof_only() {
        let command = Commands::ProveReleaseGate { report_path: None };

        assert!(is_proof_command(&command));
    }

    #[test]
    fn nested_prove_group_is_proof_only() {
        let command = Commands::Prove {
            command: crate::crytex_cli_commands::ProveCommands::ReleaseGate { report_path: None },
        };

        assert!(is_proof_command(&command));
    }
}
