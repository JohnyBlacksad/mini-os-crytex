use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "crytex-diag")]
#[command(about = "Lightweight Crytex diagnostic CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Print configured inference backends and lightweight diagnostic support
    ListBackends,
    /// Validate diagnostic configuration before running expensive probes
    Doctor {
        #[arg(short, long)]
        backend: Vec<String>,
        #[arg(long)]
        require_gpu: bool,
    },
    /// Run baseline and LoRA runtime probe matrix without booting the full kernel
    ProbeRuntimeMatrix {
        #[arg(short, long)]
        id: String,
        #[arg(short, long)]
        backend: Vec<String>,
        #[arg(short, long)]
        model: Option<String>,
        #[arg(long)]
        lora: Vec<String>,
        #[arg(long)]
        trace_id: Option<String>,
        #[arg(long)]
        report_dir: Option<PathBuf>,
        #[arg(long, default_value = "16")]
        max_tokens: usize,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::ListBackends => {
            let config = crytex_diag::DiagConfig::load();
            print_pretty_json(&crytex_diag::list_backends(&config));
        }
        Commands::Doctor {
            backend,
            require_gpu,
        } => {
            let config = crytex_diag::DiagConfig::load();
            let report = crytex_diag::doctor_with_api_checks_and_options(
                &config,
                &backend,
                crytex_diag::DoctorOptions { require_gpu },
            )
            .await;
            print_pretty_json(&report);
            if !report.passed {
                std::process::exit(2);
            }
        }
        Commands::ProbeRuntimeMatrix {
            id,
            backend,
            model,
            lora,
            trace_id,
            report_dir,
            max_tokens,
        } => {
            let config = crytex_diag::DiagConfig::load();
            let backend_ids = crytex_diag::configured_backend_ids(&config, &backend);
            if backend_ids.is_empty() {
                eprintln!("No backend was provided and no default backend is configured");
                std::process::exit(1);
            }

            let first_backend_id = backend_ids.first().expect("backend ids were checked");
            let model_name =
                crytex_diag::runtime_model_name(&config, first_backend_id, model.as_deref(), &id);
            let entries = crytex_diag::build_runtime_matrix_entries(
                &backend_ids,
                &lora,
                &id,
                &model_name,
                max_tokens,
            );
            let backends = crytex_diag::create_remote_backends(&config, &backend_ids)
                .unwrap_or_else(|error| {
                    eprintln!("Failed to create lightweight inference backends: {error}");
                    std::process::exit(1);
                });
            let trace_id = trace_id.unwrap_or_else(|| format!("diag-{}", current_timestamp_ms()));
            let report = crytex_diag::run_runtime_matrix_with_config(
                trace_id,
                entries,
                &backends,
                Some(&config),
            )
            .await;
            let report_dir = report_dir
                .unwrap_or_else(|| config.paths.data_dir.join("reports").join("runtime-matrix"));
            let report_path = crytex_diag::write_report_pretty_json(&report, report_dir)
                .unwrap_or_else(|error| {
                    eprintln!("Failed to write runtime matrix report: {error}");
                    std::process::exit(1);
                });
            eprintln!("Runtime matrix report written to {}", report_path.display());
            print_pretty_json(&report);
            if !report.passed {
                std::process::exit(2);
            }
        }
    }
}

fn print_pretty_json<T: serde::Serialize>(value: &T) {
    let json = serde_json::to_string_pretty(value).unwrap_or_else(|error| {
        eprintln!("Failed to serialize diagnostic report: {error}");
        std::process::exit(1);
    });
    println!("{json}");
}

fn current_timestamp_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}
