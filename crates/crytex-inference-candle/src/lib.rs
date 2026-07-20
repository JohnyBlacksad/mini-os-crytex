//! Candle-based LoRA trainer.
//!
//! This crate provides a pure-Rust LoRA training backend using
//! [Candle](https://github.com/huggingface/candle).  The current implementation
//! fine-tunes a Llama-style causal language model with low-rank adapters on the
//! attention `q_proj` and `v_proj` matrices.  When no pretrained base model is
//! provided, a tiny built-in transformer is trained instead, which keeps the
//! workspace tests green without requiring multi-gigabyte downloads.

use std::collections::HashMap;
use std::path::Path;

use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap, loss};
use crytex_core::models::TrainingExample;
use crytex_core::services::{
    LoraMetrics, LoraTrainer, LoraTrainingConfig, LoraTrainingError, LoraTrainingResult,
};
use tracing::{debug, info};
use ulid::Ulid;

pub mod model;
pub mod tokenizer;

use model::{LoraCausalLM, ModelConfig, load_quantized_base, select_device};
use tokenizer::Tokenizer;

const MIN_EXAMPLES: usize = 2;
const EOS_TOKEN: u32 = 0;

/// Candle-based implementation of [`LoraTrainer`].
#[derive(Debug, Default, Clone, Copy)]
pub struct CandleLoraTrainer;

impl CandleLoraTrainer {
    /// Create a new Candle LoRA trainer.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl LoraTrainer for CandleLoraTrainer {
    async fn train(
        &self,
        examples: Vec<TrainingExample>,
        config: LoraTrainingConfig,
        output_dir: &Path,
    ) -> Result<LoraTrainingResult, LoraTrainingError> {
        if examples.len() < MIN_EXAMPLES {
            return Err(LoraTrainingError::NotEnoughExamples(
                examples.len(),
                MIN_EXAMPLES,
            ));
        }

        tokio::fs::create_dir_all(output_dir).await?;

        let device = select_device().map_err(|e| LoraTrainingError::Backend(e.to_string()))?;
        let (train, val) = split_train_val(examples, config.validation_ratio);

        #[allow(unused_mut)]
        let mut model_cfg = match &config.base_model_path {
            Some(path) if is_gguf_path(path) => {
                let gguf_path = resolve_gguf_path(path)?;
                let cfg = ModelConfig::from_gguf(&gguf_path)
                    .map_err(|e| LoraTrainingError::Backend(e.to_string()))?;
                cfg.with_lora(config.rank, config.alpha, config.target_modules.clone())
            }
            Some(path) => {
                let cfg = ModelConfig::from_pretrained_dir(path)
                    .map_err(|e| LoraTrainingError::Backend(e.to_string()))?;
                cfg.with_lora(config.rank, config.alpha, config.target_modules.clone())
            }
            None => ModelConfig::tiny_for_tests(config.rank, config.alpha, config.max_seq_len)
                .with_lora(config.rank, config.alpha, config.target_modules.clone()),
        };

        #[cfg(feature = "flash-attn")]
        if model_cfg.dtype == DType::F16 || model_cfg.dtype == DType::BF16 {
            model_cfg.use_flash_attn = true;
        }

        let vb = match &config.base_model_path {
            Some(path) if is_gguf_path(path) => {
                let gguf_path = resolve_gguf_path(path)?;
                load_quantized_base(&gguf_path, &device, model_cfg.dtype)
                    .map_err(|e| LoraTrainingError::Backend(e.to_string()))?
            }
            Some(path) => {
                let files = safetensor_files(path)?;
                unsafe {
                    VarBuilder::from_mmaped_safetensors(&files, model_cfg.dtype, &device)
                        .map_err(|e| LoraTrainingError::Backend(e.to_string()))?
                }
            }
            None => {
                let varmap = VarMap::new();
                VarBuilder::from_varmap(&varmap, model_cfg.dtype, &device)
            }
        };

        let model = LoraCausalLM::load(vb, &model_cfg)
            .map_err(|e| LoraTrainingError::Backend(e.to_string()))?;

        let tokenizer =
            tokenizer::build_tokenizer(model.vocab_size(), config.tokenizer_path.as_deref())
                .map_err(|e| LoraTrainingError::Backend(e.to_string()))?;

        let params = ParamsAdamW {
            lr: config.learning_rate,
            ..ParamsAdamW::default()
        };
        let mut optimizer = AdamW::new(model.lora_vars(), params)
            .map_err(|e| LoraTrainingError::Backend(e.to_string()))?;

        let mut last_train_loss = f64::NAN;
        for epoch in 0..config.epochs {
            let mut epoch_loss = 0.0f64;
            let mut count = 0usize;
            for ex in &train {
                if let Some(loss) = train_step(&model, &*tokenizer, ex, &device, &model_cfg)? {
                    optimizer
                        .backward_step(&loss)
                        .map_err(|e| LoraTrainingError::Backend(e.to_string()))?;
                    epoch_loss += loss.to_scalar::<f64>().unwrap_or(0.0);
                    count += 1;
                }
            }
            last_train_loss = if count > 0 {
                epoch_loss / count as f64
            } else {
                0.0
            };
            debug!(
                epoch,
                train_loss = last_train_loss,
                "candle causal-lm epoch"
            );
        }

        let val_loss = evaluate(&model, &*tokenizer, &val, &device, &model_cfg)?;
        info!(
            train_loss = last_train_loss,
            val_loss, "Candle LoRA training finished"
        );

        let adapter_id = format!("candle-lora-{}", Ulid::new());
        let adapter_path = output_dir.join(&adapter_id);
        tokio::fs::create_dir_all(&adapter_path).await?;

        let tensors: HashMap<String, Tensor> = model.lora_tensors();
        candle_core::safetensors::save(&tensors, adapter_path.join("adapter_model.safetensors"))
            .map_err(|e| LoraTrainingError::AdapterSerialization(e.to_string()))?;
        write_adapter_config(&adapter_path, &config).await?;

        let average_reward = train.iter().map(|e| e.reward).sum::<f64>() / train.len() as f64;

        Ok(LoraTrainingResult {
            adapter_id,
            adapter_path,
            metrics: LoraMetrics {
                train_loss: last_train_loss,
                validation_loss: val_loss,
                average_reward,
            },
        })
    }
}

async fn write_adapter_config(
    adapter_path: &Path,
    config: &LoraTrainingConfig,
) -> Result<(), LoraTrainingError> {
    let base_model = config
        .base_model_path
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "crytex-candle-tiny".to_string());
    let payload = serde_json::json!({
        "peft_type": "LORA",
        "base_model_name_or_path": base_model,
        "r": config.rank,
        "lora_alpha": config.alpha,
        "target_modules": config.target_modules,
        "task_type": "CAUSAL_LM",
        "crytex_trainer": "candle",
    });
    tokio::fs::write(
        adapter_path.join("adapter_config.json"),
        serde_json::to_vec_pretty(&payload)
            .map_err(|e| LoraTrainingError::AdapterSerialization(e.to_string()))?,
    )
    .await?;
    Ok(())
}

fn is_gguf_path(path: &Path) -> bool {
    if path.is_file() {
        return path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
    }
    if path.is_dir() {
        return std::fs::read_dir(path)
            .ok()
            .and_then(|mut rd| {
                rd.find(|e| {
                    e.as_ref()
                        .map(|e| {
                            e.path()
                                .extension()
                                .and_then(|s| s.to_str())
                                .map(|s| s.eq_ignore_ascii_case("gguf"))
                                .unwrap_or(false)
                        })
                        .unwrap_or(false)
                })
            })
            .is_some();
    }
    false
}

fn resolve_gguf_path(path: &Path) -> Result<std::path::PathBuf, LoraTrainingError> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    let mut files: Vec<_> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    files.into_iter().next().ok_or_else(|| {
        LoraTrainingError::Backend(format!("no .gguf files found in {}", path.display()))
    })
}

fn safetensor_files(path: &Path) -> Result<Vec<std::path::PathBuf>, LoraTrainingError> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut files: Vec<_> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|s| s == "safetensors")
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(LoraTrainingError::Backend(format!(
            "no .safetensors files found in {}",
            path.display()
        )));
    }
    Ok(files)
}

fn split_train_val(
    examples: Vec<TrainingExample>,
    validation_ratio: f64,
) -> (Vec<TrainingExample>, Vec<TrainingExample>) {
    let val_count = (examples.len() as f64 * validation_ratio.clamp(0.0, 0.5)).ceil() as usize;
    let train_count = examples.len().saturating_sub(val_count.max(1));
    let train: Vec<_> = examples.iter().take(train_count).cloned().collect();
    let val: Vec<_> = examples.iter().skip(train_count).cloned().collect();
    (train, val)
}

fn prepare_tokens(tokenizer: &dyn Tokenizer, ex: &TrainingExample, max_seq_len: usize) -> Vec<u32> {
    let mut tokens = Vec::new();
    tokens.extend(tokenizer.encode(&ex.input_text).unwrap_or_default());
    tokens.extend(tokenizer.encode(&ex.output_text).unwrap_or_default());
    tokens.push(EOS_TOKEN);
    if tokens.len() > max_seq_len {
        tokens.truncate(max_seq_len);
    }
    tokens
}

fn causal_loss(
    model: &LoraCausalLM,
    tokens: &[u32],
    device: &Device,
) -> CandleResult<Option<Tensor>> {
    if tokens.len() < 2 {
        return Ok(None);
    }
    let input_ids = Tensor::new(tokens, device)?.reshape((1, tokens.len()))?;
    let logits = model.forward(&input_ids)?;
    let logits = logits.to_dtype(DType::F32)?;
    let pred_len = tokens.len() - 1;
    let logits = logits.i((.., 0..pred_len, ..))?.contiguous()?;
    let logits = logits.reshape((pred_len, model.vocab_size()))?;
    let targets = Tensor::new(&tokens[1..], device)?;
    let loss = loss::cross_entropy(&logits, &targets)?;
    Ok(Some(loss))
}

fn train_step(
    model: &LoraCausalLM,
    tokenizer: &dyn Tokenizer,
    ex: &TrainingExample,
    device: &Device,
    cfg: &ModelConfig,
) -> Result<Option<Tensor>, LoraTrainingError> {
    let tokens = prepare_tokens(tokenizer, ex, cfg.max_seq_len);
    causal_loss(model, &tokens, device).map_err(|e| LoraTrainingError::Backend(e.to_string()))
}

fn evaluate(
    model: &LoraCausalLM,
    tokenizer: &dyn Tokenizer,
    examples: &[TrainingExample],
    device: &Device,
    cfg: &ModelConfig,
) -> Result<f64, LoraTrainingError> {
    if examples.is_empty() {
        return Ok(0.0);
    }
    let mut total = 0.0f64;
    let mut count = 0usize;
    for ex in examples {
        if let Some(loss) = causal_loss(
            model,
            &prepare_tokens(tokenizer, ex, cfg.max_seq_len),
            device,
        )
        .map_err(|e| LoraTrainingError::Backend(e.to_string()))?
        {
            total += loss.to_scalar::<f64>().unwrap_or(0.0);
            count += 1;
        }
    }
    Ok(if count > 0 { total / count as f64 } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crytex_core::models::TrainingExample;
    use std::path::PathBuf;

    fn example(input: &str, output: &str, reward: f64) -> TrainingExample {
        TrainingExample {
            id: format!("ex-{}", Ulid::new()),
            task_id: "t1".into(),
            project_id: Some("p1".into()),
            prompt_version_id: Some("pv1".into()),
            task_kind: "codegen".into(),
            agent_role: None,
            input_text: input.into(),
            output_text: output.into(),
            reward,
            created_at: 0,
        }
    }

    #[tokio::test]
    async fn candle_lora_trains_and_saves_adapter() {
        let trainer = CandleLoraTrainer::new();
        let output = PathBuf::from(format!(
            "{}/candle-lora-test",
            std::env::temp_dir().to_string_lossy()
        ));
        let _ = tokio::fs::remove_dir_all(&output).await;

        let examples = vec![
            example(
                "Implement add",
                "fn add(a: i32, b: i32) -> i32 { a + b }",
                4.0,
            ),
            example(
                "Implement subtract",
                "fn sub(a: i32, b: i32) -> i32 { a - b }",
                5.0,
            ),
            example(
                "Implement multiply",
                "fn mul(a: i32, b: i32) -> i32 { a * b }",
                4.5,
            ),
            example(
                "Implement divide",
                "fn div(a: i32, b: i32) -> i32 { a / b }",
                4.2,
            ),
        ];

        let config = LoraTrainingConfig {
            rank: 4,
            alpha: 8,
            epochs: 5,
            learning_rate: 1e-2,
            validation_ratio: 0.25,
            max_seq_len: 64,
            ..Default::default()
        };

        let result = trainer.train(examples, config, &output).await.unwrap();

        assert!(result.adapter_path.is_dir());
        assert!(result.adapter_path.join("adapter_config.json").exists());
        assert!(
            result
                .adapter_path
                .join("adapter_model.safetensors")
                .exists()
        );
        assert!(
            result.metrics.train_loss.is_finite(),
            "train loss should be finite"
        );
        assert!(
            result.metrics.validation_loss.is_finite(),
            "validation loss should be finite"
        );
        assert!((result.metrics.average_reward - 4.5).abs() < 0.001);

        let _ = tokio::fs::remove_dir_all(&output).await;
    }

    #[tokio::test]
    async fn candle_lora_rejects_empty_examples() {
        let trainer = CandleLoraTrainer::new();
        let output = PathBuf::from(format!(
            "{}/candle-lora-empty",
            std::env::temp_dir().to_string_lossy()
        ));
        let result = trainer
            .train(vec![], LoraTrainingConfig::default(), &output)
            .await;
        assert!(matches!(
            result,
            Err(LoraTrainingError::NotEnoughExamples(0, _))
        ));
    }

    #[tokio::test]
    async fn candle_lora_loads_pretrained_base_and_trains() {
        let trainer = CandleLoraTrainer::new();
        let base_dir = PathBuf::from(format!(
            "{}/candle-lora-base",
            std::env::temp_dir().to_string_lossy()
        ));
        let output = PathBuf::from(format!(
            "{}/candle-lora-pretrained-test",
            std::env::temp_dir().to_string_lossy()
        ));
        let _ = tokio::fs::remove_dir_all(&base_dir).await;
        let _ = tokio::fs::remove_dir_all(&output).await;
        tokio::fs::create_dir_all(&base_dir).await.unwrap();

        let cfg = ModelConfig::tiny_for_tests(4, 8, 64);
        write_random_base_weights(&base_dir, &cfg);

        let config = LoraTrainingConfig {
            rank: 4,
            alpha: 8,
            epochs: 5,
            learning_rate: 1e-2,
            validation_ratio: 0.25,
            max_seq_len: 64,
            base_model_path: Some(base_dir.clone()),
            ..Default::default()
        };

        let examples = vec![
            example("Write hello", "println!(\"hello\");", 3.0),
            example("Write world", "println!(\"world\");", 4.0),
            example("Write foo", "println!(\"foo\");", 3.5),
            example("Write bar", "println!(\"bar\");", 4.2),
        ];

        let result = trainer.train(examples, config, &output).await.unwrap();
        assert!(result.adapter_path.exists());
        assert!(result.metrics.train_loss.is_finite());
        assert!(result.metrics.validation_loss.is_finite());

        let _ = tokio::fs::remove_dir_all(&base_dir).await;
        let _ = tokio::fs::remove_dir_all(&output).await;
    }

    #[tokio::test]
    async fn candle_lora_generates_after_training() {
        let trainer = CandleLoraTrainer::new();
        let output = PathBuf::from(format!(
            "{}/candle-lora-generate",
            std::env::temp_dir().to_string_lossy()
        ));
        let _ = tokio::fs::remove_dir_all(&output).await;

        let examples = vec![
            example("A", "B", 1.0),
            example("C", "D", 1.0),
            example("E", "F", 1.0),
            example("G", "H", 1.0),
        ];
        let config = LoraTrainingConfig {
            rank: 4,
            alpha: 8,
            epochs: 2,
            learning_rate: 1e-2,
            max_seq_len: 32,
            ..Default::default()
        };

        trainer.train(examples, config, &output).await.unwrap();

        let device = select_device().unwrap();
        let cfg = ModelConfig::tiny_for_tests(4, 8, 32);
        let vb = VarBuilder::from_varmap(&VarMap::new(), cfg.dtype, &device);
        let model = LoraCausalLM::load(vb, &cfg).unwrap();
        let prompt = tokenizer::ByteTokenizer::new(cfg.vocab_size)
            .encode("hi")
            .unwrap();
        let generated = model.generate(&prompt, 5, None, &device).unwrap();
        assert_eq!(generated.len(), prompt.len() + 5);

        let _ = tokio::fs::remove_dir_all(&output).await;
    }

    #[tokio::test]
    async fn candle_lora_loads_adapter_and_generates() {
        let trainer = CandleLoraTrainer::new();
        let base_dir = PathBuf::from(format!(
            "{}/candle-lora-adapter-base",
            std::env::temp_dir().to_string_lossy()
        ));
        let output = PathBuf::from(format!(
            "{}/candle-lora-adapter-out",
            std::env::temp_dir().to_string_lossy()
        ));
        let _ = tokio::fs::remove_dir_all(&base_dir).await;
        let _ = tokio::fs::remove_dir_all(&output).await;
        tokio::fs::create_dir_all(&base_dir).await.unwrap();

        let cfg = ModelConfig::tiny_for_tests(4, 8, 32);
        write_random_base_weights(&base_dir, &cfg);

        let config = LoraTrainingConfig {
            rank: 4,
            alpha: 8,
            epochs: 2,
            learning_rate: 1e-2,
            max_seq_len: 32,
            base_model_path: Some(base_dir.clone()),
            ..Default::default()
        };

        let examples = vec![
            example("A", "B", 1.0),
            example("C", "D", 1.0),
            example("E", "F", 1.0),
            example("G", "H", 1.0),
        ];
        let result = trainer.train(examples, config, &output).await.unwrap();

        let device = select_device().unwrap();
        let files = safetensor_files(&base_dir).unwrap();
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&files, cfg.dtype, &device) }.unwrap();
        let mut model = LoraCausalLM::load(vb, &cfg).unwrap();
        model
            .load_adapter(&result.adapter_path.join("adapter_model.safetensors"))
            .unwrap();

        let prompt = tokenizer::ByteTokenizer::new(cfg.vocab_size)
            .encode("x")
            .unwrap();
        let generated = model.generate(&prompt, 3, None, &device).unwrap();
        assert_eq!(generated.len(), prompt.len() + 3);

        let _ = tokio::fs::remove_dir_all(&base_dir).await;
        let _ = tokio::fs::remove_dir_all(&output).await;
    }

    #[tokio::test]
    async fn candle_lora_adapter_contains_only_lora_tensors() {
        let trainer = CandleLoraTrainer::new();
        let output = PathBuf::from(format!(
            "{}/candle-lora-adapter-only",
            std::env::temp_dir().to_string_lossy()
        ));
        let _ = tokio::fs::remove_dir_all(&output).await;

        let examples = vec![
            example(
                "Write add function",
                "fn add(a: i32, b: i32) -> i32 { a + b }",
                4.0,
            ),
            example(
                "Write sub function",
                "fn sub(a: i32, b: i32) -> i32 { a - b }",
                4.0,
            ),
            example(
                "Write mul function",
                "fn mul(a: i32, b: i32) -> i32 { a * b }",
                4.0,
            ),
            example(
                "Write div function",
                "fn div(a: i32, b: i32) -> i32 { a / b }",
                4.0,
            ),
        ];
        let config = LoraTrainingConfig {
            rank: 4,
            alpha: 8,
            epochs: 2,
            learning_rate: 1e-2,
            max_seq_len: 32,
            ..Default::default()
        };

        let result = trainer.train(examples, config, &output).await.unwrap();
        assert!(result.adapter_path.is_dir());
        assert!(result.adapter_path.join("adapter_config.json").exists());
        let adapter_bytes =
            std::fs::read(result.adapter_path.join("adapter_model.safetensors")).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&adapter_bytes).unwrap();
        let names = tensors.names();

        assert!(!names.is_empty());
        assert!(
            names.iter().all(|name| {
                name.ends_with(".lora_A.weight") || name.ends_with(".lora_B.weight")
            })
        );
        assert!(!names.iter().any(|name| {
            name.contains("embed_tokens")
                || name.contains("lm_head")
                || name.ends_with(".q_proj.weight")
                || name.ends_with(".v_proj.weight")
        }));

        let _ = tokio::fs::remove_dir_all(&output).await;
    }

    #[tokio::test]
    async fn candle_lora_trains_quantized_base() {
        let trainer = CandleLoraTrainer::new();
        let base_dir = PathBuf::from(format!(
            "{}/candle-lora-quantized-base",
            std::env::temp_dir().to_string_lossy()
        ));
        let output = PathBuf::from(format!(
            "{}/candle-lora-quantized-out",
            std::env::temp_dir().to_string_lossy()
        ));
        let _ = tokio::fs::remove_dir_all(&base_dir).await;
        let _ = tokio::fs::remove_dir_all(&output).await;
        tokio::fs::create_dir_all(&base_dir).await.unwrap();

        let cfg = ModelConfig::tiny_for_tests(4, 8, 32);
        write_quantized_base_weights(&base_dir, &cfg);

        let config = LoraTrainingConfig {
            rank: 4,
            alpha: 8,
            epochs: 2,
            learning_rate: 1e-2,
            max_seq_len: 32,
            base_model_path: Some(base_dir.clone()),
            ..Default::default()
        };

        let examples = vec![
            example("A", "B", 1.0),
            example("C", "D", 1.0),
            example("E", "F", 1.0),
            example("G", "H", 1.0),
        ];
        let result = trainer.train(examples, config, &output).await.unwrap();
        assert!(result.adapter_path.exists());
        assert!(result.metrics.train_loss.is_finite());
        assert!(result.metrics.validation_loss.is_finite());

        let _ = tokio::fs::remove_dir_all(&base_dir).await;
        let _ = tokio::fs::remove_dir_all(&output).await;
    }

    fn write_random_base_weights(dir: &Path, cfg: &ModelConfig) {
        let device = Device::Cpu;
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let v = cfg.vocab_size;
        let kv_h = cfg.kv_hidden_size();
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        macro_rules! insert {
            ($name:expr, $shape:expr) => {
                tensors.insert(
                    $name.to_string(),
                    Tensor::randn(0.0f32, 0.02f32, $shape, &device).unwrap(),
                );
            };
        }
        insert!("model.embed_tokens.weight", (v, h));
        insert!("model.norm.weight", (h,));
        insert!("lm_head.weight", (v, h));
        for layer in 0..cfg.num_layers {
            let p = format!("model.layers.{layer}");
            insert!(&format!("{p}.input_layernorm.weight"), (h,));
            insert!(&format!("{p}.post_attention_layernorm.weight"), (h,));
            insert!(&format!("{p}.self_attn.q_proj.weight"), (h, h));
            insert!(&format!("{p}.self_attn.k_proj.weight"), (kv_h, h));
            insert!(&format!("{p}.self_attn.v_proj.weight"), (kv_h, h));
            insert!(&format!("{p}.self_attn.o_proj.weight"), (h, h));
            insert!(&format!("{p}.mlp.gate_proj.weight"), (i, h));
            insert!(&format!("{p}.mlp.up_proj.weight"), (i, h));
            insert!(&format!("{p}.mlp.down_proj.weight"), (h, i));
        }
        candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();

        let config_json = serde_json::json!({
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_layers,
            "num_attention_heads": cfg.num_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "intermediate_size": cfg.intermediate_size,
            "max_position_embeddings": cfg.max_seq_len,
            "rms_norm_eps": cfg.rms_norm_eps,
            "rope_theta": cfg.rope_theta,
        });
        std::fs::write(dir.join("config.json"), config_json.to_string()).unwrap();
    }

    fn write_quantized_base_weights(dir: &Path, cfg: &ModelConfig) {
        use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
        let device = Device::Cpu;
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let v = cfg.vocab_size;
        let kv_h = cfg.kv_hidden_size();

        macro_rules! qtensor {
            ($shape:expr) => {
                QTensor::quantize(
                    &Tensor::randn(0.0f32, 0.02f32, $shape, &device).unwrap(),
                    GgmlDType::Q8_0,
                )
                .unwrap()
            };
        }

        let mut tensors: Vec<(String, QTensor)> = Vec::new();
        tensors.push(("token_embd.weight".into(), qtensor!((v, h))));
        tensors.push(("output.weight".into(), qtensor!((v, h))));
        tensors.push(("output_norm.weight".into(), qtensor!((h,))));
        for layer in 0..cfg.num_layers {
            tensors.push((format!("blk.{layer}.attn_norm.weight"), qtensor!((h,))));
            tensors.push((format!("blk.{layer}.ffn_norm.weight"), qtensor!((h,))));
            tensors.push((format!("blk.{layer}.attn_q.weight"), qtensor!((h, h))));
            tensors.push((format!("blk.{layer}.attn_k.weight"), qtensor!((kv_h, h))));
            tensors.push((format!("blk.{layer}.attn_v.weight"), qtensor!((kv_h, h))));
            tensors.push((format!("blk.{layer}.attn_output.weight"), qtensor!((h, h))));
            tensors.push((format!("blk.{layer}.ffn_gate.weight"), qtensor!((i, h))));
            tensors.push((format!("blk.{layer}.ffn_up.weight"), qtensor!((i, h))));
            tensors.push((format!("blk.{layer}.ffn_down.weight"), qtensor!((h, i))));
        }

        let metadata_values: Vec<(String, gguf_file::Value)> = vec![
            (
                "general.architecture".into(),
                gguf_file::Value::String("llama".into()),
            ),
            ("llama.vocab_size".into(), gguf_file::Value::U32(v as u32)),
            (
                "llama.embedding_length".into(),
                gguf_file::Value::U32(h as u32),
            ),
            (
                "llama.block_count".into(),
                gguf_file::Value::U32(cfg.num_layers as u32),
            ),
            (
                "llama.attention.head_count".into(),
                gguf_file::Value::U32(cfg.num_heads as u32),
            ),
            (
                "llama.attention.head_count_kv".into(),
                gguf_file::Value::U32(cfg.num_key_value_heads as u32),
            ),
            (
                "llama.feed_forward_length".into(),
                gguf_file::Value::U32(i as u32),
            ),
            (
                "llama.context_length".into(),
                gguf_file::Value::U32(cfg.max_seq_len as u32),
            ),
            (
                "llama.rope.freq_base".into(),
                gguf_file::Value::F32(cfg.rope_theta),
            ),
        ];
        let metadata: Vec<(&str, &gguf_file::Value)> = metadata_values
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect();

        let qtensors: Vec<(&str, &QTensor)> =
            tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let mut file = std::fs::File::create(dir.join("model.gguf")).unwrap();
        gguf_file::write(&mut file, &metadata, &qtensors).unwrap();
    }
}
