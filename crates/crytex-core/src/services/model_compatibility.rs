//! Model compatibility planning before loading model weights.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};

use crate::config::BackendKind;
use crate::services::hardware::DeviceKind;
use crate::services::model_manager::ManagedModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    Gguf,
    HuggingFace,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFeature {
    Dense,
    Moe,
    Gdn,
    Gguf,
    HuggingFace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStrategy {
    Cpu,
    Metal,
    CudaFused,
    CudaWithFallback,
    Remote,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompatibilityStatus {
    Ready,
    Degraded,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeFeatureSet {
    pub cuda_available: bool,
    pub metal_available: bool,
    pub gdn_cuda_available: bool,
    pub cuda_unquantized_moe_fallback_available: bool,
}

impl RuntimeFeatureSet {
    pub fn fully_enabled_cuda() -> Self {
        Self {
            cuda_available: true,
            metal_available: false,
            gdn_cuda_available: true,
            cuda_unquantized_moe_fallback_available: true,
        }
    }

    pub fn from_device(device: &DeviceKind) -> Self {
        Self {
            cuda_available: matches!(device, DeviceKind::Cuda { .. }),
            metal_available: matches!(device, DeviceKind::Metal { .. }),
            gdn_cuda_available: true,
            cuda_unquantized_moe_fallback_available: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCompatibilityPlan {
    pub format: ModelFormat,
    pub features: Vec<ModelFeature>,
    pub strategy: ExecutionStrategy,
    pub status: CompatibilityStatus,
    pub actions: Vec<String>,
    pub warnings: Vec<String>,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelMetadata {
    pub format: Option<ModelFormat>,
    pub architecture: Option<String>,
    pub model_type: Option<String>,
    pub architectures: Vec<String>,
    pub expert_count: Option<usize>,
    pub has_gdn: bool,
    pub has_moe: bool,
}

pub struct ModelMetadataInspector;

impl ModelMetadataInspector {
    pub fn inspect(model: &ManagedModel) -> Option<ModelMetadata> {
        let local_path = model.local_path.as_deref()?;
        inspect_hf_config(local_path)
            .or_else(|| inspect_gguf_binary(local_path))
            .or_else(|| inspect_gguf_sidecar(local_path))
    }
}

pub struct ModelCompatibilityPlanner;

impl ModelCompatibilityPlanner {
    pub fn plan(
        model: &ManagedModel,
        device: &DeviceKind,
        runtime: &RuntimeFeatureSet,
    ) -> ModelCompatibilityPlan {
        if model.preferred_backend != BackendKind::MistralRs {
            return ModelCompatibilityPlan {
                format: ModelFormat::Unknown,
                features: vec![ModelFeature::Dense],
                strategy: ExecutionStrategy::Remote,
                status: CompatibilityStatus::Ready,
                actions: vec!["route to configured non-mistral backend".to_string()],
                warnings: vec![],
                blockers: vec![],
            };
        }

        let metadata = ModelMetadataInspector::inspect(model);
        let format = classify_format(model, metadata.as_ref());
        let features = classify_features(model, format, metadata.as_ref());
        let blockers = compatibility_blockers(&features, device, runtime);
        let warnings = compatibility_warnings(&features, device);
        let strategy = execution_strategy(&features, device, &blockers);
        let status = compatibility_status(&blockers, &warnings);
        let actions = execution_actions(format, &features, strategy);

        ModelCompatibilityPlan {
            format,
            features,
            strategy,
            status,
            actions,
            warnings,
            blockers,
        }
    }
}

fn inspect_hf_config(local_path: &Path) -> Option<ModelMetadata> {
    let config_path = local_path
        .is_dir()
        .then(|| local_path.join("config.json"))?;
    let config = fs::read_to_string(config_path).ok()?;
    let value = serde_json::from_str::<Value>(&config).ok()?;
    let model_type = string_field(&value, "model_type");
    let architectures = string_array_field(&value, "architectures");
    let layer_types = string_array_field(&value, "layer_types");
    let expert_count = integer_field(&value, "num_experts")
        .or_else(|| integer_field(&value, "n_routed_experts"))
        .or_else(|| integer_field(&value, "moe_num_experts"));
    let searchable = metadata_terms(
        [
            model_type.as_deref(),
            architectures.iter().map(String::as_str).next(),
        ]
        .into_iter()
        .flatten(),
    );
    let layer_terms = metadata_terms(layer_types.iter().map(String::as_str));

    Some(ModelMetadata {
        format: Some(ModelFormat::HuggingFace),
        architecture: architectures.first().cloned(),
        model_type,
        architectures,
        expert_count,
        has_gdn: searchable.contains("qwen3_next")
            || searchable.contains("qwen3next")
            || searchable.contains("gdn")
            || searchable.contains("linear_attn")
            || searchable.contains("linear-attn")
            || layer_terms.contains("linear_attention"),
        has_moe: expert_count.is_some()
            || value.get("num_experts_per_tok").is_some()
            || value.get("moe_intermediate_size").is_some()
            || searchable.contains("moe")
            || searchable.contains("mixtral")
            || searchable.contains("deepseek"),
    })
}

fn inspect_gguf_sidecar(local_path: &Path) -> Option<ModelMetadata> {
    let sidecar_path = gguf_sidecar_candidates(local_path)
        .into_iter()
        .find(|path| path.exists())?;
    let sidecar = fs::read_to_string(sidecar_path).ok()?;
    let value = serde_json::from_str::<Value>(&sidecar).ok()?;
    gguf_metadata_from_json(&value)
}

fn inspect_gguf_binary(local_path: &Path) -> Option<ModelMetadata> {
    let mut file = fs::File::open(local_path).ok()?;
    let metadata = read_gguf_metadata(&mut file)?;
    let value = Value::Object(metadata.into_iter().collect());
    gguf_metadata_from_json(&value)
}

fn gguf_metadata_from_json(value: &Value) -> Option<ModelMetadata> {
    let architecture = string_field(value, "general.architecture");
    let expert_count = gguf_expert_count(value, architecture.as_deref());
    let attention_type = architecture
        .as_deref()
        .and_then(|arch| string_field(value, &format!("{arch}.attention.type")));
    let searchable = metadata_terms(
        [
            architecture.as_deref(),
            attention_type.as_deref(),
            string_field(value, "attention.type").as_deref(),
        ]
        .into_iter()
        .flatten(),
    );

    Some(ModelMetadata {
        format: Some(ModelFormat::Gguf),
        architecture,
        model_type: None,
        architectures: Vec::new(),
        expert_count,
        has_gdn: searchable.contains("qwen3_next")
            || searchable.contains("qwen3next")
            || searchable.contains("gdn")
            || searchable.contains("linear_attn")
            || searchable.contains("linear-attn"),
        has_moe: expert_count.is_some()
            || searchable.contains("moe")
            || searchable.contains("mixtral")
            || searchable.contains("deepseek"),
    })
}

fn read_gguf_metadata(reader: &mut (impl Read + Seek)) -> Option<serde_json::Map<String, Value>> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).ok()?;
    if magic != *b"GGUF" {
        return None;
    }

    let version = read_u32(reader)?;
    if version < 2 {
        return None;
    }

    let _tensor_count = read_u64(reader)?;
    let metadata_count = read_u64(reader)?;
    let mut metadata = serde_json::Map::new();

    for _ in 0..metadata_count {
        let key = read_gguf_string(reader)?;
        let value_type = read_u32(reader)?;
        let value = read_gguf_value(reader, value_type)?;
        metadata.insert(key, value);
    }

    Some(metadata)
}

fn classify_format(model: &ManagedModel, metadata: Option<&ModelMetadata>) -> ModelFormat {
    if let Some(format) = metadata.and_then(|metadata| metadata.format) {
        return format;
    }

    let haystack = model_haystack(model);
    if haystack.contains(".gguf") || haystack.contains("-gguf") || haystack.contains("_gguf") {
        ModelFormat::Gguf
    } else if model.preferred_backend == BackendKind::MistralRs {
        ModelFormat::HuggingFace
    } else {
        ModelFormat::Unknown
    }
}

fn classify_features(
    model: &ManagedModel,
    format: ModelFormat,
    metadata: Option<&ModelMetadata>,
) -> Vec<ModelFeature> {
    let haystack = model_haystack(model);
    let mut features = Vec::new();

    match format {
        ModelFormat::Gguf => features.push(ModelFeature::Gguf),
        ModelFormat::HuggingFace => features.push(ModelFeature::HuggingFace),
        ModelFormat::Unknown => {}
    }

    if metadata.is_some_and(|metadata| metadata.has_gdn)
        || haystack.contains("qwen3-next")
        || haystack.contains("gdn")
        || haystack.contains("linear-attn")
    {
        features.push(ModelFeature::Gdn);
    }

    if metadata.is_some_and(|metadata| metadata.has_moe)
        || haystack.contains("moe")
        || haystack.contains("mixtral")
        || haystack.contains("deepseek")
    {
        features.push(ModelFeature::Moe);
    }

    if !features
        .iter()
        .any(|feature| matches!(feature, ModelFeature::Moe | ModelFeature::Gdn))
    {
        features.push(ModelFeature::Dense);
    }

    features.sort_by_key(|feature| *feature as u8);
    features.dedup();
    features
}

fn string_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn string_array_field(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn integer_field(value: &Value, key: &str) -> Option<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn metadata_terms<'a>(terms: impl Iterator<Item = &'a str>) -> String {
    terms
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

fn gguf_sidecar_candidates(local_path: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    candidates.push(PathBuf::from(format!(
        "{}.metadata.json",
        local_path.display()
    )));

    if let Some(file_stem) = local_path.file_stem().and_then(|stem| stem.to_str()) {
        candidates.push(local_path.with_file_name(format!("{file_stem}.metadata.json")));
    }

    candidates
}

fn gguf_expert_count(value: &Value, architecture: Option<&str>) -> Option<usize> {
    architecture
        .and_then(|arch| integer_field(value, &format!("{arch}.expert_count")))
        .or_else(|| integer_field(value, "expert_count"))
        .or_else(|| integer_field(value, "llm.expert_count"))
}

fn read_gguf_value(reader: &mut (impl Read + Seek), value_type: u32) -> Option<Value> {
    match value_type {
        0 => read_u8(reader).map(|value| Value::from(value as u64)),
        1 => read_i8(reader).map(|value| Value::from(value as i64)),
        2 => read_u16(reader).map(|value| Value::from(value as u64)),
        3 => read_i16(reader).map(|value| Value::from(value as i64)),
        4 => read_u32(reader).map(|value| Value::from(value as u64)),
        5 => read_i32(reader).map(|value| Value::from(value as i64)),
        6 => read_f32(reader).map(Value::from),
        7 => read_u8(reader).map(|value| Value::Bool(value != 0)),
        8 => read_gguf_string(reader).map(Value::String),
        9 => read_gguf_array(reader),
        10 => read_u64(reader).map(Value::from),
        11 => read_i64(reader).map(Value::from),
        12 => read_f64(reader).map(Value::from),
        _ => None,
    }
}

fn read_gguf_array(reader: &mut (impl Read + Seek)) -> Option<Value> {
    let value_type = read_u32(reader)?;
    let len = read_u64(reader)?;
    let values = (0..len)
        .map(|_| read_gguf_value(reader, value_type))
        .collect::<Option<Vec<_>>>()?;
    Some(Value::Array(values))
}

fn read_gguf_string(reader: &mut (impl Read + Seek)) -> Option<String> {
    let len = usize::try_from(read_u64(reader)?).ok()?;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes).ok()?;
    String::from_utf8(bytes).ok()
}

fn read_u8(reader: &mut (impl Read + Seek)) -> Option<u8> {
    let mut bytes = [0u8; 1];
    reader.read_exact(&mut bytes).ok()?;
    Some(bytes[0])
}

fn read_i8(reader: &mut (impl Read + Seek)) -> Option<i8> {
    read_u8(reader).map(|value| value as i8)
}

fn read_u16(reader: &mut (impl Read + Seek)) -> Option<u16> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes).ok()?;
    Some(u16::from_le_bytes(bytes))
}

fn read_i16(reader: &mut (impl Read + Seek)) -> Option<i16> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes).ok()?;
    Some(i16::from_le_bytes(bytes))
}

fn read_u32(reader: &mut (impl Read + Seek)) -> Option<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).ok()?;
    Some(u32::from_le_bytes(bytes))
}

fn read_i32(reader: &mut (impl Read + Seek)) -> Option<i32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).ok()?;
    Some(i32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut (impl Read + Seek)) -> Option<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes).ok()?;
    Some(u64::from_le_bytes(bytes))
}

fn read_i64(reader: &mut (impl Read + Seek)) -> Option<i64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes).ok()?;
    Some(i64::from_le_bytes(bytes))
}

fn read_f32(reader: &mut (impl Read + Seek)) -> Option<f32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).ok()?;
    Some(f32::from_le_bytes(bytes))
}

fn read_f64(reader: &mut (impl Read + Seek)) -> Option<f64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes).ok()?;
    Some(f64::from_le_bytes(bytes))
}

fn compatibility_blockers(
    features: &[ModelFeature],
    device: &DeviceKind,
    runtime: &RuntimeFeatureSet,
) -> Vec<String> {
    if !matches!(device, DeviceKind::Cuda { .. }) {
        return vec![];
    }

    let mut blockers = Vec::new();
    if features.contains(&ModelFeature::Gdn) && !runtime.gdn_cuda_available {
        blockers.push("GDN CUDA kernel support is unavailable in this build".to_string());
    }
    if features.contains(&ModelFeature::Moe) && !runtime.cuda_unquantized_moe_fallback_available {
        blockers.push(
            "CUDA MoE fallback for unquantized Tensor/TensorF16 experts is unavailable".to_string(),
        );
    }
    blockers
}

fn compatibility_warnings(features: &[ModelFeature], device: &DeviceKind) -> Vec<String> {
    if matches!(device, DeviceKind::Cpu)
        && features
            .iter()
            .any(|feature| matches!(feature, ModelFeature::Gdn | ModelFeature::Moe))
    {
        return vec!["MoE/GDN models can run slowly on CPU; prefer GPU when available".to_string()];
    }
    vec![]
}

fn execution_strategy(
    features: &[ModelFeature],
    device: &DeviceKind,
    blockers: &[String],
) -> ExecutionStrategy {
    if !blockers.is_empty() {
        return ExecutionStrategy::Unsupported;
    }

    match device {
        DeviceKind::Cpu => ExecutionStrategy::Cpu,
        DeviceKind::Metal { .. } => ExecutionStrategy::Metal,
        DeviceKind::Cuda { .. } => {
            if features
                .iter()
                .any(|feature| matches!(feature, ModelFeature::Moe | ModelFeature::Gdn))
            {
                ExecutionStrategy::CudaWithFallback
            } else {
                ExecutionStrategy::CudaFused
            }
        }
    }
}

fn compatibility_status(blockers: &[String], warnings: &[String]) -> CompatibilityStatus {
    if !blockers.is_empty() {
        CompatibilityStatus::Unsupported
    } else if !warnings.is_empty() {
        CompatibilityStatus::Degraded
    } else {
        CompatibilityStatus::Ready
    }
}

fn execution_actions(
    format: ModelFormat,
    features: &[ModelFeature],
    strategy: ExecutionStrategy,
) -> Vec<String> {
    let mut actions = vec![format!("use {strategy:?} execution strategy")];
    if format == ModelFormat::HuggingFace {
        actions.push("read Hugging Face config before weight load when available".to_string());
    }
    if features.contains(&ModelFeature::Moe) {
        actions.push("enable MoE expert fallback checks before generation".to_string());
    }
    if features.contains(&ModelFeature::Gdn) {
        actions.push("require GDN kernel/probe evidence before GPU generation".to_string());
    }
    actions
}

fn model_haystack(model: &ManagedModel) -> String {
    [
        Some(model.id.as_str()),
        Some(model.name.as_str()),
        model.repo.as_deref(),
        model.filename.as_deref(),
        model.local_path.as_ref().and_then(|path| path.to_str()),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(" ")
    .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BackendKind;
    use crate::services::hardware::DeviceKind;
    use crate::services::model_manager::{ManagedModel, ModelStatus};
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn should_plan_qwen3_next_moe_gdn_with_cuda_fallbacks_instead_of_model_specific_fix() {
        let model = model("tiny-random/qwen3-next-moe", None, None);
        let features = RuntimeFeatureSet {
            cuda_available: true,
            metal_available: false,
            gdn_cuda_available: true,
            cuda_unquantized_moe_fallback_available: true,
        };

        let plan = ModelCompatibilityPlanner::plan(
            &model,
            &DeviceKind::Cuda {
                name: "RTX 5080".into(),
                vram_mb: 16_303,
                driver_version: "596.36".into(),
            },
            &features,
        );

        assert_eq!(plan.status, CompatibilityStatus::Ready);
        assert_eq!(plan.format, ModelFormat::HuggingFace);
        assert!(plan.features.contains(&ModelFeature::Gdn));
        assert!(plan.features.contains(&ModelFeature::Moe));
        assert_eq!(plan.strategy, ExecutionStrategy::CudaWithFallback);
        assert!(plan.blockers.is_empty());
    }

    #[test]
    fn should_reject_gdn_cuda_when_kernel_support_is_missing_before_model_load() {
        let model = model("tiny-random/qwen3-next-moe", None, None);
        let features = RuntimeFeatureSet {
            cuda_available: true,
            metal_available: false,
            gdn_cuda_available: false,
            cuda_unquantized_moe_fallback_available: true,
        };

        let plan = ModelCompatibilityPlanner::plan(
            &model,
            &DeviceKind::Cuda {
                name: "RTX 5080".into(),
                vram_mb: 16_303,
                driver_version: "596.36".into(),
            },
            &features,
        );

        assert_eq!(plan.status, CompatibilityStatus::Unsupported);
        assert!(
            plan.blockers
                .iter()
                .any(|blocker| blocker.contains("GDN CUDA kernel"))
        );
    }

    #[test]
    fn should_classify_gguf_dense_model_as_cuda_fused_candidate() {
        let model = model(
            "tinyllama-q2",
            Some("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"),
            Some(PathBuf::from("tinyllama-1.1b-chat-v1.0.Q2_K.gguf")),
        );

        let plan = ModelCompatibilityPlanner::plan(
            &model,
            &DeviceKind::Cuda {
                name: "RTX 5080".into(),
                vram_mb: 16_303,
                driver_version: "596.36".into(),
            },
            &RuntimeFeatureSet::fully_enabled_cuda(),
        );

        assert_eq!(plan.status, CompatibilityStatus::Ready);
        assert_eq!(plan.format, ModelFormat::Gguf);
        assert!(plan.features.contains(&ModelFeature::Gguf));
        assert_eq!(plan.strategy, ExecutionStrategy::CudaFused);
    }

    #[test]
    fn should_detect_hf_moe_gdn_from_config_without_name_hints() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{
                "model_type": "qwen3_next",
                "architectures": ["Qwen3NextForCausalLM"],
                "num_experts": 32,
                "num_experts_per_tok": 4,
                "layer_types": ["linear_attention"]
            }"#,
        )
        .unwrap();
        let model = model("local-model", None, Some(dir.path().to_path_buf()));

        let plan = ModelCompatibilityPlanner::plan(
            &model,
            &DeviceKind::Cuda {
                name: "RTX 5080".into(),
                vram_mb: 16_303,
                driver_version: "596.36".into(),
            },
            &RuntimeFeatureSet::fully_enabled_cuda(),
        );

        assert_eq!(plan.format, ModelFormat::HuggingFace);
        assert!(plan.features.contains(&ModelFeature::Moe));
        assert!(plan.features.contains(&ModelFeature::Gdn));
        assert_eq!(plan.strategy, ExecutionStrategy::CudaWithFallback);
    }

    #[test]
    fn should_detect_gguf_moe_gdn_from_sidecar_metadata_without_name_hints() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();
        fs::write(
            dir.path().join("model.gguf.metadata.json"),
            r#"{
                "general.architecture": "qwen3_next",
                "qwen3_next.expert_count": 32,
                "qwen3_next.attention.type": "gdn"
            }"#,
        )
        .unwrap();
        let model = model("local-artifact", None, Some(model_path));

        let plan = ModelCompatibilityPlanner::plan(
            &model,
            &DeviceKind::Cuda {
                name: "RTX 5080".into(),
                vram_mb: 16_303,
                driver_version: "596.36".into(),
            },
            &RuntimeFeatureSet::fully_enabled_cuda(),
        );

        assert_eq!(plan.format, ModelFormat::Gguf);
        assert!(plan.features.contains(&ModelFeature::Moe));
        assert!(plan.features.contains(&ModelFeature::Gdn));
        assert_eq!(plan.strategy, ExecutionStrategy::CudaWithFallback);
    }

    #[test]
    fn should_detect_gguf_moe_gdn_from_binary_metadata_without_sidecar() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        write_minimal_gguf(
            &model_path,
            &[
                (
                    "general.architecture",
                    GgufFixtureValue::String("qwen3_next"),
                ),
                ("qwen3_next.expert_count", GgufFixtureValue::U32(32)),
                ("qwen3_next.attention.type", GgufFixtureValue::String("gdn")),
            ],
        );
        fs::OpenOptions::new()
            .append(true)
            .open(&model_path)
            .unwrap()
            .write_all(&vec![0u8; 1024 * 1024])
            .unwrap();
        let model = model("local-artifact", None, Some(model_path));

        let plan = ModelCompatibilityPlanner::plan(
            &model,
            &DeviceKind::Cuda {
                name: "RTX 5080".into(),
                vram_mb: 16_303,
                driver_version: "596.36".into(),
            },
            &RuntimeFeatureSet::fully_enabled_cuda(),
        );

        assert_eq!(plan.format, ModelFormat::Gguf);
        assert!(plan.features.contains(&ModelFeature::Moe));
        assert!(plan.features.contains(&ModelFeature::Gdn));
        assert_eq!(plan.strategy, ExecutionStrategy::CudaWithFallback);
    }

    fn model(id: &str, repo: Option<&str>, local_path: Option<PathBuf>) -> ManagedModel {
        ManagedModel {
            id: id.into(),
            name: id.into(),
            repo: repo.map(str::to_string),
            filename: local_path
                .as_ref()
                .and_then(|path| path.file_name())
                .and_then(|name| name.to_str())
                .map(str::to_string),
            local_path,
            quantization: None,
            preferred_backend: BackendKind::MistralRs,
            params_b: None,
            status: ModelStatus::Available,
        }
    }

    enum GgufFixtureValue<'a> {
        String(&'a str),
        U32(u32),
    }

    fn write_minimal_gguf(path: &std::path::Path, metadata: &[(&str, GgufFixtureValue<'_>)]) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

        metadata.iter().for_each(|(key, value)| {
            write_gguf_string(&mut bytes, key);
            match value {
                GgufFixtureValue::String(value) => {
                    bytes.extend_from_slice(&8u32.to_le_bytes());
                    write_gguf_string(&mut bytes, value);
                }
                GgufFixtureValue::U32(value) => {
                    bytes.extend_from_slice(&4u32.to_le_bytes());
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
        });

        fs::write(path, bytes).unwrap();
    }

    fn write_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
}
