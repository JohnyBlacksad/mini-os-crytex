//! A minimal Llama-style causal language model with LoRA on attention layers.
//!
//! The architecture intentionally follows the Hugging Face Llama weight naming
//! convention (`model.embed_tokens`, `model.layers.{i}.self_attn.q_proj`, etc.)
//! so that pretrained base weights can be loaded via `VarBuilder`.  When no
//! pretrained weights are provided, a tiny random transformer is used for
//! testing and for environments without a multi-gigabyte model download.

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor, Var};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, ops::silu,
    ops::softmax_last_dim, rms_norm,
};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::repeat_kv;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Configuration for the causal LM architecture and LoRA adapters.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub rank: usize,
    pub alpha: usize,
    pub target_modules: Vec<String>,
    pub dtype: DType,
    pub use_flash_attn: bool,
}

impl ModelConfig {
    /// A tiny transformer suitable for unit tests and local experiments.
    pub fn tiny_for_tests(rank: usize, alpha: usize, max_seq_len: usize) -> Self {
        Self {
            vocab_size: 256,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 2,
            num_key_value_heads: 1,
            intermediate_size: 64,
            max_seq_len,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rank,
            alpha,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            dtype: DType::F32,
            use_flash_attn: false,
        }
    }

    /// Load architecture hyper-parameters from a Hugging Face `config.json`.
    pub fn from_pretrained_dir(dir: &Path) -> CandleResult<Self> {
        let path = dir.join("config.json");
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| candle_core::Error::Msg(format!("cannot read config.json: {e}")))?;
        let hf: HfConfig = serde_json::from_str(&raw)
            .map_err(|e| candle_core::Error::Msg(format!("cannot parse config.json: {e}")))?;
        let num_key_value_heads = hf.num_key_value_heads.unwrap_or(hf.num_attention_heads);
        let dtype = hf
            .torch_dtype
            .as_deref()
            .map(parse_torch_dtype)
            .unwrap_or(DType::F32);
        Ok(Self {
            vocab_size: hf.vocab_size,
            hidden_size: hf.hidden_size,
            num_layers: hf.num_hidden_layers,
            num_heads: hf.num_attention_heads,
            num_key_value_heads,
            intermediate_size: hf.intermediate_size,
            max_seq_len: hf.max_position_embeddings.unwrap_or(2048),
            rms_norm_eps: hf.rms_norm_eps.unwrap_or(1e-6),
            rope_theta: hf.rope_theta.unwrap_or(10_000.0),
            rank: 0,
            alpha: 0,
            target_modules: Vec::new(),
            dtype,
            use_flash_attn: false,
        })
    }

    /// Load architecture hyper-parameters from a GGUF file's metadata.
    pub fn from_gguf(path: &Path) -> CandleResult<Self> {
        use candle_core::quantized::gguf_file;
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let m = &content.metadata;
        let arch = m
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|v| v.as_str().to_string())
            .unwrap_or_default();
        if arch != "llama" {
            return Err(candle_core::Error::Msg(format!(
                "GGUF architecture '{arch}' is not supported, only 'llama' is"
            )));
        }
        let get_u32 =
            |key: &str| m.get(key).and_then(|v| v.to_u32().ok()).unwrap_or_default() as usize;
        let num_heads = get_u32("llama.attention.head_count");
        let num_key_value_heads = m
            .get("llama.attention.head_count_kv")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(num_heads);
        Ok(Self {
            vocab_size: get_u32("llama.vocab_size"),
            hidden_size: get_u32("llama.embedding_length"),
            num_layers: get_u32("llama.block_count"),
            num_heads,
            num_key_value_heads,
            intermediate_size: get_u32("llama.feed_forward_length"),
            max_seq_len: m
                .get("llama.context_length")
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(2048),
            rms_norm_eps: 1e-6,
            rope_theta: m
                .get("llama.rope.freq_base")
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(10_000.0),
            rank: 0,
            alpha: 0,
            target_modules: Vec::new(),
            dtype: DType::F32,
            use_flash_attn: false,
        })
    }

    pub fn with_lora(mut self, rank: usize, alpha: usize, target_modules: Vec<String>) -> Self {
        self.rank = rank;
        self.alpha = alpha;
        self.target_modules = target_modules;
        self
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn kv_hidden_size(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }
}

#[derive(Debug, Deserialize)]
struct HfConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    num_key_value_heads: Option<usize>,
    max_position_embeddings: Option<usize>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f32>,
    torch_dtype: Option<String>,
}

fn parse_torch_dtype(s: &str) -> DType {
    match s {
        "float16" => DType::F16,
        "bfloat16" => DType::BF16,
        "float32" | "float" => DType::F32,
        _ => DType::F32,
    }
}

/// A linear layer with frozen base weights and a trainable low-rank adapter.
///
/// Forward: `y = x @ W0^T + (x @ A^T @ B^T) * (alpha / rank)`.
#[derive(Clone, Debug)]
pub struct LoraLinear {
    base: Tensor,
    lora_a: Var,
    lora_b: Var,
    scale_tensor: Tensor,
}

impl LoraLinear {
    /// Create a LoRA layer with randomly initialized base weights.
    #[allow(dead_code)]
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        let base = Tensor::randn(0.0f32, 0.02f32, (out_features, in_features), device)?;
        Self::from_base(base, rank, alpha)
    }

    /// Wrap a pretrained base weight matrix with LoRA adapters.
    pub fn from_base(base: Tensor, rank: usize, alpha: usize) -> CandleResult<Self> {
        let device = base.device();
        let dtype = base.dtype();
        let (out_features, in_features) = base.dims2()?;
        let rank = rank.max(1);
        let lora_a_t =
            Tensor::randn(0.0f32, 0.02f32, (rank, in_features), device)?.to_dtype(dtype)?;
        let lora_b_t =
            Tensor::randn(0.0f32, 0.02f32, (out_features, rank), device)?.to_dtype(dtype)?;
        let lora_a = Var::from_tensor(&lora_a_t)?;
        let lora_b = Var::from_tensor(&lora_b_t)?;
        let scale = alpha as f64 / rank as f64;
        let scale_tensor = Tensor::new(scale as f32, device)?.to_dtype(dtype)?;
        Ok(Self {
            base,
            lora_a,
            lora_b,
            scale_tensor,
        })
    }

    pub fn vars(&self) -> Vec<Var> {
        vec![self.lora_a.clone(), self.lora_b.clone()]
    }

    pub fn named_tensors(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(
            format!("{prefix}.lora_A.weight"),
            self.lora_a.as_tensor().clone(),
        );
        map.insert(
            format!("{prefix}.lora_B.weight"),
            self.lora_b.as_tensor().clone(),
        );
        map
    }

    pub fn detached_named_tensors(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(
            format!("{prefix}.lora_A.weight"),
            self.lora_a.as_detached_tensor(),
        );
        map.insert(
            format!("{prefix}.lora_B.weight"),
            self.lora_b.as_detached_tensor(),
        );
        map
    }

    pub fn set_lora_a(&mut self, tensor: &Tensor) -> CandleResult<()> {
        self.lora_a.set(tensor)
    }

    pub fn set_lora_b(&mut self, tensor: &Tensor) -> CandleResult<()> {
        self.lora_b.set(tensor)
    }
}

impl Module for LoraLinear {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let dims = xs.dims();
        let in_features = *dims
            .last()
            .expect("input tensor must have at least one dimension");
        let batch = dims[..dims.len() - 1].iter().product::<usize>();
        let xs_2d = xs.reshape((batch, in_features))?;

        let base_out = xs_2d.matmul(&self.base.t()?)?;
        let lora_out = xs_2d
            .matmul(&self.lora_a.as_tensor().t()?)?
            .matmul(&self.lora_b.as_tensor().t()?)?;
        let out_2d = (base_out + lora_out.broadcast_mul(&self.scale_tensor)?)?;

        let out_features = self.base.dims()[0];
        let mut out_dims = dims.to_vec();
        out_dims.pop();
        out_dims.push(out_features);
        out_2d.reshape(out_dims)
    }
}

/// Key/value cache for autoregressive generation.
#[derive(Clone, Debug)]
pub struct KvCache {
    k: Tensor,
    v: Tensor,
}

impl KvCache {
    pub fn len(&self) -> usize {
        self.k.dims()[2]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn k(&self) -> &Tensor {
        &self.k
    }

    pub fn v(&self) -> &Tensor {
        &self.v
    }
}

struct CausalSelfAttention {
    q_proj: LoraLinear,
    k_proj: Linear,
    v_proj: LoraLinear,
    o_proj: Linear,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    use_flash_attn: bool,
}

impl CausalSelfAttention {
    fn lora_vars(&self) -> Vec<Var> {
        let mut vars = self.q_proj.vars();
        vars.extend(self.v_proj.vars());
        vars
    }

    fn lora_tensors(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut tensors = self
            .q_proj
            .named_tensors(&format!("{prefix}.self_attn.q_proj"));
        tensors.extend(
            self.v_proj
                .named_tensors(&format!("{prefix}.self_attn.v_proj")),
        );
        tensors
    }

    fn lora_detached_tensors(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut tensors = self
            .q_proj
            .detached_named_tensors(&format!("{prefix}.self_attn.q_proj"));
        tensors.extend(
            self.v_proj
                .detached_named_tensors(&format!("{prefix}.self_attn.v_proj")),
        );
        tensors
    }

    fn load_adapter(
        &mut self,
        prefix: &str,
        tensors: &HashMap<String, Tensor>,
    ) -> CandleResult<()> {
        let q_prefix = format!("{prefix}.self_attn.q_proj");
        let v_prefix = format!("{prefix}.self_attn.v_proj");
        if has_lora_linear_adapter(&q_prefix, tensors) {
            load_lora_linear_adapter(&mut self.q_proj, &q_prefix, tensors)?;
        }
        if has_lora_linear_adapter(&v_prefix, tensors) {
            load_lora_linear_adapter(&mut self.v_proj, &v_prefix, tensors)?;
        }
        Ok(())
    }

    fn forward_with_cache(&self, x: &Tensor, cache: &mut Option<KvCache>) -> CandleResult<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let reshape_q = |t: Tensor| {
            t.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let reshape_kv = |t: Tensor| {
            t.reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };

        let q = reshape_q(q)?;
        let k = reshape_kv(k)?;
        let v = reshape_kv(v)?;

        let start_pos = cache.as_ref().map(KvCache::len).unwrap_or(0);
        let total_len = start_pos + seq_len;
        let (cos, sin) = rope_freqs(total_len, self.head_dim, self.rope_theta, x.device())?;
        let cos = cos.i(start_pos..)?;
        let sin = sin.i(start_pos..)?;
        let q = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;

        let (k_full, v_full) = if let Some(c) = cache {
            let k_full = Tensor::cat(&[c.k(), &k], 2)?;
            let v_full = Tensor::cat(&[c.v(), &v], 2)?;
            *cache = Some(KvCache {
                k: k_full.clone(),
                v: v_full.clone(),
            });
            (k_full, v_full)
        } else {
            (k.clone(), v.clone())
        };

        let n_rep = self.num_heads / self.num_key_value_heads;
        let k_full = repeat_kv(k_full, n_rep)?;
        let v_full = repeat_kv(v_full, n_rep)?;

        let y = if self.use_flash_attn && flash_available(x.device(), q.dtype()) {
            flash_attention(&q, &k_full, &v_full, self.head_dim)?
        } else {
            let scores = (q.matmul(&k_full.t()?)? / (self.head_dim as f64).sqrt())?;
            let mask = causal_mask(seq_len, start_pos, x.device())?
                .to_dtype(scores.dtype())?
                .broadcast_as(scores.shape())?;
            let mask_penalty = Tensor::new(-1.0e9f32, x.device())?
                .to_dtype(scores.dtype())?
                .broadcast_as(scores.shape())?;
            let scores = (scores + mask.broadcast_mul(&mask_penalty)?)?;
            let attn = softmax_last_dim(&scores)?;
            attn.matmul(&v_full.contiguous()?)?
        };

        let y = y.transpose(1, 2)?.reshape((b_sz, seq_len, hidden_size))?;
        self.o_proj.forward(&y)
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &ModelConfig) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate_proj: linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = silu(&self.gate_proj.forward(x)?)?;
        let x = (gate * self.up_proj.forward(x)?)?;
        self.down_proj.forward(&x)
    }
}

struct Block {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    self_attn: CausalSelfAttention,
    mlp: Mlp,
}

impl Block {
    fn lora_vars(&self) -> Vec<Var> {
        self.self_attn.lora_vars()
    }

    fn lora_tensors(&self, prefix: &str) -> HashMap<String, Tensor> {
        self.self_attn.lora_tensors(prefix)
    }

    fn lora_detached_tensors(&self, prefix: &str) -> HashMap<String, Tensor> {
        self.self_attn.lora_detached_tensors(prefix)
    }

    fn load_adapter(
        &mut self,
        prefix: &str,
        tensors: &HashMap<String, Tensor>,
    ) -> CandleResult<()> {
        self.self_attn.load_adapter(prefix, tensors)
    }

    fn forward_with_cache(&self, x: &Tensor, cache: &mut Option<KvCache>) -> CandleResult<Tensor> {
        let residual = x;
        let x = (self
            .self_attn
            .forward_with_cache(&self.input_layernorm.forward(x)?, cache)?
            + residual)?;
        let residual = &x;
        let x = (self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&x)?)?
            + residual)?;
        Ok(x)
    }
}

/// A Llama-style causal language model with optional LoRA adapters.
pub struct LoraCausalLM {
    embed_tokens: Embedding,
    norm: RmsNorm,
    lm_head: Linear,
    lm_head_lora: Option<LoraLinear>,
    blocks: Vec<Block>,
    lora_layers: Vec<(String, LoraLinear)>,
    vocab_size: usize,
}

impl LoraCausalLM {
    /// Build a model from a `VarBuilder`.
    ///
    /// When `vb` is backed by a `VarMap` the base weights are randomly
    /// initialized.  When `vb` is backed by safetensors files the base weights
    /// are loaded and frozen (only the LoRA adapters are trainable).
    pub fn load(vb: VarBuilder, cfg: &ModelConfig) -> CandleResult<Self> {
        let mut lora_layers = Vec::new();
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let h = cfg.hidden_size;
        let kv_h = cfg.kv_hidden_size();
        let q_lora_alpha = target_lora_alpha(cfg, &["q_proj", "attn_q"]);
        let v_lora_alpha = target_lora_alpha(cfg, &["v_proj", "attn_v"]);
        for i in 0..cfg.num_layers {
            let block_vb = vb.pp(format!("model.layers.{i}"));
            let q_base = block_vb.pp("self_attn.q_proj").get((h, h), "weight")?;
            let v_base = block_vb.pp("self_attn.v_proj").get((kv_h, h), "weight")?;
            let q_proj = LoraLinear::from_base(q_base, cfg.rank, q_lora_alpha)?;
            let v_proj = LoraLinear::from_base(v_base, cfg.rank, v_lora_alpha)?;
            lora_layers.push((format!("model.layers.{i}.self_attn.q_proj"), q_proj));
            lora_layers.push((format!("model.layers.{i}.self_attn.v_proj"), v_proj));

            blocks.push(Block {
                input_layernorm: rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    block_vb.pp("input_layernorm"),
                )?,
                post_attention_layernorm: rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    block_vb.pp("post_attention_layernorm"),
                )?,
                self_attn: CausalSelfAttention {
                    q_proj: lora_layers[lora_layers.len() - 2].1.clone(),
                    k_proj: linear_no_bias(h, kv_h, block_vb.pp("self_attn.k_proj"))?,
                    v_proj: lora_layers[lora_layers.len() - 1].1.clone(),
                    o_proj: linear_no_bias(h, h, block_vb.pp("self_attn.o_proj"))?,
                    num_heads: cfg.num_heads,
                    num_key_value_heads: cfg.num_key_value_heads,
                    head_dim: cfg.head_dim(),
                    rope_theta: cfg.rope_theta,
                    use_flash_attn: cfg.use_flash_attn,
                },
                mlp: Mlp::load(block_vb.pp("mlp"), cfg)?,
            });
        }

        let target_lm_head = cfg
            .target_modules
            .iter()
            .any(|module| module == "lm_head" || module == "output");
        let lm_head_base = vb
            .pp("lm_head")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let lm_head_lora = target_lm_head
            .then(|| LoraLinear::from_base(lm_head_base.clone(), cfg.rank, cfg.alpha))
            .transpose()?;
        Ok(Self {
            embed_tokens: embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?,
            norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            lm_head: Linear::new(lm_head_base, None),
            lm_head_lora,
            blocks,
            lora_layers,
            vocab_size: cfg.vocab_size,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let mut cache: Vec<Option<KvCache>> = vec![None; self.blocks.len()];
        self.forward_with_cache(input_ids, &mut cache)
    }

    pub fn forward_with_cache(
        &self,
        input_ids: &Tensor,
        cache: &mut [Option<KvCache>],
    ) -> CandleResult<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        for (block, block_cache) in self.blocks.iter().zip(cache.iter_mut()) {
            x = block.forward_with_cache(&x, block_cache)?;
        }
        let x = self.norm.forward(&x)?;
        self.lm_head_lora
            .as_ref()
            .map(|head| head.forward(&x))
            .unwrap_or_else(|| self.lm_head.forward(&x))
    }

    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: Option<f64>,
        device: &Device,
    ) -> CandleResult<Vec<u32>> {
        let mut tokens = prompt_tokens.to_vec();
        let sampling = match temperature {
            None | Some(0.0) => Sampling::ArgMax,
            Some(t) => Sampling::TopK {
                temperature: t,
                k: 1,
            },
        };
        let mut processor = LogitsProcessor::from_sampling(0, sampling);
        let mut cache: Vec<Option<KvCache>> = vec![None; self.blocks.len()];

        let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
        let logits = self.forward_with_cache(&input, &mut cache)?;
        let mut next = sample_next(&logits, &mut processor)?;
        tokens.push(next);

        for _ in 1..max_new_tokens {
            let input = Tensor::new(&[next], device)?.unsqueeze(0)?;
            let logits = self.forward_with_cache(&input, &mut cache)?;
            next = sample_next(&logits, &mut processor)?;
            tokens.push(next);
        }
        Ok(tokens)
    }

    pub fn load_adapter(&mut self, path: &Path) -> CandleResult<()> {
        let device = self.lora_layers[0].1.lora_a.as_tensor().device().clone();
        let tensors = candle_core::safetensors::load(path, &device)?;
        for (layer, block) in self.blocks.iter_mut().enumerate() {
            block.load_adapter(&format!("model.layers.{layer}"), &tensors)?;
        }
        if let Some(head) = &mut self.lm_head_lora
            && has_lora_linear_adapter("lm_head", &tensors)
        {
            load_lora_linear_adapter(head, "lm_head", &tensors)?;
        }
        Ok(())
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn lora_vars(&self) -> Vec<Var> {
        let mut vars: Vec<Var> = self.blocks.iter().flat_map(Block::lora_vars).collect();
        if let Some(head) = &self.lm_head_lora {
            vars.extend(head.vars());
        }
        vars
    }

    pub fn lora_tensors(&self) -> HashMap<String, Tensor> {
        let mut tensors: HashMap<String, Tensor> = self
            .blocks
            .iter()
            .enumerate()
            .flat_map(|(layer, block)| block.lora_tensors(&format!("model.layers.{layer}")))
            .collect();
        if let Some(head) = &self.lm_head_lora {
            tensors.extend(head.named_tensors("lm_head"));
        }
        tensors
    }

    pub fn lora_detached_tensors(&self) -> HashMap<String, Tensor> {
        let mut tensors: HashMap<String, Tensor> = self
            .blocks
            .iter()
            .enumerate()
            .flat_map(|(layer, block)| {
                block.lora_detached_tensors(&format!("model.layers.{layer}"))
            })
            .collect();
        if let Some(head) = &self.lm_head_lora {
            tensors.extend(head.detached_named_tensors("lm_head"));
        }
        tensors
    }
}

fn target_lora_alpha(cfg: &ModelConfig, aliases: &[&str]) -> usize {
    if cfg
        .target_modules
        .iter()
        .any(|module| aliases.iter().any(|alias| module == alias))
    {
        cfg.alpha
    } else {
        0
    }
}

fn has_lora_linear_adapter(name: &str, tensors: &HashMap<String, Tensor>) -> bool {
    let a_key = format!("{name}.lora_A.weight");
    let b_key = format!("{name}.lora_B.weight");
    let legacy_a_key = format!("{name}.lora_a");
    let legacy_b_key = format!("{name}.lora_b");
    (tensors.contains_key(&a_key) || tensors.contains_key(&legacy_a_key))
        && (tensors.contains_key(&b_key) || tensors.contains_key(&legacy_b_key))
}

fn load_lora_linear_adapter(
    lora: &mut LoraLinear,
    name: &str,
    tensors: &HashMap<String, Tensor>,
) -> CandleResult<()> {
    let a_key = format!("{name}.lora_A.weight");
    let b_key = format!("{name}.lora_B.weight");
    let legacy_a_key = format!("{name}.lora_a");
    let legacy_b_key = format!("{name}.lora_b");
    let a = tensors
        .get(&a_key)
        .or_else(|| tensors.get(&legacy_a_key))
        .ok_or_else(|| candle_core::Error::Msg(format!("adapter missing tensor {a_key}")))?;
    let b = tensors
        .get(&b_key)
        .or_else(|| tensors.get(&legacy_b_key))
        .ok_or_else(|| candle_core::Error::Msg(format!("adapter missing tensor {b_key}")))?;
    lora.set_lora_a(a)?;
    lora.set_lora_b(b)
}

fn sample_next(logits: &Tensor, processor: &mut LogitsProcessor) -> CandleResult<u32> {
    let seq_len = logits.dim(1)?;
    let logits = logits
        .i((0, seq_len - 1, ..))?
        .to_dtype(DType::F32)?
        .squeeze(0)?;
    processor.sample(&logits)
}

fn rope_freqs(
    seq_len: usize,
    head_dim: usize,
    theta: f32,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / theta.powf(i as f32 / head_dim as f32))
        .collect();
    let theta_t = Tensor::new(inv_freq, device)?.reshape((1, head_dim / 2))?;
    let pos = Tensor::arange(0u32, seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((seq_len, 1))?;
    let idx_theta = pos.matmul(&theta_t)?;
    Ok((idx_theta.cos()?, idx_theta.sin()?))
}

fn causal_mask(seq_len: usize, start_pos: usize, device: &Device) -> CandleResult<Tensor> {
    let total_len = start_pos + seq_len;
    let q_pos = Tensor::arange(start_pos as u32, total_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((seq_len, 1))?;
    let k_pos = Tensor::arange(0u32, total_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((1, total_len))?;
    // Mask out the upper triangle (positions where query index < key index).
    q_pos.broadcast_lt(&k_pos)
}

/// Select the best available compute device (CUDA, Metal, or CPU).
pub fn select_device() -> CandleResult<Device> {
    if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)
    } else if candle_core::utils::metal_is_available() {
        Device::new_metal(0)
    } else {
        Ok(Device::Cpu)
    }
}

/// Map a llama.cpp-style GGUF tensor name to the Hugging Face Llama naming
/// expected by [`LoraCausalLM::load`].
fn map_gguf_tensor_name(name: &str) -> String {
    let mut parts = name.split('.');
    let first = parts.next().unwrap_or(name);
    match first {
        "token_embd" => name.replace("token_embd.weight", "model.embed_tokens.weight"),
        "output" => name.replace("output.weight", "lm_head.weight"),
        "output_norm" => name.replace("output_norm.weight", "model.norm.weight"),
        "blk" => {
            // blk.{i}.attn_q.weight -> model.layers.{i}.self_attn.q_proj.weight
            let rest: Vec<&str> = parts.collect();
            if rest.len() < 2 {
                return name.to_string();
            }
            let layer = rest[0];
            let tensor = rest[1];
            let mapped = match tensor {
                "attn_norm" => "input_layernorm",
                "ffn_norm" => "post_attention_layernorm",
                "attn_q" => "self_attn.q_proj",
                "attn_k" => "self_attn.k_proj",
                "attn_v" => "self_attn.v_proj",
                "attn_output" => "self_attn.o_proj",
                "ffn_gate" => "mlp.gate_proj",
                "ffn_up" => "mlp.up_proj",
                "ffn_down" => "mlp.down_proj",
                _ => return name.to_string(),
            };
            format!("model.layers.{layer}.{mapped}.weight")
        }
        _ => name.to_string(),
    }
}

/// Load a quantized GGUF base model, dequantize the weights, and build a
/// standard `VarBuilder` that [`LoraCausalLM::load`] can consume.
pub fn load_quantized_base<'a>(
    path: &Path,
    device: &'a Device,
    dtype: DType,
) -> CandleResult<VarBuilder<'a>> {
    use candle_core::quantized::gguf_file;
    let mut file = std::fs::File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut tensors = HashMap::new();
    for name in content.tensor_infos.keys() {
        let qtensor = content.tensor(&mut file, name, device)?;
        let tensor = qtensor.dequantize(device)?.to_dtype(dtype)?;
        let mapped = map_gguf_tensor_name(name);
        tensors.insert(mapped, tensor);
    }
    Ok(VarBuilder::from_tensors(tensors, dtype, device))
}

#[cfg(feature = "flash-attn")]
fn flash_available(device: &Device, dtype: DType) -> bool {
    matches!(device, Device::Cuda(_)) && (dtype == DType::F16 || dtype == DType::BF16)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_available(_device: &Device, _dtype: DType) -> bool {
    false
}

#[cfg(feature = "flash-attn")]
fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, head_dim: usize) -> CandleResult<Tensor> {
    let out_dtype = q.dtype();
    let q = q.transpose(1, 2)?.to_dtype(DType::F16)?;
    let k = k.transpose(1, 2)?.to_dtype(DType::F16)?;
    let v = v.transpose(1, 2)?.to_dtype(DType::F16)?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let out = flash_attn(&q, &k, &v, scale, true)?;
    out.to_dtype(out_dtype)?.transpose(1, 2)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attention(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _head_dim: usize,
) -> CandleResult<Tensor> {
    Err(candle_core::Error::Msg(
        "flash attention requested but the flash-attn feature is not enabled".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{AdamW, Optimizer, ParamsAdamW};

    #[test]
    fn lora_linear_backward_step_updates_adapter_weight() {
        let device = Device::Cpu;
        let layer = LoraLinear::new(3, 2, 2, 4, &device).unwrap();
        let before = layer
            .lora_b
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut optimizer = AdamW::new(
            layer.vars(),
            ParamsAdamW {
                lr: 1.0,
                ..Default::default()
            },
        )
        .unwrap();
        let xs = Tensor::new(&[[1.0f32, 2.0, 3.0]], &device).unwrap();
        let ys = layer.forward(&xs).unwrap();
        let loss = ys.sqr().unwrap().sum_all().unwrap();

        optimizer.backward_step(&loss).unwrap();

        let after = layer
            .lora_b
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let delta = after
            .iter()
            .zip(before.iter())
            .map(|(after, before)| (after - before).powi(2))
            .sum::<f32>();
        assert!(delta > 0.0, "expected LoRA weight update, got {delta}");
    }

    #[test]
    fn lora_causal_lm_optimizer_updates_model_adapter_vars() {
        let device = Device::Cpu;
        let cfg = ModelConfig::tiny_for_tests(2, 4, 16);
        let vb = VarBuilder::from_tensors(
            super::super::random_tiny_base_tensors(&cfg, &device).unwrap(),
            DType::F32,
            &device,
        );
        let model = LoraCausalLM::load(vb, &cfg).unwrap();
        let before = model
            .lora_tensors()
            .into_values()
            .next()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut optimizer = AdamW::new(
            model.lora_vars(),
            ParamsAdamW {
                lr: 1.0,
                ..Default::default()
            },
        )
        .unwrap();
        let loss = model
            .lora_tensors()
            .into_values()
            .map(|tensor| tensor.sqr().unwrap().sum_all().unwrap())
            .reduce(|acc, loss| (acc + loss).unwrap())
            .unwrap();

        optimizer.backward_step(&loss).unwrap();

        let after = model
            .lora_tensors()
            .into_values()
            .next()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let delta = after
            .iter()
            .zip(before.iter())
            .map(|(after, before)| (after - before).powi(2))
            .sum::<f32>();
        assert!(delta > 0.0, "expected model LoRA var update, got {delta}");
    }
}
