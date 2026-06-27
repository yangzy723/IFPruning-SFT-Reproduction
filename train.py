#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust IFPruning SFT Trainer.

Objective: Implement the Supervised Fine-Tuning (SFT) phase of IFPruning.
1) A lightweight predictor reads the user prompt and outputs channel-wise FFN scores for each layer.
2) The SoftTopK operator generates a dynamic mask retaining a fixed number of activation channels.
3) The masked Large Language Model (LLM) is trained on response tokens using standard next-token prediction loss.
4) The LLM and the predictor's MLP head are optimized concurrently, while the predictor's backbone remains frozen.
5) Ensure absolute state safety and complete checkpoint recovery in distributed DeepSpeed environments.
6) Blends Alpaca and OpenHermes datasets directly from local disk.
"""

import os
import argparse
import inspect
import datetime as _dt
import json
import logging
import math
import shutil
import signal
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset, concatenate_datasets
from safetensors.torch import save_file, load_file
from deepspeed.ops.adam import FusedAdam
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# =============================================================================
# 0. Distributed Environment & Logging Initialization
# =============================================================================
def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

RANK = env_int("RANK", env_int("LOCAL_RANK", 0))
LOCAL_RANK = env_int("LOCAL_RANK", 0)
WORLD_SIZE = env_int("WORLD_SIZE", 1)
IS_RANK0 = (RANK == 0)

class RankFilter(logging.Filter):
    """Filters log records to inject distributed rank information."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = RANK
        record.local_rank = LOCAL_RANK
        record.world_size = WORLD_SIZE
        return True

def setup_logging(output_dir: str, log_level: str = "INFO") -> Tuple[logging.Logger, Path]:
    """Initializes a concurrent-safe logging directory and logger instance."""
    out = Path(output_dir)
    log_dir = out / "logs"
    
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Critical error: Failed to create log directory at {log_dir}. Exception: {e}", file=sys.stderr)
        raise

    if not log_dir.is_dir():
        raise RuntimeError(f"Expected directory, but found file at: {log_dir!s}")
        
    logger = logging.getLogger("ifpruning_sft")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.addFilter(RankFilter())

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [rank=%(rank)s/%(world_size)s pid=%(process)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        fh = logging.FileHandler(log_dir / f"rank_{RANK}.log", mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        logger.addHandler(fh)
    except Exception as e:
        print(f"Critical error: Failed to initialize FileHandler. Exception: {e}", file=sys.stderr)
        raise

    if IS_RANK0:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        logger.addHandler(sh)

    return logger, log_dir

LOGGER = logging.getLogger("ifpruning_sft")

def rank0_json_dump(path: Path, obj: Any) -> None:
    """Safely dumps JSON objects strictly on Rank 0 to prevent file corruption."""
    if not IS_RANK0:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.replace(path)
    except Exception as e:
        LOGGER.error(f"Failed to dump JSON to {path}. Exception: {e}", exc_info=True)

# =============================================================================
# 1. Hyperparameter Configuration
# =============================================================================
@dataclass
class RunConfig:
    base_model: str = "./gemma-4-12b"
    predictor_model: str = "./Qwen3.5-0.8b"
    output_dir: str = "./gemma-12b-ifpruning-output"
    
    dataset_alpaca: str = "./alpaca-cleaned/alpaca_data_cleaned.json"
    dataset_hermes: str = "./OpenHermes-2.5/openhermes2_5.json"
    hermes_sample_size: int = 100000
    cache_dir: str = "./hf_cache"
    
    local_files_only: bool = True

    target_intermediate_dim: int = 4096
    max_seq_length: int = 2048
    max_response_length: int = 512
    max_predictor_length: int = 1024

    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 1
    max_steps: int = -1
    base_lr: float = 2e-6
    predictor_lr: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    mask_warmup_steps: int = 1000
    mask_temperature: float = 1.0
    softtopk_iters: int = 32
    hard_mask_eval: bool = True
    abort_on_zero_loss_steps: int = 5

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    attn_implementation: str = "sdpa"
    deepspeed: bool = True
    zero_stage: int = 2

    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    dataloader_num_workers: int = 2
    preprocessing_num_proc: int = 8
    preprocessing_batch_size: int = 1000
    sanity_sample_count: int = 128
    seed: int = 42
    report_to: str = "none"
    resume: str = "auto"
    prompt_template: str = "auto"
    gemma_user_role: str = "user"
    gemma_assistant_role: str = "model"

def parse_args() -> RunConfig:
    """Parses command-line arguments mapped to the RunConfig dataclass."""
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for k, v in asdict(RunConfig()).items():
        if type(v) == bool:
            p.add_argument(f"--{k}", action=argparse.BooleanOptionalAction, default=v)
        else:
            p.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
    return RunConfig(**vars(p.parse_known_args()[0]))

def make_deepspeed_config(cfg: RunConfig, log_dir: Path) -> Optional[str]:
    """Constructs and serializes the DeepSpeed configuration dictionary."""
    if not cfg.deepspeed:
        return None
        
    ds = {
        "bf16": {
            "enabled": bool(cfg.bf16)
        },
        "fp16": {
            "enabled": bool(cfg.fp16)
        },
        "zero_optimization": {
            "stage": cfg.zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "train_batch_size": "auto",
        "zero_allow_untested_optimizer": True,
        "wall_clock_breakdown": False,
    }
    
    if cfg.zero_stage >= 2:
        ds["zero_optimization"].update({
            "allgather_partitions": True,
            "reduce_scatter": True
        })

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        if not log_dir.is_dir():
            raise RuntimeError(f"Expected directory, found file at: {log_dir!s}")
            
        path = log_dir / f"ds_config.rank{RANK}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(ds, f, ensure_ascii=False, indent=2, sort_keys=True)
        return str(path)
    except Exception as e:
        LOGGER.error(f"Failed to generate DeepSpeed configuration. Exception: {e}", exc_info=True)
        raise

# =============================================================================
# 2. Core Model Components
# =============================================================================
class SparsityPredictor(nn.Module):
    """
    Predictor module. Freezes the backbone and routes scores via a 2-layer MLP 
    to map pruning channels.
    """
    def __init__(
        self,
        num_layers: int,
        ffn_dim: int,
        extractor_path: str,
        local_files_only: bool,
        cache_dir: Optional[str]
    ):
        super().__init__()
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        
        try:
            self.extractor = AutoModel.from_pretrained(
                extractor_path,
                torch_dtype=torch.bfloat16,
                local_files_only=local_files_only,
                cache_dir=cache_dir
            )
        except Exception as e:
            LOGGER.error(f"Failed to load predictor backbone from {extractor_path}.", exc_info=True)
            raise

        if hasattr(self.extractor.config, "use_cache"):
            self.extractor.config.use_cache = False
            
        self.extractor.eval()
        for p in self.extractor.parameters():
            p.requires_grad_(False)

        hidden_size = getattr(self.extractor.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = self.extractor.get_input_embeddings().weight.shape[1]
            
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, self.num_layers * self.ffn_dim)
        )
        
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.mlp[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.extractor.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.extractor(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )
            
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        seq_lens = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        
        last = hidden[torch.arange(input_ids.shape[0], device=input_ids.device), seq_lens]
        last = last.to(dtype=self.mlp[0].weight.dtype)
        
        return self.mlp(last).view(-1, self.num_layers, self.ffn_dim)

class BoundedSoftTopK(nn.Module):
    """
    Differentiable topological operator based on binary search to bound the threshold tau.
    """
    def __init__(self, k: int, temperature: float = 1.0, iters: int = 32, hard_mask_eval: bool = True):
        super().__init__()
        self.k = int(k)
        self.temp = max(float(temperature), 1e-6)
        self.iters = int(iters)
        self.hard_mask_eval = bool(hard_mask_eval)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_fp32 = z.float() / self.temp
        
        with torch.no_grad():
            lo = z_fp32.min(dim=-1, keepdim=True).values - 20.0
            hi = z_fp32.max(dim=-1, keepdim=True).values + 20.0
            
            for _ in range(self.iters):
                mid = (lo + hi) * 0.5
                s = torch.sigmoid(z_fp32 - mid).sum(dim=-1, keepdim=True)
                lo = torch.where(s > self.k, mid, lo)
                hi = torch.where(s > self.k, hi, mid)
            tau = (lo + hi) * 0.5

        lam = torch.sigmoid(z_fp32 - tau)
        indicator = torch.zeros_like(lam).scatter_(-1, torch.topk(lam, self.k, dim=-1).indices, 1.0)
        
        if not self.training and self.hard_mask_eval:
            return indicator.to(dtype=z.dtype)
        
        # Straight-Through Estimator (STE) logic
        return (lam + (indicator - lam).detach()).to(dtype=z.dtype)

class DynamicMaskedFFN(nn.Module):
    """
    Injection layer targeting the native LLM's FFN, applying dynamic pruning gates.
    """
    def __init__(
        self,
        mlp: nn.Module,
        target_dim: int,
        mask_temperature: float,
        softtopk_iters: int,
        hard_mask_eval: bool
    ):
        super().__init__()
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn
        
        self.mask_op = BoundedSoftTopK(
            target_dim,
            mask_temperature,
            softtopk_iters,
            hard_mask_eval
        )
        
        self.layer_scores: Optional[torch.Tensor] = None
        self.register_buffer("mask_alpha", torch.tensor(0.0, dtype=torch.float32), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        
        if self.layer_scores is not None:
            mask = self.mask_op(self.layer_scores).to(dtype=hidden.dtype).unsqueeze(1)
            alpha_val = float(self.mask_alpha.item())
            
            # Defense mechanism: Break computation graph deadlocks when alpha=0
            if alpha_val >= 1.0: 
                hidden = hidden * mask
            elif alpha_val <= 0.0: 
                hidden = hidden + (mask * 0.0).to(hidden.dtype)
            else: 
                hidden = hidden.mul(1.0 - alpha_val).add(hidden.mul(mask), alpha=alpha_val)
                
        return self.down_proj(hidden)

# =============================================================================
# 3. Model Patching Pipeline
# =============================================================================
def patch_model_for_ifpruning(base_model: nn.Module, cfg: RunConfig) -> Tuple[nn.Module, nn.ModuleList]:
    """Modifies the base model architecture in-place to support IFPruning."""
    try:
        llm_cfg = getattr(base_model.config, "text_config", base_model.config)
        num_layers = int(getattr(llm_cfg, "num_hidden_layers"))
        ffn_dim = int(getattr(llm_cfg, "intermediate_size"))
        
        layers = max(
            [m for m in base_model.modules() if isinstance(m, nn.ModuleList) and hasattr(m[0], "mlp")],
            key=len
        )
    except Exception as e:
        LOGGER.error("Failed to parse base model config or extract layers.", exc_info=True)
        raise

    # 1. In-place replacement of the FFN structure
    for layer in layers:
        layer.mlp = DynamicMaskedFFN(
            layer.mlp,
            cfg.target_intermediate_dim,
            cfg.mask_temperature,
            cfg.softtopk_iters,
            cfg.hard_mask_eval
        )

    # 2. Append predictor to allow DeepSpeed to flatten memory topology
    base_model.predictor = SparsityPredictor(
        num_layers,
        ffn_dim,
        cfg.predictor_model,
        cfg.local_files_only,
        cfg.cache_dir
    )
    
    orig_forward = base_model.forward

    # 3. Hijack the top-level forward method
    def ifp_forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        predictor_input_ids=None,
        predictor_attention_mask=None,
        **kwargs
    ):
        kwargs.pop("use_cache", None)

        # Defense mechanism: Explicitly check `is not None` to avoid implicit boolean evaluation on Tensors
        p_ids = predictor_input_ids if predictor_input_ids is not None else input_ids
        p_mask = predictor_attention_mask if predictor_attention_mask is not None else attention_mask
        
        scores = self.predictor(p_ids, p_mask)
        for i, layer in enumerate(layers): 
            layer.mlp.layer_scores = scores[:, i, :]
            
        return orig_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            **kwargs
        )

    def set_mask_alpha(self, alpha: float):
        for layer in layers:
            layer.mlp.mask_alpha.fill_(float(max(0.0, min(1.0, alpha))))

    # 4. Decouple and export predictor parameters during save operations
    orig_save = base_model.save_pretrained
    
    def ifp_save_pretrained(self, save_directory: str, state_dict=None, **kwargs):
        if state_dict is None: 
            state_dict = self.state_dict()
            
        # Defense mechanism: In certain DeepSpeed configurations (ZeRO-3), non-rank0 states 
        # may yield an empty dictionary. Exit early to avoid traversal crashes.
        if not state_dict:
            orig_save(save_directory, state_dict=state_dict, **kwargs)
            return
            
        base_state = {k: v for k, v in state_dict.items() if not k.startswith("predictor.")}
        orig_save(save_directory, state_dict=base_state, **kwargs)
        
        if IS_RANK0:
            pred_state = {}
            for k, v in state_dict.items():
                if k.startswith("predictor.mlp."):
                    clean_key = k.replace("predictor.mlp.", "")
                    
                    # Defense mechanism: Validate tensor types against DeepSpeed parameter wrappers
                    if isinstance(v, torch.Tensor):
                        pred_state[clean_key] = v.cpu().contiguous()
                    else:
                        pred_state[clean_key] = v
                        
            # Defensive write: Ensure parameters were extracted before initiating I/O operations
            if pred_state:
                try:
                    save_path = Path(save_directory) / "predictor_mlp.safetensors"
                    save_file(pred_state, str(save_path))
                    
                    config_path = Path(save_directory) / "ifpruning_config.json"
                    rank0_json_dump(config_path, {"target_intermediate_dim": cfg.target_intermediate_dim})
                except Exception as e:
                    LOGGER.error("Failed to write predictor state to disk.", exc_info=True)
    
    base_model.forward = types.MethodType(ifp_forward, base_model)
    base_model.set_mask_alpha = types.MethodType(set_mask_alpha, base_model)
    base_model.save_pretrained = types.MethodType(ifp_save_pretrained, base_model)
    
    return base_model, layers

# =============================================================================
# 4. Data Processing Pipeline
# =============================================================================
def extract_prompt_response(examples: Dict, i: int) -> Tuple[str, str]:
    """Safely extracts prompt and response strings from varying dataset structures."""
    try:
        # Compatibility for ShareGPT/OpenHermes ("conversations") & standard ("messages")
        if "messages" in examples or "conversations" in examples:
            msg_key = "messages" if "messages" in examples else "conversations"
            raw_msg = examples[msg_key][i]
            msgs = json.loads(raw_msg) if isinstance(raw_msg, str) else raw_msg
            
            u = next((m.get("content", m.get("value", "")) for m in msgs 
                      if m.get("role", m.get("from", "")) in {"user", "human"}), "")
            a = next((m.get("content", m.get("value", "")) for m in reversed(msgs) 
                      if m.get("role", m.get("from", "")) in {"assistant", "gpt", "model"}), "")
            return u.strip(), a.strip()
        
        # Compatibility for Alpaca standard structure
        if "instruction" in examples and "output" in examples:
            inst = examples["instruction"][i].strip()
            inp = examples.get("input", [""] * len(examples["instruction"]))[i].strip()
            return f"{inst}\n{inp}".strip() if inp else inst, examples["output"][i].strip()
            
        for pk, rk in [("prompt", "response"), ("question", "answer"), ("query", "answer"), ("input", "output")]:
            if pk in examples and rk in examples:
                return examples[pk][i].strip(), examples[rk][i].strip()
                
    except Exception as e:
        LOGGER.error(f"Failed to parse entry index {i}. Exception: {e}", exc_info=True)
        
    return "", ""

def tokenize_sft_dataset(
    raw: Dataset,
    b_tok: PreTrainedTokenizerBase,
    p_tok: PreTrainedTokenizerBase,
    cfg: RunConfig,
    t_args: TrainingArguments
) -> Dataset:
    """Preprocesses and tokenizes the dataset into model-consumable tensors."""
    if b_tok.pad_token_id is None:
        b_tok.pad_token = b_tok.eos_token
    if p_tok.pad_token_id is None:
        p_tok.pad_token = p_tok.eos_token
        
    b_tok.padding_side = "right"
    
    chat_tpl = getattr(b_tok, "chat_template", None)
    if not chat_tpl:
        chat_tpl = (
            "{% for m in messages %}"
            "{{'<|turn>' + m['role'] + '\\n' + m['content'] + '<turn|>\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{'<|turn>model\\n'}}{% endif %}"
        )
        
    vocab = getattr(b_tok, "get_vocab", lambda: {})()
    stop_id = b_tok.convert_tokens_to_ids("<turn|>") if "<turn|>" in vocab else b_tok.eos_token_id

    def process_batch(examples):
        out = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "predictor_input_ids": [],
            "predictor_attention_mask": [],
            "num_target_tokens": []
        }
        prompts = []

        batch_size = len(next(iter(examples.values())))
        for i in range(batch_size):
            p, r = extract_prompt_response(examples, i)
            if not p or not r:
                continue
            
            try:
                # Defense mechanism: Bypass inter-process null pointer exceptions with explicit string templates
                p_text = b_tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=chat_tpl
                )
                p_ids = b_tok(p_text, add_special_tokens=False)["input_ids"]
                
                r_text = r + ("<turn|>\n" if stop_id and stop_id != b_tok.eos_token_id else b_tok.eos_token)
                r_ids = b_tok(r_text, add_special_tokens=False)["input_ids"]
                r_ids = r_ids[:cfg.max_response_length]
                
                if stop_id is not None:
                    r_ids[-1] = int(stop_id)
                
                max_p_len = cfg.max_seq_length - len(r_ids)
                p_ids = p_ids[-max_p_len:] if max_p_len > 0 else p_ids[-1:]
                
                out["input_ids"].append(p_ids + r_ids)
                out["attention_mask"].append([1] * len(p_ids + r_ids))
                out["labels"].append([-100] * len(p_ids) + r_ids)
                out["num_target_tokens"].append(len(r_ids))
                prompts.append(p)
            except Exception as e:
                LOGGER.error(f"Error processing token sequence for index {i}. Exception: {e}", exc_info=True)
                continue
            
        if prompts:
            try:
                p_enc = p_tok(
                    prompts,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=cfg.max_predictor_length
                )
                out["predictor_input_ids"] = p_enc["input_ids"]
                out["predictor_attention_mask"] = p_enc["attention_mask"]
            except Exception as e:
                LOGGER.error(f"Error encoding predictor prompts. Exception: {e}", exc_info=True)
                
        return out

    with t_args.main_process_first(desc="dataset tokenization"):
        return raw.map(
            process_batch,
            batched=True,
            batch_size=cfg.preprocessing_batch_size,
            num_proc=max(1, cfg.preprocessing_num_proc),
            remove_columns=raw.column_names
        ).filter(lambda x: x["num_target_tokens"] > 0)

class DualCollator:
    """Collator mapping padded tensors for both the base model and the predictor."""
    def __init__(self, b_pad: int, p_pad: int):
        self.b_pad = b_pad
        self.p_pad = p_pad
        
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        try:
            mb = max(len(x["input_ids"]) for x in features)
            mp = max(len(x.get("predictor_input_ids", [])) for x in features)
            
            def pad(v, m, p):
                return v + [p] * (m - len(v))
                
            return {
                "input_ids": torch.tensor([pad(x["input_ids"], mb, self.b_pad) for x in features], dtype=torch.long),
                "attention_mask": torch.tensor([pad(x["attention_mask"], mb, 0) for x in features], dtype=torch.long),
                "labels": torch.tensor([pad(x["labels"], mb, -100) for x in features], dtype=torch.long),
                "predictor_input_ids": torch.tensor([pad(x["predictor_input_ids"], mp, self.p_pad) for x in features], dtype=torch.long),
                "predictor_attention_mask": torch.tensor([pad(x["predictor_attention_mask"], mp, 0) for x in features], dtype=torch.long),
            }
        except Exception as e:
            LOGGER.error(f"Collation failed during batch formation. Exception: {e}", exc_info=True)
            raise

# =============================================================================
# 5. Trainer & Callbacks
# =============================================================================
class IFPruningTrainer(Trainer):
    """Custom Trainer implementing distinct learning rates for base and predictor networks."""
    def __init__(self, *args, p_lr: float, b_lr: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_lr = p_lr
        self.b_lr = b_lr

    def _is_no_decay(self, name: str, param: nn.Parameter) -> bool:
        return param.ndim < 2 or any(k in name.lower() for k in ["bias", "norm", "ln"])

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
            
        groups = {
            "pred_decay": [],
            "pred_nd": [],
            "base_decay": [],
            "base_nd": []
        }

        # Defense mechanism: Ensure stable ZeRO partition alignment
        for name, param in sorted(self.model.named_parameters(), key=lambda x: x[0]):
            if not param.requires_grad: 
                continue
                
            is_pred = "predictor.mlp" in name
            is_nd = self._is_no_decay(name, param)
            
            group_key = f"{'pred' if is_pred else 'base'}_{'nd' if is_nd else 'decay'}"
            groups[group_key].append(param)
            
        optim_groups = [
            {"params": groups["pred_decay"], "lr": self.p_lr, "weight_decay": self.args.weight_decay},
            {"params": groups["pred_nd"], "lr": self.p_lr, "weight_decay": 0.0},
            {"params": groups["base_decay"], "lr": self.b_lr, "weight_decay": self.args.weight_decay},
            {"params": groups["base_nd"], "lr": self.b_lr, "weight_decay": 0.0},
        ]

        try:
            self.optimizer = FusedAdam(optim_groups, betas=(self.args.adam_beta1, self.args.adam_beta2))
        except Exception as e:
            LOGGER.error("Failed to initialize FusedAdam optimizer.", exc_info=True)
            raise
            
        return self.optimizer

class IFPruningCallback(TrainerCallback):
    """Callback evaluating and managing the dynamic mask alpha schedule."""
    def __init__(self, layers: nn.ModuleList, warmup: int, abort_zero: int):
        self.layers = layers
        self.warmup = warmup
        self.abort = abort_zero
        self.zero_streak = 0

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model:
            alpha = 1.0 if self.warmup <= 0 else min(1.0, max(0.0, state.global_step / self.warmup))
            getattr(model.module if hasattr(model, "module") else model, "set_mask_alpha", lambda a: None)(alpha)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
            
        loss = logs.get("loss")
        if loss is not None:
            if not math.isfinite(loss):
                control.should_training_stop = True
            elif loss <= 1e-8 and state.global_step > 1:
                self.zero_streak += 1
                if self.abort > 0 and self.zero_streak >= self.abort:
                    control.should_training_stop = True
            else:
                self.zero_streak = 0
            
        if IS_RANK0 and "loss" in logs:
            logs["mask_alpha"] = float(self.layers[0].mlp.mask_alpha.item()) if self.layers else 0.0
            LOGGER.info(
                f"Step {state.global_step} | "
                f"Loss={logs.get('loss', 0):.4f} | "
                f"LR={logs.get('learning_rate', 0):.3e} | "
                f"Alpha={logs['mask_alpha']:.4f}"
            )

# =============================================================================
# 6. Pipeline Execution
# =============================================================================
def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)
    
    cfg = parse_args()
    global LOGGER
    LOGGER, log_dir = setup_logging(cfg.output_dir)
    set_seed(cfg.seed)

    def sig_handler(s, f):
        sys.exit(0)
        
    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)

    rank0_json_dump(log_dir / "run_config.json", asdict(cfg))

    ta_kwargs = {
        "output_dir": cfg.output_dir,
        "do_train": True,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "num_train_epochs": cfg.num_train_epochs,
        "max_steps": cfg.max_steps,
        "learning_rate": cfg.base_lr,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "bf16": cfg.bf16,
        "fp16": cfg.fp16,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "safe_serialization": True, # FIXED: Replaced save_safetensors with safe_serialization
        "report_to": [] if cfg.report_to == "none" else cfg.report_to.split(","),
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "deepspeed": make_deepspeed_config(cfg, log_dir),
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if cfg.gradient_checkpointing else None
    }
    
    valid_args = inspect.signature(TrainingArguments.__init__).parameters
    t_args = TrainingArguments(**{k: v for k, v in ta_kwargs.items() if k in valid_args})

    LOGGER.info("Initializing Tokenizers...")
    try:
        b_tok = AutoTokenizer.from_pretrained(
            cfg.base_model,
            local_files_only=cfg.local_files_only,
            use_fast=True
        )
        p_tok = AutoTokenizer.from_pretrained(
            cfg.predictor_model,
            local_files_only=cfg.local_files_only,
            use_fast=True
        )
        
        LOGGER.info("Parsing RAW JSON datasets from local disk...")
        alpaca_raw = load_dataset(
            "json", 
            data_files=cfg.dataset_alpaca, 
            split="train", 
            cache_dir=cfg.cache_dir
        )
        hermes_raw = load_dataset(
            "json", 
            data_files=cfg.dataset_hermes, 
            split="train", 
            cache_dir=cfg.cache_dir
        )
        
        safe_sample_size = min(cfg.hermes_sample_size, len(hermes_raw))
        hermes_raw = hermes_raw.shuffle(seed=cfg.seed).select(range(safe_sample_size))
        
        LOGGER.info("Tokenizing and Caching Data Streams...")
        alpaca_tok = tokenize_sft_dataset(alpaca_raw, b_tok, p_tok, cfg, t_args)
        hermes_tok = tokenize_sft_dataset(hermes_raw, b_tok, p_tok, cfg, t_args)
        
        tokenized_dataset = concatenate_datasets([alpaca_tok, hermes_tok]).shuffle(seed=cfg.seed)
        LOGGER.info(f"Dataset preparation complete. Total blended samples: {len(tokenized_dataset)}")
        # ---------------------------------------------------------
        
    except Exception as e:
        LOGGER.error("Failed during Tokenizer initialization or Dataset loading.", exc_info=True)
        raise

    LOGGER.info("Initializing and Patching Base Model architecture...")
    try:
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if cfg.bf16 else torch.float16,
            "local_files_only": cfg.local_files_only,
            "attn_implementation": cfg.attn_implementation
        }
        base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
        model, layers = patch_model_for_ifpruning(base_model, cfg)
        
        if cfg.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    except Exception as e:
        LOGGER.error("Failed during Model load or architectural patching.", exc_info=True)
        raise

    trainer = IFPruningTrainer(
        model=model, 
        args=t_args, 
        train_dataset=tokenized_dataset, 
        data_collator=DualCollator(b_tok.pad_token_id, p_tok.pad_token_id),
        callbacks=[IFPruningCallback(layers, cfg.mask_warmup_steps, cfg.abort_on_zero_loss_steps)],
        p_lr=cfg.predictor_lr, 
        b_lr=cfg.base_lr
    )

    resume_ckpt = None
    if cfg.resume == "auto" and os.path.isdir(cfg.output_dir):
        resume_ckpt = get_last_checkpoint(cfg.output_dir)
    elif cfg.resume != "none":
        resume_ckpt = cfg.resume
    
    if resume_ckpt:
        pred_ckpt_path = Path(resume_ckpt) / "predictor_mlp.safetensors"
        if pred_ckpt_path.exists():
            LOGGER.info(f"Initiating safety restoration of predictor parameters from: {pred_ckpt_path}")
            try:
                ckpt_state = load_file(str(pred_ckpt_path))
                
                # Defensive constraint: Log alignment warnings to prevent silent partial restorations
                load_result = model.predictor.mlp.load_state_dict(ckpt_state, strict=False)
                if load_result.missing_keys:
                    LOGGER.warning(f"Predictor restoration missing keys: {load_result.missing_keys}")
                if load_result.unexpected_keys:
                    LOGGER.warning(f"Predictor restoration unexpected keys: {load_result.unexpected_keys}")
            except Exception as e:
                LOGGER.error(f"Failed to read or load predictor parameters from {pred_ckpt_path}.", exc_info=True)
                raise
        else:
            # Defensive constraint: Absolute zero-tolerance policy for decoupled states.
            LOGGER.error(f"Critical integrity failure: Checkpoint directory {resume_ckpt} exists, "
                         f"but predictor weights '{pred_ckpt_path.name}' are missing.")
            raise FileNotFoundError("Checkpoint payload corrupted. Halting process to prevent state divergence.")

    LOGGER.info(f"{'Resuming state from ' + str(resume_ckpt) if resume_ckpt else 'Initiating fresh training trajectory'}...")
    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    except Exception as e:
        LOGGER.error("Fatal error encountered during Trainer execution.", exc_info=True)
        raise
    
    try:
        trainer.save_model(cfg.output_dir)
        if IS_RANK0:
            b_tok.save_pretrained(cfg.output_dir)
            p_tok.save_pretrained(str(Path(cfg.output_dir) / "predictor_tokenizer"))
    except Exception as e:
        LOGGER.error("Failed to execute final state export.", exc_info=True)
        raise
        
    LOGGER.info("Training cycle complete.")

if __name__ == "__main__":
    main()