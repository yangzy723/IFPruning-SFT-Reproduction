#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFPruning SFT Inference Pipeline
--------------------------------
Evaluates the dynamically pruned model by computing contextual masks via the 
predictor network prior to standard autoregressive generation.
"""

import os
import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers.models.gemma.modeling_gemma import GemmaMLP
from safetensors.torch import load_file

# ==============================================================================
# Global Logging Configuration
# ==============================================================================
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] [%(process)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", 
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ifpruning_inference")

# ==============================================================================
# Module 1: Architectural Components
# ==============================================================================
class SparsityPredictor(nn.Module):
    """Routing network to compute layer-wise FFN channel importance scores."""
    def __init__(self, target_num_layers: int, target_ffn_dim: int, extractor_path: str):
        super().__init__()
        self.num_layers = target_num_layers
        self.ffn_dim = target_ffn_dim
        
        self.feature_extractor = AutoModel.from_pretrained(
            extractor_path, torch_dtype=torch.bfloat16, local_files_only=True
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        config = self.feature_extractor.config
        extractor_hidden_dim = getattr(config, "hidden_size", None) or \
                               getattr(config, "d_model", None) or \
                               getattr(config, "n_embd", None)    
        
        if extractor_hidden_dim is None:
            extractor_hidden_dim = self.feature_extractor.get_input_embeddings().weight.shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(extractor_hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, target_num_layers * target_ffn_dim)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.feature_extractor(input_ids=input_ids, attention_mask=attention_mask)
        seq_lengths = attention_mask.sum(dim=1) - 1
        last_token_states = outputs.last_hidden_state[torch.arange(input_ids.shape[0]), seq_lengths]
        return self.mlp(last_token_states).view(-1, self.num_layers, self.ffn_dim)


class GemmaDynamicMaskedFFN_Inference(nn.Module):
    """Modified FFN applying deterministic hard Top-K masking for inference."""
    def __init__(self, original_mlp: GemmaMLP, target_ffn_dim: int):
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn 
        self.target_ffn_dim = target_ffn_dim
        self.layer_scores = None 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_out = self.act_fn(self.gate_proj(x))
        up_out = self.up_proj(x)
        
        if self.layer_scores is not None:
            # Ensure spatial alignment across potential multi-GPU device maps
            scores = self.layer_scores.to(device=x.device, dtype=x.dtype)
            
            # Deterministic hard selection: bypass SoftTopK thresholding
            _, topk_idx = torch.topk(scores, self.target_ffn_dim, dim=-1)
            indicator = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
            
            mask = indicator.unsqueeze(1) 
            activated_hidden = (gate_out * up_out) * mask
        else:
            logger.warning("Layer scores not detected. Falling back to dense execution.")
            activated_hidden = gate_out * up_out
            
        return self.down_proj(activated_hidden)

# ==============================================================================
# Module 2: Checkpoint Resolution and Injection Pipeline
# ==============================================================================
class GemmaIFPruningWrapper(nn.Module):
    def __init__(self, base_model, target_ffn_dim: int, extractor_path: str):
        super().__init__()
        self.base_model = base_model
        cfg = getattr(base_model.config, "text_config", base_model.config)
        
        self.predictor = SparsityPredictor(cfg.num_hidden_layers, cfg.intermediate_size, extractor_path)
        
        # Anchor the predictor to the first device used by the base model
        target_device = next(base_model.parameters()).device
        self.predictor.to(device=target_device, dtype=base_model.dtype)
        
        self.llm_layers = [m for n, m in self.base_model.named_modules() if isinstance(m, nn.ModuleList) and hasattr(m[0], 'mlp')][0]
        
        for layer in self.llm_layers:
            layer.mlp = GemmaDynamicMaskedFFN_Inference(layer.mlp, target_ffn_dim)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def compute_and_lock_mask(self, predictor_input_ids: torch.Tensor, predictor_attention_mask: torch.Tensor):
        """Computes pruning scores once based on the prompt context and locks them into the layers."""
        with torch.no_grad():
            target_device = next(self.predictor.parameters()).device
            p_ids = predictor_input_ids.to(target_device)
            p_mask = predictor_attention_mask.to(target_device)
            
            all_layer_scores = self.predictor(p_ids, p_mask)
            for i, layer in enumerate(self.llm_layers):
                layer.mlp.layer_scores = all_layer_scores[:, i, :]
                
        logger.info(f"Contextual mask computed and locked successfully. (Target Dim: {self.llm_layers[0].mlp.target_ffn_dim})")

# ==============================================================================
# Execution Entry Point
# ==============================================================================
def main():
    checkpoint_dir = Path("./gemma-12b-ifpruning-output/checkpoint-3236")
    predictor_model_path = Path("./Qwen3.5-0.8b")
    predictor_weights_path = checkpoint_dir / "predictor_mlp.safetensors"
    
    target_dim = 4096

    if not checkpoint_dir.exists() or not predictor_weights_path.exists():
        raise FileNotFoundError("Missing checkpoint or predictor weights.")

    logger.info("Initializing tokenizers...")
    base_model_path = Path("./gemma-4-12b")
    base_tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), local_files_only=True)
    predictor_tokenizer = AutoTokenizer.from_pretrained(str(predictor_model_path), local_files_only=True)

    logger.info("Loading custom base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir), 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        local_files_only=True
    )

    try:
        embed_weight = base_model.model.language_model.embed_tokens.weight
        lm_head_weight = base_model.lm_head.weight
        if embed_weight.data_ptr() != lm_head_weight.data_ptr():
            logger.warning("检测到 LM Head 与 Embeddings 未物理绑定，正在强制同步内存...")
            base_model.lm_head.weight = embed_weight
            logger.info("权重绑定完成！")
    except Exception as e:
        logger.error(f"强制权重绑定失败，请检查架构路径: {e}")

    logger.info("Injecting dynamic activation sparsity architecture...")
    model = GemmaIFPruningWrapper(base_model, target_dim, str(predictor_model_path))

    logger.info("Restoring decoupled predictor parameters from safetensors...")
    pred_state_dict = load_file(str(predictor_weights_path))
    model.predictor.mlp.load_state_dict(pred_state_dict, strict=True)
    del pred_state_dict
    
    model.eval()
    logger.info("Inference pipeline operational.")
    
    instruction = "Tell me something about China and Jiangxi Province"
    
    # 1. Base Model 必须使用标准的 Chat Template 拼接对话结构
    chat_tpl = (
        "{% for m in messages %}"
        "{{'<|turn>' + m['role'] + '\\n' + m['content'] + '<turn|>\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{'<|turn>model\\n'}}{% endif %}"
    )
    messages = [{"role": "user", "content": instruction}]
    base_prompt = base_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        chat_template=chat_tpl
    )
    base_inputs = base_tokenizer(base_prompt, return_tensors="pt", add_special_tokens=False)
    
    # 2. Predictor 必须且只能输入原始的 instruction 纯文本 (与 train.py 完全对齐)
    pred_inputs = predictor_tokenizer(instruction, return_tensors="pt", add_special_tokens=True)
    
    # 计算掩码
    model.compute_and_lock_mask(
        predictor_input_ids=pred_inputs["input_ids"],
        predictor_attention_mask=pred_inputs["attention_mask"]
    )

    logger.info("Initiating autoregressive generation...")
    input_device = next(base_model.parameters()).device
    base_inputs = {k: v.to(input_device) for k, v in base_inputs.items()}
    
    with torch.no_grad():
        outputs = model.base_model.generate(
            **base_inputs, 
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True
        )
    
    input_length = base_inputs["input_ids"].shape[1]
    response = base_tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    print("\n" + "=" * 80)
    print(f"User Prompt:\n{instruction}")
    print("-" * 80)
    print(f"Sparse Model Response:\n{response}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()