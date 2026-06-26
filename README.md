# IFPruning-SFT: Unofficial Reproduction of AFM-3 Instruction-Following Pruning

## 1. Overview

This repository implements the Supervised Fine-Tuning (SFT) phase of the [IFPruning architecture](https://arxiv.org/abs/2501.02086). It introduces input-aware activation sparsity into large language models (LLMs) during alignment, with native support for DeepSpeed ZeRO optimization and distributed multi-GPU training.

## 2. Dynamic Activation Sparsity

IFPruning does not simply fine-tune static weights or low-rank adapters. Instead, it learns to dynamically sparsify feed-forward network (FFN) activations conditioned on the input prompt.

### 2.1 Contextual Routing

A frozen predictor backbone (for example, `Qwen3.5-0.8B`) extracts hidden states from the prompt. A trainable MLP head converts those states into continuous channel-wise importance scores:

- $S \in \mathbb{R}^{L \times D_{ffn}}$
- $L$ is the number of layers
- $D_{ffn}$ is the FFN intermediate dimension

### 2.2 Differentiable Thresholding

A strict sparsity budget $k$ is enforced by converting the importance scores into a binary mask:

- $M \in \{0, 1\}^{D_{ffn}}$
- A threshold $\tau$ is solved via binary search
- The mask remains differentiable through the backward pass using a Straight-Through Estimator (STE)

Forward and backward definitions:

$$M_{forward} = \text{TopK}(S, k)$$
$$M_{backward} = \sigma\left(\frac{S - \tau}{T}\right)$$

where $T$ controls sigmoid steepness.

### 2.3 Dynamic FFN Injection

The original FFN computation is intercepted and masked at the activation level. A warmup scalar $\alpha \in [0, 1]$ blends dense and sparse paths to stabilize early training:

$$H_{sparse} = \left((1-\alpha) \cdot H_{dense} + \alpha \cdot (M \odot H_{dense})\right) W_{down}$$
$$H_{dense} = \text{Act}(X W_{gate}) \odot (X W_{up})$$

Gradients are propagated through this sparse topology to update both the base LLM parameters and the predictor MLP.

## 3. Distributed Engineering Architecture

This codebase is designed for robust distributed training across multi-node GPU clusters.

- **Memory fragmentation mitigation:** Uses `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce CUDA allocator fragmentation and prevent OOM failures.
- **DeepSpeed ZeRO compatibility:** Registers the predictor module as a submodule of the base LLM so ZeRO-2/3 can partition gradients and optimizer states correctly.
- **Decoupled checkpointing:** Exports base LLM weights normally while saving the predictor state separately as `predictor_mlp.safetensors` from Rank-0.
- **Pre-flight validation:** Checkpoint restore performs strict integrity checks. Missing state files or incomplete optimizer slices trigger deterministic failures rather than silent corruption.

## 4. Data & Model Preparation

Pre-trained weights and datasets are not included. Download required models into the project root before training.

### 4.1 Required Models

**Base LLM (target for IFPruning):**
```bash
huggingface-cli download google/gemma-4-12B --local-dir ./gemma-4-12b
```

**Predictor backbone:**
```bash
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir ./Qwen3.5-0.8b
```

### 4.2 Expected Repository Layout

```text
AFM3/
├── gemma-4-12b/
├── Qwen3.5-0.8b/
├── gemma-12b-ifpruning-output/    # optional training output
├── train.py
├── test_ckpt.py
├── plot_loss.py
├── inference_IFP.py
└── README.md
```

> Datasets like `yahma/alpaca-cleaned` will be downloaded and cached to `./hf_cache` automatically during first training.

## 5. Execution & Deployment Guide

### 5.1 Distributed Training Launch

Start training with `torchrun` and match `nproc_per_node` to the available GPUs:

```bash
OMP_NUM_THREADS=1 \
TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=2 train.py \
    --base_model "./gemma-4-12b" \
    --predictor_model "./Qwen3.5-0.8b" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --target_intermediate_dim 4096
```

### 5.2 Checkpoint Audit Workflow

Run checkpoint validation before a full training run to verify state integrity.

**Phase 1: State checkpoint**
```bash
TEST_PHASE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=2 test_ckpt.py
```

**Phase 2: Checkpoint restore**
```bash
TEST_PHASE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=2 test_ckpt.py
```

Expected result: successful restoration of optimizer slices and loss trajectory without deadlocks or I/O corruption.

## 6. Troubleshooting

### PyTorch & CUDA version alignment

Ensure PyTorch and CUDA versions match the deployed system and the installed binary. Incorrect CUDA builds can trigger NCCL failures such as `ncclCommResume`.

Example installation for CUDA 12.4:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### Notes

- `plot_loss.py` is the training dynamics visualizer and uses the default log path `./gemma-12b-ifpruning-output/logs/rank_0.log`.
- Keep the root directory structure intact to ensure training and checkpoint scripts locate model assets correctly.
