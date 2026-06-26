#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFPruning SFT Training Dynamics Visualizer
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt

# =============================================================================
# 1. 核心配置
# =============================================================================
LOG_PATH = Path("./gemma-12b-ifpruning-output/logs/rank_0.log")
OUTPUT_PATH = Path("./loss_curve.png")

SMOOTHING_WEIGHT = 0.85

# =============================================================================
# 2. 数据解析管道
# =============================================================================
def parse_training_log(log_path: Path):
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    pattern = re.compile(r"Step\s+(\d+)\s+\|\s+Loss=([\d.eE+-]+)\s+\|\s+LR=([\d.eE+-]+)\s+\|\s+Alpha=([\d.eE+-]+)")
    steps, losses, lrs, alphas = [], [], [], []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                lrs.append(float(match.group(3)))
                alphas.append(float(match.group(4)))

    if not steps:
        raise ValueError("No valid training metrics found in the log.")

    return steps, losses, lrs, alphas

def compute_ema(values, weight=0.85):
    smoothed = []
    for val in values:
        if not smoothed:
            smoothed.append(val)
        else:
            smoothed.append(smoothed[-1] * weight + val * (1 - weight))
    return smoothed

# =============================================================================
# 3. 绘图风格配置
# =============================================================================
def set_academic_style():
    plt.rcParams.update({
        "font.family": "serif",           # 使用衬线字体
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 14,                  # 基础字号放大
        "axes.labelsize": 16,             # 坐标轴标签字号
        "axes.titlesize": 16,             # 子图标题字号
        "legend.fontsize": 12,            # 图例字号
        "xtick.labelsize": 13,            # X轴刻度字号
        "ytick.labelsize": 13,            # Y轴刻度字号
        "xtick.direction": "in",          # 刻度朝内
        "ytick.direction": "in",          # 刻度朝内
        "axes.linewidth": 1.2,            # 边框加粗
        "lines.linewidth": 2.0,           # 默认线条加粗
        "figure.facecolor": "white",      # 纯白背景
        "axes.facecolor": "white",        # 纯白绘图区
        "grid.alpha": 0.5,                # 弱化网格
        "grid.linestyle": "--"
    })

# =============================================================================
# 4. 可视化渲染
# =============================================================================
def generate_conference_plot(steps, losses, lrs, alphas):
    set_academic_style()
    
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(10, 8), 
        dpi=300, 
        sharex=True, 
        gridspec_kw={'height_ratios': [2, 1]}
    )
    
    # 全封闭边框与网格
    for ax in (ax1, ax2):
        ax.grid(True)
        for spine in ax.spines.values():
            spine.set_color('black')

    # ---------------------------------------------------------
    # 顶部子图: Loss (深蓝与砖红)
    # ---------------------------------------------------------
    smooth_losses = compute_ema(losses, weight=SMOOTHING_WEIGHT)
    
    # 原始 Loss 用极细虚线，平滑 Loss 用深色粗实线
    ax1.plot(steps, losses, color="#0066CC", linewidth=1.5, linestyle=":", alpha=0.6, label="Batch Loss")
    ax1.plot(steps, smooth_losses, color="#C00000", linewidth=2.5, linestyle="-", label="Smoothed Loss")
    
    ax1.set_ylabel("Cross Entropy Loss")
    # 图例极简化，去掉阴影，采用纯黑细边框
    ax1.legend(loc="upper right", frameon=True, edgecolor="black", fancybox=False)

    # ---------------------------------------------------------
    # 底部子图: LR & Alpha (深绿与深紫)
    # ---------------------------------------------------------
    ax2.plot(steps, lrs, color="#548235", linewidth=2.0, linestyle="-", label="Learning Rate")
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Learning Rate")

    # 右侧 Y 轴处理 Alpha
    ax3 = ax2.twinx()
    ax3.plot(steps, alphas, color="#7030A0", linewidth=2.0, linestyle="--", label="Mask Alpha")
    ax3.set_ylabel("Pruning Alpha")
    ax3.set_ylim(-0.05, 1.05) 
    ax3.tick_params(direction="in")

    # 合并底部图例
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    ax2.legend(lines_2 + lines_3, labels_2 + labels_3, loc="center right", frameon=True, edgecolor="black", fancybox=False)

    # ---------------------------------------------------------
    # 布局收尾
    # ---------------------------------------------------------
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.08) 
    fig.savefig(OUTPUT_PATH, bbox_inches='tight')
    fig.savefig(OUTPUT_PATH.with_suffix(".png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Academic plot generated: {OUTPUT_PATH}")
    print(f"Total steps processed: {len(steps)}")

if __name__ == "__main__":
    try:
        data_steps, data_losses, data_lrs, data_alphas = parse_training_log(LOG_PATH)
        generate_conference_plot(data_steps, data_losses, data_lrs, data_alphas)
    except Exception as e:
        print(f"Execution Failed: {e}")