import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns

def get_cache_attention_weights(module, input, output, cache_attention_weights):
    attn_output, attn_output_weights = output
    cache_attention_weights.append(attn_output_weights.detach())


def get_prefetch_attention_weights(module, input, output, prefetch_attention_weights):
    attn_output, attn_output_weights = output
    prefetch_attention_weights.append(attn_output_weights.detach())


def plot_attention_weights(attention_weights, title="Attention Weights"):
    num_layers = len(attention_weights)
    for layer_idx, layer_weights in enumerate(attention_weights):
        num_heads = layer_weights.shape[0]
        for head_idx in range(num_heads):
            attn = layer_weights[head_idx].detach().cpu().numpy()
            plt.figure(figsize=(8, 6))
            sns.heatmap(attn, cmap="viridis")
            plt.title(f"{title} - Layer {layer_idx + 1}, Head {head_idx + 1}")
            plt.xlabel("Sequence Position")
            plt.ylabel("Sequence Position")
            
            plot_filename = os.path.join("data/results/", f"{title}_Layer{layer_idx + 1}_Head{head_idx + 1}.png")
            plt.savefig(plot_filename)
            plt.close()  # Close the figure to avoid displaying it

def visualize_joint_attention(
    model, cache_pc, prefetch_pc, prefetch_page, prefetch_offset
):
    attn_cache, attn_prefetch = model.get_attention_weights(cache_pc, prefetch_pc, prefetch_page, prefetch_offset)
    plot_attention_weights(attn_cache, title="Cache Attention")
    plot_attention_weights(attn_prefetch, title="Prefetch Attention")