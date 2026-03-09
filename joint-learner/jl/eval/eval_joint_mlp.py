import time
import torch

from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from jl.models.mlp_replacement import (
    CacheReplacementNN,
    CacheReplacementNNTransformer,
    CacheReplacementNNJointTransformer,
)
from jl.dataloaders.joint_dataloader import get_joint_dataloader
from jl.utils import parse_args, load_config
from jl.models.contrastive_encoder import ContrastiveEncoder
from jl.data_engineering.count_labels import count_labels
from jl.vis.visualize_attention import visualize_joint_attention, plot_attention_weights
import jl.dataloaders.dataloader as dl


def eval(args):
    print(f"------------------------------")
    config = load_config(args.config)
    print(f"Config: dim {args.hidden_dim} window {args.ip_history_window}")
    print("Init Dataloader")

    _, _, dataloader, num_pcs, num_pages = get_joint_dataloader(
        args.cache_data_path,
        args.ip_history_window,
        args.prefetch_data_path,
        config,
        args.batch_size,
        name=args.dataset,
    )

    print(f"Num Prefetch PCs: {num_pcs}, Num Pages: {num_pages}")

    if args.basic_model:
        model = CacheReplacementNN(
            num_features=args.ip_history_window + config.sequence_length * 3 + 1,
            hidden_dim=args.hidden_dim,
        )
    else:
        feature_sizes = [len(dl.CACHE_IP_TO_IDX) + 1, num_pcs + 1, num_pages + 1, 65]
        model = CacheReplacementNNJointTransformer(
            num_features=feature_sizes, hidden_dim=args.hidden_dim
        )

    state_dict = torch.load(f"./data/model/{args.model_name}.pth")
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    model.eval()
    start_time = time.time()

    correct = 0
    zeroes = 0
    total = 0

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            cache_features, prefetch_pcs, prefetch_pages, prefetch_offsets, labels = (
                data
            )
            cache_features, prefetch_pcs, prefetch_pages, prefetch_offsets, labels = (
                cache_features.to(device),
                prefetch_pcs.to(device),
                prefetch_pages.to(device),
                prefetch_offsets.to(device),
                labels.to(device),
            )

            outputs = model(
                cache_features, prefetch_pcs, prefetch_pages, prefetch_offsets
            )

            total += labels.size(0)
            correct += count_correct(outputs, labels)

            probs = torch.sigmoid(outputs)
            zeroes += (probs < 0.5).sum().item()


            # if (labels.size(0) == args.batch_size):
            #     attn_cache, attn_prefetch = model.get_attention_weights(
            #         cache_features, prefetch_pcs, prefetch_pages, prefetch_offsets
            #     )
            #     # Accumulate attention weights
            #     attn_cache_total.append(torch.stack(attn_cache))
            #     attn_prefetch_total.append(torch.stack(attn_prefetch))

            if batch % 10000 == 0 and batch != 0:
                ms_per_batch = (time.time() - start_time) * 1000 / batch
                print(
                    f"batch {batch}/{len(dataloader)} | accuracy {correct}/{total} | ms/batch {ms_per_batch}"
                )

    accuracy = correct / len(dataloader.dataset) * 100

    print(f"Accuracy: {accuracy:.2f}%, Zeroes: {zeroes}")
    print(f"------------------------------")

    # avg_attn_cache = torch.mean(torch.stack(attn_cache_total), dim=0)
    # avg_attn_prefetch = torch.mean(torch.stack(attn_prefetch_total), dim=0)
    # plot_attention_weights(avg_attn_cache, title="Cache Attention")
    # plot_attention_weights(avg_attn_prefetch, title="Prefetch Attention")

    # Visualize attention weights
    # if not args.basic_model:
    #     data_iter = iter(dataloader)
    #     batch = next(data_iter)

    #     first_sample = tuple(tensor[0] for tensor in batch)

    #     print(first_sample)

    #     # cache_pc, prefetch_pc, prefetch_page, prefetch_offset, label = first_sample
    #     cache_pc = first_sample[0].unsqueeze(0)
    #     prefetch_pc = first_sample[1].unsqueeze(0)
    #     prefetch_page = first_sample[2].unsqueeze(0)
    #     prefetch_offset = first_sample[3].unsqueeze(0)
    #     label = first_sample[4].unsqueeze(0)

    #     cache_pc, prefetch_pc, prefetch_page, prefetch_offset, label = (
    #         cache_pc.to(device),
    #         prefetch_pc.to(device),
    #         prefetch_page.to(device),
    #         prefetch_offset.to(device),
    #         label.to(device),
    #     )

    #     visualize_joint_attention(
    #         model,
    #         cache_pc,
    #         prefetch_pc,
    #         prefetch_page,
    #         prefetch_offset,
    #     )


def count_correct(logits, labels):
    probs   = torch.sigmoid(logits)          # convert logits → probability
    preds   = (probs > 0.5).float()          # threshold at 0.5
    correct = preds.eq(labels).sum().item()  # count matches
    return correct

if __name__ == "__main__":
    args = parse_args()
    eval(args)
