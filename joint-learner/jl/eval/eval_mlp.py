import time
import torch
from torch import nn

from jl.models.mlp_replacement import CacheReplacementNN, CacheReplacementNNTransformer
from jl.models.transformer_encoder import TransformerEncoder
from jl.dataloaders.dataloader import get_cache_dataloader
from jl.utils import parse_args

import jl.dataloaders.dataloader as dl


args = parse_args()

def eval(args):
    print(f"------------------------------")
    print(f"Config: dim {args.hidden_dim} window {args.ip_history_window}")
    print("Init Dataloader")

    _, _, dataloader = get_cache_dataloader(args.cache_data_path, args.ip_history_window, args.batch_size)

    if args.encoder_name != "none":
        contrastive_encoder = TransformerEncoder(
            len(dl.CACHE_IP_TO_IDX) + 1, args.hidden_dim, args.hidden_dim
        )
        contrastive_encoder.load_state_dict(
            torch.load(f"./data/model/{args.encoder_name}.pth")
        )
        model = CacheReplacementNN(
            num_features=len(dl.CACHE_IP_TO_IDX) + 1,
            hidden_dim=args.hidden_dim,
            contrastive_encoder=contrastive_encoder,
        )
    elif args.basic_model:
        model = CacheReplacementNN(
            num_features=args.ip_history_window + 1, hidden_dim=args.hidden_dim
        )
    else:
        model = CacheReplacementNNTransformer(
            num_features=len(dl.CACHE_IP_TO_IDX) + 1, hidden_dim=args.hidden_dim
        )

    state_dict = torch.load(f"./data/model/{args.model_name}.pth")
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    print("Begin Eval")
    model.eval()
    start_time = time.time()

    correct = 0
    zeroes = 0
    total = 0

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            total += labels.size(0)
            correct += count_correct(outputs, labels)

            probs = torch.sigmoid(outputs)
            zeroes += (probs < 0.5).sum().item()
            
            if batch % 10000 == 0 and batch != 0:
                ms_per_batch = (time.time() - start_time) * 1000 / batch
                print(f'batch {batch}/{len(dataloader)} | accuracy {correct}/{total} | ms/batch {ms_per_batch}')

    accuracy = correct / len(dataloader.dataset) * 100

    print(f"Accuracy: {accuracy:.2f}%, Zeroes: {zeroes}")
    print(f"------------------------------")

def count_correct(logits, labels):
    probs   = torch.sigmoid(logits)          # convert logits → probability
    preds   = (probs > 0.5).float()          # threshold at 0.5
    correct = preds.eq(labels).sum().item()  # count matches
    return correct


if __name__ == "__main__":
    args = parse_args()
    eval(args)
