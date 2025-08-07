import torch

from torch import nn
import numpy as np

from jl.models.contrastive_encoder import ContrastiveEncoder
from jl.models.transformer_encoder import TransformerEncoder, JointTransformerEncoder

class CacheReplacementNN(nn.Module):
    def __init__(self, num_features, hidden_dim, contrastive_encoder=None):
        super(CacheReplacementNN, self).__init__()
        if contrastive_encoder is not None:
            # for param in contrastive_encoder.parameters():
            #     param.requires_grad = False

            self.network = nn.Sequential(
                contrastive_encoder,
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.network = nn.Sequential(
                ContrastiveEncoder(num_features, hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(hidden_dim, 1),
            )

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.network(x)

# Concat embedder features
class CacheReplacementNNConcatEmbeddings(nn.Module):
    def __init__(self, num_features, hidden_dim, contrastive_encoder=None):
        super(CacheReplacementNNConcatEmbeddings, self).__init__()
        if contrastive_encoder is not None:
            for param in contrastive_encoder.parameters():
                param.requires_grad = False

            self.contrastive_encoder = contrastive_encoder
        else:
            self.contrastive_encoder = ContrastiveEncoder(
                num_features, hidden_dim, hidden_dim
            )

        self.network = nn.Sequential(
            nn.Linear(num_features + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, 1),
        )

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        input_embeddings = self.contrastive_encoder(x)
        x = torch.cat((x, input_embeddings), dim=1)
        return self.network(x)

class CacheReplacementNNTransformer(nn.Module):
    def __init__(self, num_features, hidden_dim, contrastive_encoder=None):
        super(CacheReplacementNNTransformer, self).__init__()
        if contrastive_encoder is not None:
            for param in contrastive_encoder.parameters():
                param.requires_grad = False

            self.network = nn.Sequential(
                contrastive_encoder,
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.network = nn.Sequential(
                TransformerEncoder(num_features, hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(hidden_dim, 1),
            )

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.network(x)
    

class CacheReplacementNNJointTransformer(nn.Module):
    def __init__(self, num_features, hidden_dim, contrastive_encoder=None):
        super(CacheReplacementNNJointTransformer, self).__init__()
        self.transformer_encoder = JointTransformerEncoder(num_features, hidden_dim, hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, 1),
        )

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, cache_pc, prefetch_pc, prefetch_page, prefetch_offset):
        # First, pass the multiple features through the TransformerEncoder
        cache_vec, prefetch_vec = self.transformer_encoder(cache_pc, prefetch_pc, prefetch_page, prefetch_offset)
        combined = torch.cat([cache_vec, prefetch_vec], dim=-1)

        # Then pass the output from the TransformerEncoder through the rest of the network
        output = self.network(combined)

        return output
    
    def get_attention_weights(self, cache_pc, prefetch_pc, prefetch_page, prefetch_offset):
        return self.transformer_encoder.get_attention_weights(cache_pc, prefetch_pc, prefetch_page, prefetch_offset)
