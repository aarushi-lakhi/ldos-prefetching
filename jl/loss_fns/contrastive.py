import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        label = label.view(-1).float()

        # Calculate Euclidean distance between the output embeddings
        dist = F.pairwise_distance(output1, output2)

        pos = label * dist.pow(2)
        neg = (1.0 - label) * F.relu(self.margin - dist).pow(2)
        loss = pos + neg
        return loss.mean()
