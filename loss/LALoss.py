import torch
import torch.nn as nn

import numpy as np



class LALoss(nn.Module):
    def __init__(self, weight=None, **kwargs):
        super(LALoss, self).__init__()
        self.logits_adj = weight
        self.CE = nn.CrossEntropyLoss()

    def forward(self, logits, targets, *args):
        return self.CE(logits + self.logits_adj, targets), None

