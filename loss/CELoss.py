import torch
import torch.nn as nn

class CELoss(nn.Module):
    def __init__(self, **kwargs):
        super(CELoss, self).__init__()
        self.CE = nn.CrossEntropyLoss()

    def forward(self, logits, targets, *args):
        return self.CE(logits, targets), None

