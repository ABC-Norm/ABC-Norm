import torch
import torch.nn as nn

class PCLoss(nn.Module):
    def __init__(self, tau=0.0, weight=None, **kwargs):
        super(PCLoss, self).__init__()
        self.tau = float(tau)
        self.weight = weight
        self.reweight = True if weight is not None else False
        print('Reweight: {}'.format(self.reweight))

        self.CE = nn.CrossEntropyLoss()

    def pc_loss(self, logits):
        y_pred = torch.softmax(logits, dim=1)
        if self.reweight:
            y_pred = y_pred * self.weight
        # features: probability
        batch_size = y_pred.size(0)
        if float(batch_size) % 2 != 0:
            raise Exception('Incorrect batch size provided')
        batch_left = y_pred[:int(0.5*batch_size)]
        batch_right = y_pred[int(0.5*batch_size):]
        loss  = torch.norm((batch_left - batch_right).abs(),2, 1).sum() / float(batch_size)
        return loss

    def forward(self, logits, targets, *args, **kwargs):
        ce = self.CE(logits, targets)
        pc = self.pc_loss(logits)
        loss = (1.0 - self.tau) * ce + self.tau * pc
        return loss, ce

