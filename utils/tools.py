import torch
import numpy as np

def adjust_type1(label_freq, tro=0.5, device='cuda'):
    label_freq_array = np.array(label_freq)
    label_freq_array = label_freq_array / label_freq_array.mean()
    adjustments = label_freq_array ** tro
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(device)
    print('Create adjustment for prediction P in regularization term. Adj: {}'.format(adjustments.size()))
    return adjustments

def adjust_type2(label_freq, tro=1.0, device='cuda'):
    """compute the base probabilities"""
    label_freq_array = np.array(label_freq)
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(device)
    print('Create adjustment for logits in training. logits_adj: {}'.format(adjustments.size()))
    return adjustments
