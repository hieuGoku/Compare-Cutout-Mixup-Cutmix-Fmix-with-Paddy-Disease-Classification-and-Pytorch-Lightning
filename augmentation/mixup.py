import torch
import numpy as np

# https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
def mixup(data, targets, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = data.size()[0]

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * data + (1 - lam) * data[index, :]
    shuffled_targets = targets[index]
    return mixed_x, targets, shuffled_targets, lam

def aug_criterion(criterion, pred, targets, shuffled_targets, lam):
    return lam * criterion(pred, targets) + (1 - lam) * criterion(pred, shuffled_targets)