import numpy as np


class Pruning:
    def __init__(self):
        self.prune_ratio = 0
        self.mask = None

    def generate_mask(self, x, prune_ratio):
        self.prune_ratio = prune_ratio
        sub_x = x.cpu().numpy()
        self.mask = np.ones(sub_x.shape)
        self.mask = np.where(np.abs(sub_x) < self.prune_ratio, 0, 1)
        return self.mask

    def backward(self, dout):
        return dout * self.mask

    def get_mask(self):
        return self.mask
