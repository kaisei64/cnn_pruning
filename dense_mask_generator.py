import numpy as np


class DenseMaskGenerator:
    def __init__(self):
        self.prune_ratio = 0
        self.mask = None
        self.bias_mask = None
        self.count = 0

    def generate_mask(self, x, prune_ratio):
        self.prune_ratio = prune_ratio
        sub_x = x.cpu().numpy()
        self.mask = np.ones(sub_x.shape)
        self.mask = np.where(np.abs(sub_x) < self.prune_ratio, 0, 1)
        return self.mask

    # 2層目の枝刈りに付随して消える1層目の枝を消す
    # def new_mask(self, x, y, premask):
    #     mask = premask
    #     for i, j in enumerate(y):
    #         if np.all(self.mask[i] == 0):
    #             for k, l in enumerate(mask):
    #                 mask[k][i] = 0
    #     return mask

    # def get_bias_mask(self):
    #     self.bias_mask = np.ones(self.mask.shape[1])
    #     for i, j in enumerate(self.mask[0]):
    #         if j == 0:
    #             for k, l in enumerate(self.mask):
    #                 if self.mask[k][i] != 0:
    #                     break
    #                 if k == len(self.mask) - 1:
    #                     self.bias_mask[i] = 0
    #     return self.bias_mask

    def neuron_number(self, x):
        self.count = 0
        for i, j in enumerate(x):
            if np.all(self.mask[i] == 0):
                self.count += 1
        return self.count
