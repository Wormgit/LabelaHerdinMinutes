# Core libraries
import numpy as np
from itertools import combinations

# PyTorch
import torch

"""
File contains a selection of mining utilities for selecting triplets
Code is adapted from - https://github.com/adambielski/siamese-triplet
"""

# Find the distance between two vectors
def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

# Should return indices of selected anchors, positive and negative samples
class TripletSelector:
    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels, sub_p):
        raise NotImplementedError


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """
    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels, sub_p, labels_neg, sub_n):
                                     #positive negtive     p
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        labels_neg = labels_neg.cpu().data.numpy()
        sub_p = sub_p.cpu().data.numpy()
        sub_n = sub_n.cpu().data.numpy()
        PosandNeg = np.concatenate((labels, labels_neg), axis=0)
        PosandNeg_sub = np.concatenate((sub_p, sub_n), axis=0)

        triplets = []
        for i in np.unique(PosandNeg):
            label_mask_first = (PosandNeg == i)
            label_indices_first = np.where(label_mask_first)[0]
            if len(label_indices_first) < 2:
                continue
            sub_label = [PosandNeg_sub[m,0] for m in label_indices_first]

            if len(set(sub_label)) == 1:
                label_mask = label_mask_first
                label_indices = label_indices_first
            else:
                continue

            if len(label_indices) > 2:
                pass

            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = np.array(list(combinations(label_indices, 2)))  # All anchor-positive pairs in this batch?

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                if loss_values.size != 0:
                    hard_negative = self.negative_selection_fn(loss_values) ### find one with big loss
                    if hard_negative is not None:
                        hard_negative = negative_indices[hard_negative]
                        triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
        return torch.LongTensor(np.array(triplets)), len(triplets)



def HardestNegativeTripletSelector(margin=0, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)