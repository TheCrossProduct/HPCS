from typing import Union

import torch
from torch.nn import functional as F

from pytorch_metric_learning.losses import BaseMetricLossFunction, TripletMarginLoss
from pytorch_metric_learning.losses import CosFaceLoss

from hpcs.miner.triplet_margin_miner import RandomTripletMarginMiner
from hpcs.miner.triplet_margin_loss import TripletMarginLoss

from hpcs.distances import hyp_lca, CosineSimilarity


class HyperbolicHCLoss(BaseMetricLossFunction):
    def __init__(self, temperature: float = 0.05, t_per_anchor: int = 50, distance: str = 'cosine'):
        super(HyperbolicHCLoss, self).__init__()
        self.temperature = temperature
        self.t_per_anchor = t_per_anchor

        if distance == 'cosine':
            self.distance_sim = CosineSimilarity()
        else:
            raise KeyError(f'No implementation available for distance: {distance} ')
    def normalize_embeddings(self, embeddings, scale):
        """Normalize leaves embeddings to have the lie on a diameter."""
        min_scale = 1e-4
        max_scale = 50
        return F.normalize(embeddings, p=2, dim=1) * torch.clamp(scale, min_scale, max_scale)

    def get_triplets(self, n_samples):
        n_triplets = self.t_per_anchor * (n_samples * (n_samples -1 ) // 2)

        ij = torch.combinations(torch.arange(n_samples), r=2)
        ij = ij.repeat_interleave(self.t_per_anchor, dim=0)

        # sampling randomly the third element
        k = torch.randint(n_samples, (n_triplets, ), dtype=torch.long)
        # removing pts where i == k or j == k
        mask_i = ij[:, 0] != k
        mask_j = ij[:, 1] != k
        mask = torch.logical_and(mask_i, mask_j)

        return ij[mask,0], ij[mask, 1], k[mask]

    def compute_loss(self, embeddings, scale):
        hyp_indices_tuple = self.get_triplets(embeddings.shape[0])
        i, j, k = hyp_indices_tuple
        # move indices to correct device
        i = i.to(embeddings.device)
        j = j.to(embeddings.device)
        k = k.to(embeddings.device)

        mat_sim = self.distance_sim(embeddings)

        wij = mat_sim[i, j]
        wik = mat_sim[i, k]
        wjk = mat_sim[j, k]

        e1 = self.normalize_embeddings(embeddings[i], scale)
        e2 = self.normalize_embeddings(embeddings[j], scale)
        e3 = self.normalize_embeddings(embeddings[k], scale)

        dij = hyp_lca(e1, e2, return_coord=False).flatten()  # we flatten to avoid warnings
        dik = hyp_lca(e1, e3, return_coord=False).flatten()
        djk = hyp_lca(e2, e3, return_coord=False).flatten()

        # loss proposed by Chami et al.
        sim_triplet = torch.stack([wij, wik, wjk]).T
        lca_triplet = torch.stack([dij, dik, djk]).T
        weights = torch.softmax(lca_triplet / self.temperature, dim=-1)

        w_ord = torch.sum(sim_triplet * weights, dim=-1, keepdim=True)
        total = torch.sum(sim_triplet, dim=-1, keepdim=True) - w_ord

        loss_triplet_lca = torch.mean(total) + mat_sim.mean()
        return {
            "loss_lca": {
                "losses": loss_triplet_lca,
                "indices": hyp_indices_tuple,
                "reduction_type": "already_reduced",
            }
        }

class TripletHyperbolicLoss(BaseMetricLossFunction):
    def __init__(self, margin: float = 1.0, t_per_anchor: int = 50, fraction: float = 1.2, scale: Union[float, torch.tensor, torch.nn.Parameter] = 1e-3,
                 temperature: float = 0.05, anneal_factor: float = 0.5, normalize: bool = False, num_class: int = 4,
                 embedding: int = 4):
        super(TripletHyperbolicLoss, self).__init__()
        self.margin = margin
        self.t_per_anchor = t_per_anchor
        self.fraction = fraction
        self.scale = scale
        self.temperature = temperature
        self.anneal_factor = anneal_factor
        self.normalize = normalize
        self.num_class = num_class
        self.embedding = embedding

        self.distance_sim = CosineSimilarity()

        self.hyp_miner = RandomTripletMarginMiner(distance=self.distance_sim, margin=0, t_per_anchor=self.t_per_anchor, fraction=self.fraction, type_of_triplets='easy')
        self.triplet_miner = RandomTripletMarginMiner(distance=self.distance_sim, margin=self.margin, t_per_anchor=self.t_per_anchor, fraction=self.fraction, type_of_triplets='hard')

        self.loss_triplet_sim = TripletMarginLoss(distance=self.distance_sim, margin=self.margin)
        self.loss_cos = CosFaceLoss(num_classes=self.num_class, embedding_size=self.embedding, margin=0.35, scale=64)

    def anneal_temperature(self):
        min_scale = 0.2
        max_scale = 1
        self.temperature *= torch.clamp(self.anneal_factor, min_scale, max_scale)
        return self.temperature

    def normalize_embeddings(self, embeddings):
        """Normalize leaves embeddings to have the lie on a diameter."""
        min_scale = 1e-4
        max_scale = 50
        scale = self.scale.to(embeddings.device)
        return F.normalize(embeddings, p=2, dim=1) * torch.clamp(scale, min_scale, max_scale)

    def compute_loss(self, embeddings, labels):
        triplet_indices_tuple = self.triplet_miner(embeddings, labels)
        hyp_indices_tuple = self.hyp_miner(embeddings, labels)

        anchor_idx, positive_idx, negative_idx = hyp_indices_tuple

        mat_sim = self.distance_sim(embeddings)

        wij = mat_sim[anchor_idx, positive_idx]
        wik = mat_sim[anchor_idx, negative_idx]
        wjk = mat_sim[positive_idx, negative_idx]

        e1 = embeddings[anchor_idx]
        e2 = embeddings[positive_idx]
        e3 = embeddings[negative_idx]

        if self.normalize:
            e1 = self.normalize_embeddings(e1)
            e2 = self.normalize_embeddings(e2)
            e3 = self.normalize_embeddings(e3)

        dij = hyp_lca(e1, e2, return_coord=False)
        dik = hyp_lca(e1, e3, return_coord=False)
        djk = hyp_lca(e2, e3, return_coord=False)

        # loss proposed by Chami et al.
        sim_triplet = torch.stack([wij, wik, wjk]).T
        lca_triplet = torch.stack([dij, dik, djk]).T
        weights = torch.softmax(lca_triplet / self.temperature, dim=-1)

        w_ord = torch.sum(sim_triplet * weights, dim=-1, keepdim=True)
        total = torch.sum(sim_triplet, dim=-1, keepdim=True) - w_ord

        loss_triplet_lca = torch.mean(total) + mat_sim.mean()

        # loss_triplet_sim = self.loss_triplet_sim(embeddings, labels, triplet_indices_tuple)
        loss_cos = self.loss_cos(embeddings, labels.long())

        return {
            "loss_lca": {
                "losses": loss_triplet_lca,
                "indices": hyp_indices_tuple,
                "reduction_type": "already_reduced",
            },
            "loss_sim": {
                "losses": loss_cos,
                # "indices": triplet_indices_tuple,
                "reduction_type": "already_reduced",
            },
        }