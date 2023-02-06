import torch
from torch.nn import functional as F

from pytorch_metric_learning.losses import BaseMetricLossFunction, TripletMarginLoss

from hpcs.miner.triplet_margin_miner import RandomTripletMarginMiner
from hpcs.miner.triplet_margin_loss import TripletMarginLoss

from hpcs.distances import hyp_lca, CosineSimilarity


class TripletHyperbolicLoss(BaseMetricLossFunction):
    def __init__(self, margin: float = 1.0, t_per_anchor: int = 50, fraction: float = 1.2, scale: float = 1e-3,
                 temperature: float = 0.05, anneal_factor: float = 0.5, normalize: bool = False):
        super(TripletHyperbolicLoss, self).__init__()
        self.margin = margin
        self.t_per_anchor = t_per_anchor
        self.fraction = fraction
        self.scale = scale
        self.temperature = temperature
        self.anneal_factor = anneal_factor
        self.normalize = normalize

        self.distance_sim = CosineSimilarity()

        self.hyp_miner = RandomTripletMarginMiner(distance=self.distance_sim, margin=0, t_per_anchor=self.t_per_anchor, fraction=self.fraction, type_of_triplets='easy')
        self.triplet_miner = RandomTripletMarginMiner(distance=self.distance_sim, margin=self.margin, t_per_anchor=self.t_per_anchor, fraction=self.fraction, type_of_triplets='semihard')

        self.loss_triplet_sim = TripletMarginLoss(distance=self.distance_sim, margin=self.margin)


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

        loss_triplet_sim = self.loss_triplet_sim(embeddings, labels, triplet_indices_tuple)

        return {
            "loss_lca": {
                "losses": loss_triplet_lca,
                "indices": hyp_indices_tuple,
                "reduction_type": "already_reduced",
            },
            "loss_sim": {
                "losses": loss_triplet_sim,
                "indices": triplet_indices_tuple,
                "reduction_type": "already_reduced",
            },
        }