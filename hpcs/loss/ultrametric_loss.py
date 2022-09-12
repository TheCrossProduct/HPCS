import pytorch_lightning.utilities.enums
import torch
from torch.nn import functional as F
import numpy as np
from pytorch_metric_learning.losses import BaseMetricLossFunction, TripletMarginLoss
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from hpcs.distances.lca import hyp_lca
from hpcs.distances.poincare import project
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance


class TripletHyperbolicLoss(BaseMetricLossFunction):
    def __init__(self, sim_distance: str = 'cosine', margin: float = 1.0, scale: float = 1e-3,
                 max_scale: float = 1. - 1e-3, temperature: float = 0.05, anneal: float = 0.5):
        super(TripletHyperbolicLoss, self).__init__()

        if sim_distance == 'cosine':
            self.distance_sim = CosineSimilarity()
        elif sim_distance == 'euclidean':
            self.distance_sim = LpDistance()

        self.margin = margin
        self.scale = scale
        self.max_scale = max_scale
        self.temperature = temperature
        self.anneal = anneal

        self.loss_triplet_sim = TripletMarginLoss(distance=self.distance_sim, margin=self.margin, triplets_per_anchor=20000)

    def anneal(self):
        # TODO: review this function
        max_temp = 0.8
        min_temp = 0.01
        self.temperature = max(min(self.temperature * self.anneal, max_temp), min_temp)

    def normalize_embeddings(self, embeddings):
        """Normalize leaves embeddings to have the lie on a diameter."""
        min_scale = 1e-2
        max_scale = self.max_scale
        return F.normalize(embeddings, p=2, dim=1) * self.scale.clamp_min(min_scale).clamp_max(max_scale)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels, t_per_anchor=100):
        indices_tuple = lmu.convert_to_triplets(None, labels, t_per_anchor=t_per_anchor)

        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()

        e1 = embeddings[anchor_idx]
        e2 = embeddings[positive_idx]
        e3 = embeddings[negative_idx]
        e1 = self.normalize_embeddings(e1)
        e2 = self.normalize_embeddings(e2)
        e3 = self.normalize_embeddings(e3)
        dij = hyp_lca(e1, e2, return_coord=False)
        dik = hyp_lca(e1, e3, return_coord=False)
        djk = hyp_lca(e2, e3, return_coord=False)

        if isinstance(self.distance_sim, CosineSimilarity):
            mat_sim = 0.5 * (1 + self.distance_sim(embeddings))
            wij = mat_sim[anchor_idx, positive_idx]
            wik = mat_sim[anchor_idx, negative_idx]
            wjk = mat_sim[positive_idx, negative_idx]
        else:
            wij = torch.exp(-dij)
            wik = torch.exp(-dik)
            wjk = torch.exp(-djk)

        # loss proposed by Chami et al.
        sim_triplet = torch.stack([torch.exp(-dij), torch.exp(-dik), torch.exp(-djk)]).T    # [torch.exp(-dij), torch.exp(-dik), torch.exp(-djk)]
        lca_triplet = torch.stack([dij, dik, djk]).T
        weights = torch.softmax(lca_triplet / self.temperature, dim=-1)

        w_ord = torch.sum(sim_triplet * weights, dim=-1, keepdim=True)
        total = torch.sum(sim_triplet, dim=-1, keepdim=True) - w_ord

        loss_triplet_lca = torch.mean(total) + mat_sim.mean()

        loss_triplet_sim = self.loss_triplet_sim(embeddings, labels)

        return {
            "loss_lca": {
                "losses": loss_triplet_lca,
                "indices": (anchor_idx, positive_idx, negative_idx),
                "reduction_type": "already_reduced",
            },
            "loss_sim": {
                "losses": loss_triplet_sim,
                "indices": None,
                "reduction_type": "already_reduced",
            },
        }
