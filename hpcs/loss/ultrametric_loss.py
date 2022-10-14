import pytorch_lightning
import torch
from torch.nn import functional as F
import numpy as np
from pytorch_metric_learning.losses import BaseMetricLossFunction, TripletMarginLoss
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning import miners, losses
from hpcs.miners.triplet_margin_miner import RandomTripletMarginMiner
from hpcs.miners.batch_easy_miner import BatchEasyMiner
from hpcs.miners.batch_hard_miner import BatchHardMiner
from hpcs.miners.triplet_margin_loss import TripletMarginLoss

from hpcs.distances.lca import hyp_lca


class TripletHyperbolicLoss(BaseMetricLossFunction):
    def __init__(self, sim_distance: str = 'cosine', margin: float = 1.0, scale: float = 1e-3,
                 max_scale: float = 1. - 1e-3, temperature: float = 0.05, anneal: float = 0.5):
        super(TripletHyperbolicLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.max_scale = max_scale
        self.temperature = temperature
        self.anneal = anneal

        if sim_distance == 'cosine':
            self.distance_sim = CosineSimilarity()
        elif sim_distance == 'euclidean':
            self.distance_sim = LpDistance()

        self.hyp_miner = RandomTripletMarginMiner(distance=CosineSimilarity(), margin=0, t_per_anchor=500, type_of_triplets='easy')
        self.triplet_miner = RandomTripletMarginMiner(distance=CosineSimilarity(), margin=0.2, t_per_anchor=1000, type_of_triplets='all')

        self.loss_triplet_sim = TripletMarginLoss(distance=CosineSimilarity(), margin=0.2)

    def anneal_temperature(self):
        min_scale = 0.2
        max_scale = 0.8
        self.temperature *= self.anneal
        return self.temperature

    def normalize_embeddings(self, embeddings):
        """Normalize leaves embeddings to have the lie on a diameter."""
        min_scale = 1e-2
        max_scale = self.max_scale
        return F.normalize(embeddings, p=2, dim=1) * torch.clamp(self.scale, min_scale, max_scale)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels, t_per_anchor):
        dij_distances = []
        dik_distances = []
        djk_distances = []
        wij_similarities = []
        wik_similarities = []
        wjk_similarities = []
        loss_triplets = []
        embedding = iter(embeddings)
        label = iter(labels)
        for i in range(embeddings.size(0)):
            embeddings, labels = next(embedding), next(label)
            triplet_indices_tuple = self.triplet_miner(embeddings, labels)
            hyp_indices_tuple = self.hyp_miner(embeddings, labels)

            anchor_idx, positive_idx, negative_idx = hyp_indices_tuple
            # if len(anchor_idx) == 0:
            #     anchor_idx, positive_idx, negative_idx = lmu.convert_to_triplets(None, labels, t_per_anchor=500)

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

            dij_distances.append(dij)
            dik_distances.append(dik)
            djk_distances.append(djk)
            wij_similarities.append(wij)
            wik_similarities.append(wik)
            wjk_similarities.append(wjk)

            loss_triplet = self.loss_triplet_sim(embeddings, labels, triplet_indices_tuple)
            loss_triplets.append(loss_triplet)

        dij_distances = torch.cat(dij_distances)
        dik_distances = torch.cat(dik_distances)
        djk_distances = torch.cat(djk_distances)
        wij_similarities = torch.cat(wij_similarities)
        wik_similarities = torch.cat(wik_similarities)
        wjk_similarities = torch.cat(wjk_similarities)
        loss_triplets = torch.stack(loss_triplets)

        # loss proposed by Chami et al.
        sim_triplet = torch.stack([wij_similarities, wik_similarities, wjk_similarities]).T
        lca_triplet = torch.stack([dij_distances, dik_distances, djk_distances]).T
        weights = torch.softmax(lca_triplet / self.temperature, dim=-1)

        w_ord = torch.sum(sim_triplet * weights, dim=-1, keepdim=True)
        total = torch.sum(sim_triplet, dim=-1, keepdim=True) - w_ord

        loss_triplet_lca = torch.mean(total) + mat_sim.mean()

        loss_triplet_sim = torch.mean(loss_triplets)

        return {
            "loss_lca": {
                "losses": loss_triplet_lca,
            },
            "loss_sim": {
                "losses": loss_triplet_sim,
            },
        }