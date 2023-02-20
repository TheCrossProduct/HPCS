from typing import Union

import torch
from torch.nn import functional as F

from pytorch_metric_learning.losses import BaseMetricLossFunction, TripletMarginLoss
from pytorch_metric_learning.losses import CosFaceLoss

from hpcs.loss.hierarchical_cosface_loss import HierarchicalCosFaceLoss
from hpcs.miner.triplet_margin_miner import RandomTripletMarginMiner
from hpcs.miner.triplet_margin_loss import TripletMarginLoss

from hpcs.distances import hyp_lca, CosineSimilarity


class MetricHyperbolicLoss(BaseMetricLossFunction):
    def __init__(self, margin: float = 1.0, t_per_anchor: int = 50, fraction: float = 1.2, scale: Union[float, torch.tensor, torch.nn.Parameter] = 1e-3,
                 temperature: float = 0.05, anneal_factor: float = 0.5, num_class: int = 4, embedding_size: int = 4,
                 cosface: bool = True, miner: bool = False):
        super(MetricHyperbolicLoss, self).__init__()
        self.margin = margin
        self.t_per_anchor = t_per_anchor
        self.fraction = fraction
        self.scale = scale
        self.temperature = temperature
        self.anneal_factor = anneal_factor
        self.num_class = num_class
        self.embedding_size = embedding_size
        self.cosface = cosface
        self.miner = miner
        self.distance_sim = CosineSimilarity()

        if self.miner:
            self.hyp_miner = RandomTripletMarginMiner(distance=self.distance_sim, margin=0, t_per_anchor=self.t_per_anchor, fraction=self.fraction, type_of_triplets='easy')

        if self.cosface:
            self.loss_cosface = CosFaceLoss(num_classes=self.num_class, embedding_size=self.embedding_size, margin=0.35, scale=2)
        else:
            self.triplet_miner = RandomTripletMarginMiner(distance=self.distance_sim, margin=self.margin, t_per_anchor=self.t_per_anchor, fraction=self.fraction, type_of_triplets='semihard')
            self.loss_triplet = TripletMarginLoss(distance=self.distance_sim, margin=self.margin)

    def get_triplets(self, n_samples):
        n_triplets = self.t_per_anchor * (n_samples * (n_samples - 1) // 2)

        ij = torch.combinations(torch.arange(n_samples), r=2)
        ij = ij.repeat_interleave(self.t_per_anchor, dim=0)

        # sampling randomly the third element
        k = torch.randint(n_samples, (n_triplets, ), dtype=torch.long)
        # removing pts where i == k or j == k
        mask_i = ij[:, 0] != k
        mask_j = ij[:, 1] != k
        mask = torch.logical_and(mask_i, mask_j)

        return ij[mask,0], ij[mask, 1], k[mask]

    def compute_hyp(self, x_poincare, labels):
        # print(f"SELF.MINER {self.miner}")
        if self.miner:
            hyp_indices_tuple = self.hyp_miner(x_poincare, labels)
        else:
            hyp_indices_tuple = self.get_triplets(x_poincare.shape[0])

        anchor_idx, positive_idx, negative_idx = hyp_indices_tuple
        mat_sim = self.distance_sim(x_poincare)

        wij = mat_sim[anchor_idx, positive_idx]
        wik = mat_sim[anchor_idx, negative_idx]
        wjk = mat_sim[positive_idx, negative_idx]

        e1 = x_poincare[anchor_idx]
        e2 = x_poincare[positive_idx]
        e3 = x_poincare[negative_idx]

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

        loss_hyperbolic = torch.mean(total) + mat_sim.mean()

        return loss_hyperbolic

    def get_logits(self, embeddings, labels):
        if hasattr(self, 'loss_cosface'):
            dtype, device = embeddings.dtype, embeddings.device
            self.loss_cosface.cast_types(dtype, device)
            mask = self.loss_cosface.get_target_mask(embeddings, labels)
            cosine = self.loss_cosface.get_cosine(embeddings)
            cosine_of_target_classes = cosine[mask == 1]
            modified_cosine_of_target_classes = self.loss_cosface.modify_cosine_of_target_classes(
                cosine_of_target_classes
            )
            diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(
                1
            )
            logits = cosine + (mask * diff)
            logits = self.loss_cosface.scale_logits(logits, embeddings)
            return logits
        else:
            raise ValueError("Cannot get logits since this class doesn't use any CosFaceLoss")
    def compute_loss(self, x_euclidean, x_poincare, labels, *args):

        loss_hyperbolic = self.compute_hyp(x_poincare, labels)

        if self.cosface:
            loss_metric = self.loss_cosface(x_poincare, labels.long())
        else:
            triplet_indices_tuple = self.triplet_miner(x_poincare, labels)
            loss_metric = self.loss_triplet_sim(x_poincare, labels, triplet_indices_tuple)

        return {
            "loss_hyp": {
                "losses": loss_hyperbolic
            },
            "loss_metric": {
                "losses": loss_metric
            },
        }

    def anneal_temperature(self):
        min_scale = 0.2
        max_scale = 1
        self.temperature *= torch.clamp(self.anneal_factor, min_scale, max_scale)
        return self.temperature

    def normalize_embeddings(self, embeddings):
        """Normalize leaves embeddings to have the lie on a diameter."""
        min_scale = 1e-4
        max_scale = 1
        return F.normalize(embeddings, p=2, dim=1) * torch.clamp(self.scale, min_scale, max_scale)


class HierarchicalMetricHyperbolicLoss(MetricHyperbolicLoss):
    def __init__(self, margin: float = 1.0, t_per_anchor: int = 50, fraction: float = 1.2, scale: Union[
        float, torch.tensor, torch.nn.Parameter] = 1e-3, temperature: float = 0.05, anneal_factor: float = 0.5,
                 num_class: int = 4, embedding_size: int = 4, miner: bool = False, hierarchy_list: list = []):
        super(HierarchicalMetricHyperbolicLoss, self).__init__(margin=margin,
                                                               t_per_anchor=t_per_anchor,
                                                               fraction=fraction,
                                                               scale=scale,
                                                               temperature=temperature,
                                                               anneal_factor=anneal_factor,
                                                               num_class=num_class,
                                                               embedding_size=embedding_size,
                                                               cosface=True,
                                                               miner=miner)
        self.hierarchy_list = hierarchy_list
        self.loss_cosface = HierarchicalCosFaceLoss(num_classes=self.num_class, embedding_size=self.euclidean_size,
                                                    margin=0.35, scale=64, hierarchy_list=self.hierarchy_list)

    def compute_loss(self, x_euclidean, x_poincare, labels, *args):
        loss_hyperbolic = self.compute_hyp(x_poincare, labels)

        loss_metric = self.loss_cosface(x_euclidean, labels.long())

        return {
            "loss_hyp": {
                "losses": loss_hyperbolic
            },
            "loss_metric": {
                "losses": loss_metric
            },
        }
