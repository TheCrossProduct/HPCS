import torch
from torch.nn import functional as F
import numpy as np
from pytorch_metric_learning.losses import BaseMetricLossFunction, TripletMarginLoss
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from hpcs.distances.poincare import HyperbolicDistance, HyperbolicLCA
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from typing import Union

class TripletHyperbolicLoss(BaseMetricLossFunction):
    def __init__(self, sim_distance: str = 'cosine', margin: float = 1.0, init_rescale: float = 1e-3,
                 min_scale: float = 1e-2, max_scale: float = 1. - 1e-3,
                 temperature: float = 0.05, anneal: float = 0.5):
        super(TripletHyperbolicLoss, self).__init__()
        # Triplet Loss on similarities between embeddings
        if sim_distance == 'cosine':
            self.distace_sim = CosineSimilarity()
            # self.loss_triplet_sim = losses.TripletMarginLoss(distance=self.distace_sim, margin=self.margin)
        elif sim_distance == 'hyperbolic':
            self.distace_sim = HyperbolicDistance()
            # self.loss_triplet_sim = losses.TripletMarginLoss(distance=self.distace_sim, margin=self.margin)
        elif sim_distance == 'euclidean':
            self.distace_sim = LpDistance()
            # self.loss_triplet_sim = losses.TripletMarginLoss(distance=self.distace_sim, margin=self.margin,
            #                                                  embedding_regularizer=regularizers.LpRegularizer())
        else:
            raise ValueError(
                f"The option {sim_distance} is not available for sim_distance. The only available are ['cosine', 'euclidean', 'hyperbolic'].")

        self.margin = margin

        self.rescale = torch.nn.Parameter(torch.Tensor([init_rescale]), requires_grad=True)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.temperature = temperature
        self.anneal = anneal

        # distances to compute losses
        self.loss_triplet_sim = TripletMarginLoss(distance=self.distace_sim, margin=self.margin)
        # loss between embeddings in
        self.distance_lca = HyperbolicLCA()

    def anneal(self):
        # TODO: review this function
        max_temp = 0.8
        min_temp = 0.01
        self.temperature = max(min(self.temperature * self.anneal, max_temp), min_temp)
    def _rescale_emb(self, embeddings):
        """Normalize leaves embeddings to have the lie on a diameter."""
        min_scale = self.min_scale # self.init_size
        max_scale = self.max_scale
        return F.normalize(embeddings, p=2, dim=1) * self.rescale.clamp_min(min_scale).clamp_max(max_scale)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels, t_per_anchor = 100):
        indices_tuple = lmu.convert_to_triplets(None, labels, t_per_anchor=t_per_anchor)

        points = torch.arange(0, len(labels))
        duplets = torch.combinations(points, r=2)
        random = torch.randint(0, 150, (len(duplets), 1))
        triplets = torch.cat((duplets, random), dim=-1)
        for i, x in enumerate(triplets):
            if len(torch.unique(x)) != 3:
                triplets = triplets[torch.arange(triplets.size(0)) != i]
        triplet = torch.unbind(triplets, dim=1)

        anchor_idx, positive_idx, negative_idx = triplet
        if len(anchor_idx) == 0:
            return self.zero_losses()

        if isinstance(self.distace_sim, CosineSimilarity):
            mat_sim = 0.5 * (1 + self.distace_sim(embeddings))
        else:
            # mat_sim = 0.5 * (1 + self.distace_sim(project(x_emb_samples)))
            mat_sim = torch.exp(-self.distace_sim(embeddings))

        mat_lca = self.distance_lca(self._rescale_emb(embeddings))
        wij = mat_sim[anchor_idx, positive_idx]
        wik = mat_sim[anchor_idx, negative_idx]
        wjk = mat_sim[positive_idx, negative_idx]

        dij = mat_lca[anchor_idx, positive_idx]
        dik = mat_lca[anchor_idx, negative_idx]
        djk = mat_lca[positive_idx, negative_idx]

        # loss proposed by Chami et al.
        sim_triplet = torch.stack([wij, wik, wjk]).T
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


def make_pairs(labels: torch.Tensor, idx: torch.Tensor):
    """
    Create all triplets from the provided labeld set. This function returns:

    :param labels: provided labels
    :param idx: vertex indices of the provided labels
    :return: a tuple ``pairs=(vertices1, vertices2)`` indexing all the pairs of elements in ``ìdx``
    """
    pairs = labels[None] == labels[:, None]
    src, dst = torch.triu_indices(pairs.size(0), pairs.size(1), 1)
    labels = pairs[src, dst].type(torch.float)
    pairs = idx[src], idx[dst]

    return pairs, labels


def make_triplets(labels, idx):
    """
    Create all triplets from the provided labeld set. This function creates:

        - a tuple ``pairs=(vertices1, vertices2)`` describing all the pairs of elements in ``ìdx``
        - a tuple ``(ref, neg)`` decribing couples of pairs of the form ``(i,j)`` and ``(k,j)`` such that ``i`` and
        ``j`` have the same label and ``k`` and ``j`` have different labels.

    :param labels: provided labels
    :param idx: vertex indices of the provided labels
    :return: a tuple ``(pairs, (ref, neg))``
    """

    from itertools import combinations

    def list_all_triplets():
        # labels = np.array(labels)

        triplets = []
        for label in set(labels):

            # positive indices
            positive = label == labels
            positive_indices = positive.nonzero()

            if len(positive_indices) < 2:
                continue

            # negative indices
            negative = torch.logical_not(positive)
            negative_indices = negative.nonzero()

            # All positive pairs
            pairs = list(combinations(positive_indices, 2))

            # Add all negatives for all positive pairs
            for anchor, positive in pairs:
                for negative in negative_indices:
                    triplets += [(anchor, positive, negative)]

        # return np.array(triplets)
        return torch.tensor(triplets, dtype=torch.long)

    def get_triplets():

        # triplets
        triplets = list_all_triplets()
        anchor = triplets[:, 0]
        positive = triplets[:, 1]
        negative = triplets[:, 2]

        # linearized indices
        n = labels.numel()
        idx_table = torch.zeros((n, n), dtype=torch.long)
        row, col = torch.triu_indices(n, n, 1)
        idx_table[row, col] = torch.arange(n * (n - 1) // 2)
        idx_table += idx_table.T
        positive_pairs = idx_table[anchor, positive]
        negative_pairs = idx_table[anchor, negative]

        return positive_pairs, negative_pairs

    pairs, _ = make_pairs(labels, idx)
    pos, neg = get_triplets()

    return pairs, (pos, neg)


# def loss_cluster_size(graph, edge_weights, ultrametric, hierarchy, top_nodes=0, dtype=torch.float64):
#     """
#     Cluster size regularization:
#
#      .. math::
#
#         loss = \\frac{1}{|E|}\sum_{e_{xy}\in E}\\frac{ultrametric(e_{xy})}{\min\{|c|\, | \, c\in Children(lca(x,y))\}}
#
#     :param graph: input graph (``higra.UndirectedGraph``)
#     :param edge_weights: edge weights of the input graph (``torch.Tensor``, autograd is supported)
#     :param ultrametric; ultrametric on the input graph  (``torch.Tensor``, autograd is supported)
#     :param hierarchy: optional,  if provided must be a tuple ``(tree, altitudes)`` corresponding to the result of ``higra.bpt_canonical`` on the input edge weighted graph
#     :param top_nodes: if different from 0, only the top ``top_nodes`` of the hiearchy are taken into account in the cluster size regularization
#     :return: loss value as a pytorch scalar
#
#     """
#     tree, altitudes = hierarchy
#     lca_map = hg.attribute_lca_map(tree)
#
#     if top_nodes <= 0:
#         top_nodes = tree.num_vertices()
#     top_nodes = max(tree.num_vertices() - top_nodes, tree.num_leaves())
#     top_edges, = np.nonzero(lca_map >= top_nodes)
#
#     area = hg.attribute_area(tree)
#     min_area = hg.accumulate_parallel(tree, area, hg.Accumulators.min)
#     min_area = min_area[lca_map[top_edges]]
#     min_area = torch.from_numpy(min_area)
#     min_area.type(dtype)
#
#     cluster_size_loss = ultrametric[top_edges] / min_area
#
#     return cluster_size_loss.max()


# def loss_triplet(graph, edge_weights, ultrametric, hierarchy, triplets, margin):
#     """
#     Triplet loss regularization with triplet :math:`\mathcal{T}`:
#
#      .. math::
#
#         loss = \sum_{(ref, pos, neg)\in \mathcal{T}} \max(0, ultrametric(ref, pos) - ultrametric(ref, neg) + margin)
#
#     :param graph: input graph (``higra.UndirectedGraph``)
#     :param edge_weights: edge weights of the input graph (``torch.Tensor``, autograd is supported)
#     :param ultrametric; ultrametric on the input graph  (``torch.Tensor``, autograd is supported)
#     :param hierarchy: optional,  if provided must be a tuple ``(tree, altitudes)`` corresponding to the result of ``higra.bpt_canonical`` on the input edge weighted graph
#     :param triplets:
#     :param margin:
#     :return: loss value as a pytorch scalar
#     """
#     tree, altitudes = hierarchy
#     lcaf = hg.make_lca_fast(tree)
#
#     pairs, (pos, neg) = triplets
#     pairs_distances = altitudes[lcaf.lca(*pairs)]  # ultrametric[mst_map[lcaf.lca(*pairs) - tree.num_leaves()]]
#
#     triplet_loss = torch.relu(pairs_distances[pos] - pairs_distances[neg] + margin)
#
#     return triplet_loss.mean()


# def loss_siamese_triplet(hierarchy, triplets, margin):
#     """
#     Triplet loss regularization with triplet :math:`\mathcal{T}`:
#
#      .. math::
#
#         loss = \sum_{(ref, pos, neg)\in \mathcal{T}} \max(0, ultrametric(ref, pos) - ultrametric(ref, neg) + margin)
#
#     :param graph: input graph (``higra.UndirectedGraph``)
#     :param edge_weights: edge weights of the input graph (``torch.Tensor``, autograd is supported)
#     :param ultrametric; ultrametric on the input graph  (``torch.Tensor``, autograd is supported)
#     :param hierarchy: optional,  if provided must be a tuple ``(tree, altitudes)`` corresponding to the result of ``higra.bpt_canonical`` on the input edge weighted graph
#     :param triplets:
#     :param margin:
#     :return: loss value as a pytorch scalar
#     """
#     tree, altitudes = hierarchy
#     lcaf = hg.make_lca_fast(tree)
#
#     pairs, (pos, neg) = triplets
#     pairs_distances = altitudes[lcaf.lca(*pairs)]  # ultrametric[mst_map[lcaf.lca(*pairs) - tree.num_leaves()]]
#
#     triplet_loss = torch.relu(pairs_distances[pos] - pairs_distances[neg] + margin)
#
#     return triplet_loss.mean()


def loss_closest(graph, edge_weights, ultrametric, hierarchy):
    """
    Mean square error between ``edge_weights`` and ``ultrametric`` on the input graph :math:`G=(V,E)`:

    .. math::

        loss = \\frac{1}{|E|}\sum_{e\in E}(edge_weights(e) - ultrametric(e))^2

    :param graph: input graph (``higra.UndirectedGraph``)
    :param edge_weights: edge weights of the input graph (``torch.Tensor``, autograd is supported)
    :param ultrametric; ultrametric on the input graph  (``torch.Tensor``, autograd is supported)
    :param hierarchy: optional,  if provided must be a tuple ``(tree, altitudes)`` corresponding to the result of ``higra.bpt_canonical`` on the input edge weighted graph
    :return: loss value as a pytorch scalar
    """
    errors = (ultrametric - edge_weights) ** 2
    return torch.mean(errors)


# def loss_cluster_and_triplet(graph, edge_weights, ultrametric, hierarchy, top_nodes, triplets, margin, gamma=1):
#     """
#     Mean square error between ``edge_weights`` and ``ultrametric`` on the input graph :math:`G=(V,E)` plus triplet loss regularization with triplet :math:`\mathcal{T}`:
#
#      .. math::
#
#         loss = \\frac{1}{|E|}\sum_{e_{xy}\in E}(edge_weights(e_{xy}) - ultrametric(e_{xy}))^2 + \gamma \sum_{(ref, pos, neg)\in \mathcal{T}} \max(0, ultrametric(ref, pos) - ultrametric(ref, neg) + margin)
#
#     :param graph: input graph (``higra.UndirectedGraph``)
#     :param edge_weights: edge weights of the input graph (``torch.Tensor``, autograd is supported)
#     :param ultrametric; ultrametric on the input graph  (``torch.Tensor``, autograd is supported)
#     :param hierarchy: optional,  if provided must be a tuple ``(tree, altitudes)`` corresponding to the result of ``higra.bpt_canonical`` on the input edge weighted graph
#     :param triplets:
#     :param margin:
#     :param gamma: weighting of the cluster size regularization (float)
#     :return: loss value as a pytorch scalar
#     """
#     closest_loss = loss_cluster_size(graph, edge_weights, ultrametric, hierarchy, top_nodes)
#     triplet_loss = loss_triplet(graph, edge_weights, ultrametric, hierarchy, triplets, margin)
#
#     return closest_loss + gamma * triplet_loss


# def loss_closest_and_cluster_size(graph, edge_weights, ultrametric, hierarchy, gamma=1, top_nodes=0, dtype=torch.float64):
#     """
#     Mean square error between ``edge_weights`` and ``ultrametric`` on the input graph :math:`G=(V,E)` plus cluster size regularization:
#
#     :param graph: input graph (``higra.UndirectedGraph``)
#     :param edge_weights: edge weights of the input graph (``torch.Tensor``, autograd is supported)
#     :param ultrametric; ultrametric on the input graph  (``torch.Tensor``, autograd is supported)
#     :param hierarchy: optional,  if provided must be a tuple ``(tree, altitudes)`` corresponding to the result of ``higra.bpt_canonical`` on the input edge weighted graph
#     :param gamma: weighting of the cluster size regularization (float)
#     :param top_nodes: if different from 0, only the top ``top_nodes`` of the hiearchy are taken into account in the cluster size regularization
#     :return: loss value as a pytorch scalar
#
#     """
#
#     closest_loss = loss_closest(graph, edge_weights, ultrametric, hierarchy)
#     cluster_size_loss = loss_cluster_size(graph, edge_weights, ultrametric, hierarchy, top_nodes, dtype)
#
#     return closest_loss + gamma * cluster_size_loss


# def loss_closest_and_triplet(graph, edge_weights, ultrametric, hierarchy, triplets, margin, gamma=1):
#     """
#     Mean square error between ``edge_weights`` and ``ultrametric`` on the input graph :math:`G=(V,E)` plus triplet loss regularization with triplet :math:`\mathcal{T}`:
#
#      .. math::
#
#         loss = \\frac{1}{|E|}\sum_{e_{xy}\in E}(edge_weights(e_{xy}) - ultrametric(e_{xy}))^2 + \gamma \sum_{(ref, pos, neg)\in \mathcal{T}} \max(0, ultrametric(ref, pos) - ultrametric(ref, neg) + margin)
#
#     :param graph: input graph (``higra.UndirectedGraph``)
#     :param edge_weights: edge weights of the input graph (``torch.Tensor``, autograd is supported)
#     :param ultrametric; ultrametric on the input graph  (``torch.Tensor``, autograd is supported)
#     :param hierarchy: optional,  if provided must be a tuple ``(tree, altitudes)`` corresponding to the result of ``higra.bpt_canonical`` on the input edge weighted graph
#     :param triplets:
#     :param margin:
#     :param gamma: weighting of the cluster size regularization (float)
#     :return: loss value as a pytorch scalar
#     """
#     closest_loss = loss_closest(graph, edge_weights, ultrametric, hierarchy)
#     triplet_loss = loss_triplet(graph, edge_weights, ultrametric, hierarchy, triplets, margin)
#
#     return closest_loss + gamma * triplet_loss


# def loss_closest_cluster_and_triplet(graph, edge_weights, ultrametric, hierarchy, triplets, margin, gamma=1):
#     """
#     Mean square error between ``edge_weights`` and ``ultrametric`` on the input graph :math:`G=(V,E)` plus triplet loss regularization with triplet :math:`\mathcal{T}`:
#
#      .. math::
#
#         loss = \\frac{1}{|E|}\sum_{e_{xy}\in E}(edge_weights(e_{xy}) - ultrametric(e_{xy}))^2 + \gamma \sum_{(ref, pos, neg)\in \mathcal{T}} \max(0, ultrametric(ref, pos) - ultrametric(ref, neg) + margin)
#
#     :param graph: input graph (``higra.UndirectedGraph``)
#     :param edge_weights: edge weights of the input graph (``torch.Tensor``, autograd is supported)
#     :param ultrametric; ultrametric on the input graph  (``torch.Tensor``, autograd is supported)
#     :param hierarchy: optional,  if provided must be a tuple ``(tree, altitudes)`` corresponding to the result of ``higra.bpt_canonical`` on the input edge weighted graph
#     :param triplets:
#     :param margin:
#     :param gamma: weighting of the cluster size regularization (float)
#     :return: loss value as a pytorch scalar
#     """
#     closest_loss = loss_closest(graph, edge_weights, ultrametric, hierarchy)
#     cluster_loss = loss_cluster_size(graph, edge_weights, ultrametric, hierarchy)
#     triplet_loss = loss_triplet(graph, edge_weights, ultrametric, hierarchy, triplets, margin)
#
#     return closest_loss + gamma * triplet_loss + gamma * cluster_loss
