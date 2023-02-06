from pytorch_metric_learning import miners

from hpcs.miner.loss_and_miner_utils import get_balanced_random_triplet_indices


class RandomTripletMarginMiner(miners.TripletMarginMiner):
    def __init__(self, t_per_anchor, fraction, margin=0.2, type_of_triplets="all", **kwargs):
        super().__init__(margin=margin, type_of_triplets=type_of_triplets, **kwargs)
        self.t_per_anchor = t_per_anchor
        self.fraction = fraction


    def mine(self, embeddings, labels, ref_emb, ref_labels):
        anchor_idx, positive_idx, negative_idx = get_balanced_random_triplet_indices(labels=labels, ref_labels=None,
                                                                                     fraction=self.fraction, t_per_anchor=self.t_per_anchor)
        mat = self.distance(embeddings, ref_emb)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]

        triplet_margin = (
            ap_dist - an_dist if self.distance.is_inverted else an_dist - ap_dist
        )

        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin

        else:
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= triplet_margin <= 0
            elif self.type_of_triplets == "semihard":
                threshold_condition &= triplet_margin > 0
        # print(f"Number of easy triplets : {threshold_condition.sum()}")
        return (
            anchor_idx[threshold_condition],
            positive_idx[threshold_condition],
            negative_idx[threshold_condition],
        )