import torch
from pytorch_metric_learning.distances import DotProductSimilarity

class CosineSimilarity(DotProductSimilarity):
    def __init__(self, **kwargs):
        super().__init__(normalize_embeddings=True, **kwargs)
        assert self.is_inverted
        assert self.normalize_embeddings

    def compute_mat(self, query_emb, ref_emb):
        dist_mat = torch.matmul(query_emb, ref_emb.t())

        return 0.5 * (1 + dist_mat)

    def pairwise_distance(self, query_emb, ref_emb):
        return 0.5 * (1 + torch.sum(query_emb * ref_emb, dim=1))