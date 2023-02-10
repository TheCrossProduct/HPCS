import torch
from torch.nn import functional as F

from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses import LargeMarginSoftmaxLoss


def hierarchical_loss(probabilities, targets, hierarchy_list):
    loss = 0
    for level_loss_list in hierarchy_list:
        probabilities_tosum = probabilities.clone()
        summed_probabilities = probabilities_tosum
        for branch in level_loss_list:
            branch_probs = torch.FloatTensor()
            branch_probs = branch_probs.to(probabilities_tosum.device)
            for channel in branch:
                branch_probs = torch.cat((branch_probs, probabilities_tosum[:, channel].unsqueeze(1)), 1)

            summed_tree_branch_slice = branch_probs.sum(1, keepdim=True)

            for channel in branch:
                summed_probabilities[:, channel:(channel + 1)] = summed_tree_branch_slice

        level_loss = F.nll_loss(torch.log(summed_probabilities), targets)
        loss = loss + level_loss
    return loss


class HierarchicalCosFaceLoss(LargeMarginSoftmaxLoss):
    """
    Implementation of https://arxiv.org/pdf/1801.07698.pdf
    """

    def __init__(self, *args, margin=0.35, scale=64, **kwargs):
        super().__init__(*args, margin=margin, scale=scale, **kwargs)

    def init_margin(self):
        pass

    def cast_types(self, dtype, device):
        self.W.data = c_f.to_device(self.W.data, device=device, dtype=dtype)

    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        if self.collect_stats:
            with torch.no_grad():
                self.get_angles(
                    cosine_of_target_classes
                )  # For the purpose of collecting stats
        return cosine_of_target_classes - self.margin

    def scale_logits(self, logits, *_):
        return logits * self.scale

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        indices_tuple = None
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        mask = self.get_target_mask(embeddings, labels)
        cosine = self.get_cosine(embeddings)
        cosine_of_target_classes = cosine[mask == 1]
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes
        )
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(
            1
        )
        logits = cosine + (mask * diff)
        logits = self.scale_logits(logits, embeddings)

        probabilities = F.softmax(logits, dim=1)
        loss_hier = hierarchical_loss(probabilities, labels, self.hierarchy_list)

        miner_weighted_loss = loss_hier * miner_weights
        loss_dict = {
            "loss": {
                "losses": miner_weighted_loss,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }
        self.add_weight_regularization_to_loss_dict(loss_dict, self.W.t())
        return loss_dict
