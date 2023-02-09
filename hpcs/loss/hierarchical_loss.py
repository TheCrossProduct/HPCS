import torch
from torch.nn import functional as F

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