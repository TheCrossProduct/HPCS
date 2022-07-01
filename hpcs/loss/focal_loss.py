import torch

class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=.25, eps=1e-7):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, y_pred, y_true):
        logits = y_pred.clamp(self.eps, 1. - self.eps)
        p_t = torch.where(y_true == 1, logits, 1 - logits)
        alpha_factor = self.alpha * torch.ones_like(y_true)
        alpha_t = torch.where(y_true == 1, alpha_factor, 1 - alpha_factor)
        cross_entropy = - torch.log(p_t)
        weight = alpha_t * torch.pow((1 - p_t), self.gamma)
        loss = weight * cross_entropy
        return loss.mean()