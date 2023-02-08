import torch

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def remap_labels(y_true):
    y_remap = torch.zeros_like(y_true)
    for i, l in enumerate(torch.unique(y_true)):
        y_remap[y_true==l] = i
    return y_remap