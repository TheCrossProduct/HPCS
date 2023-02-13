import torch
import numpy as np
import pytorch_lightning as pl
from torchmetrics import MeanMetric

from torchmetrics import ClasswiseWrapper
from torchmetrics.classification import MulticlassAccuracy,Accuracy

from torch.nn import functional as F
from torch.optim import lr_scheduler

from scipy.cluster.hierarchy import linkage
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

from hpcs.optim import RAdam
from hpcs.distances.poincare import project
from hpcs.distances.poincare import mobius_add
from hpcs.loss.ultrametric_loss import TripletHyperbolicLoss
from hpcs.utils.viz import plot_hyperbolic_eval
from hpcs.utils.scores import get_optimal_k


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

def _decode_linkage(leaves_embeddings):
    leaves_embeddings=(leaves_embeddings).detach().cpu()
    Z = linkage(leaves_embeddings, method='complete', metric='cosine')
    return Z


class SimilarityHypHC(pl.LightningModule):
    """
    Args:
        nn: torch.nn.Module
            model used to do feature extraction

        sim_distance: optional {'cosine', 'hyperbolic'}
            similarity distance to use to compute the miner loss function in the features' space

        temperature: float
            factor used in the HypHC loss

        margin: float
            margin value used in the miner loss

        init_rescale: float
            scale value used to rescale leaf embeddings in the Poincare's Disk

        max_scale: float
            max scale value to use to rescale leaf embeddings

        lr: float
            learning rate

        patience: int
            patience value for the scheduler

        factor: float
            learning rate reduction factor

        min_lr: float
            minimum value for learning rate
    """

    def __init__(self, nn: torch.nn.Module, model_name: str = 'vn_dgcnn_partseg', train_rotation: str = 'so3', test_rotation: str = 'so3',
                 dataset: str = 'shapenet', lr: float = 1e-3, embedding: int = 6, k: int = 10, margin: float = 1.0, t_per_anchor: int = 50,
                 fraction: float = 1.2, temperature: float = 0.05, anneal_factor: float = 0.5, anneal_step: int = 0, num_class: int = 4,
                 normalize: bool = False, class_vector: bool = False, normalize_classification: bool = False, trade_off: float = 0.1, hierarchical: bool = False):
        super(SimilarityHypHC, self).__init__()
        self.model = nn
        self.model_name = model_name
        self.train_rotation = train_rotation
        self.test_rotation = test_rotation
        self.dataset = dataset
        self.lr = lr
        self.embedding = embedding
        self.k = k
        self.margin = margin
        self.t_per_anchor = t_per_anchor
        self.fraction = fraction
        self.temperature = temperature
        self.anneal_factor = anneal_factor
        self.anneal_step = anneal_step
        self.num_class = num_class
        self.normalize = normalize
        self.class_vector = class_vector
        self.trade_off = trade_off
        self.hierarchical = hierarchical
        self.loss=torch.nn.CrossEntropyLoss()
        self.normalize_classification=normalize_classification
        self.multiclass_train=Accuracy(task="multiclass",num_classes=num_class)
        self.multiclass_val=Accuracy(task="multiclass",num_classes=num_class)
        self.save_hyperparameters()
        self.log("kgraph", k)
        self.log("dropout",self.model.dropout)
        self.log("normalize_classification",self.model.normalize_classification)

    def configure_optimizers(self):
        optim = RAdam(self.parameters(), lr=self.lr)
        scheduler = [
            {
                'scheduler': lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=4,
                                                            min_lr=1e-5, verbose=True),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': True,
            },
        ]

        return [optim], scheduler

    def training_step(self, batch, batch_idx):
        points, label, targets = batch
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        points, label, targets = points.float().to(device), label.long().to(device), targets.long().to(device)
        points = points.transpose(2, 1)
        predictions,features = self.model(points)
        loss=self.loss(predictions,targets)
        self.log("train_loss", loss)
        self.multiclass_train(predictions,targets)
        self.log('train_MulticlassAccuracy',self.multiclass_train,on_step=True,on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        points, label, targets = batch
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        points, label, targets = points.float().to(device), label.long().to(device), targets.long().to(device)
        points = points.transpose(2, 1)
        predictions,features = self.model(points)
        features= features.transpose(2, 1)
        loss=self.loss(predictions,targets)
        self.log("val_loss", loss)
        self.multiclass_val(predictions,targets)
        self.log('val_MulticlassAccuracy',self.multiclass_val,on_step=True,on_epoch=True)


        linkage_matrix = []
        for object_idx in range(points.size(0)):
            Z = _decode_linkage(features[object_idx])
            linkage_matrix.append(Z)
        indexes = []
        indexesk = []
        for object_idx in range(points.size(0)):
            best_pred, best_k, best_score = get_optimal_k(targets[object_idx].cpu(), linkage_matrix[object_idx], 'iou')
            indexes.append(best_score)
            indexesk.append(best_k)
        score = torch.mean(torch.tensor(indexes))
        meank = torch.mean(torch.tensor(indexesk).float())
        self.log("val_score", score)
        self.log("val_meanK", meank)
        return loss

    def test_step(self, batch, batch_idx):
        points, label, targets = batch
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        points, label, targets = points.float().to(device), label.long().to(device), targets.long().to(device)
        points = points.transpose(2, 1)
        predictions,features = self.model(points)
        features=features.transpose(2, 1)
        loss=self.loss(predictions,targets)
        self.log("test_loss", loss)

        linkage_matrix = []
        for object_idx in range(points.size(0)):
            Z = _decode_linkage(features[object_idx])
            linkage_matrix.append(Z)

        indexes = []
        indexesk = []

        for object_idx in range(points.size(0)):
            best_pred, best_k, best_score = get_optimal_k(targets[object_idx].cpu(), linkage_matrix[object_idx], 'iou')
            indexes.append(best_score)
            indexesk.append(best_k)
        score = torch.mean(torch.tensor(indexes))
        meank = torch.mean(torch.tensor(indexesk).float())
        self.log("test_score", score)
        self.log("meanK", meank)
        return loss

