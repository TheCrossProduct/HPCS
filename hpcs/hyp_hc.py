import torch
import numpy as np
import pytorch_lightning as pl

from torch.nn import functional as F
from torch.optim import lr_scheduler

from scipy.cluster.hierarchy import linkage
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

from hpcs.optim import RAdam
from hpcs.distances.poincare import project
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
                 fraction: float = 1.2, temperature: float = 0.05, anneal_factor: float = 0.5, anneal_step: int = 0, num_class: int = 4):
        super(SimilarityHypHC, self).__init__()
        self.save_hyperparameters()
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
        self.scale = torch.nn.Parameter(torch.Tensor([1e-3]), requires_grad=True)

        self.triplet_loss = TripletHyperbolicLoss(margin=self.margin,
                                                  t_per_anchor=self.t_per_anchor,
                                                  fraction=self.fraction,
                                                  scale=self.scale,
                                                  temperature=self.temperature,
                                                  anneal_factor=self.anneal_factor)

    def _decode_linkage(self, leaves_embeddings):
        """Build linkage matrix from leaves' embeddings. Assume points are normalized to same radius."""
        leaves_embeddings = self.triplet_loss.normalize_embeddings(leaves_embeddings)
        leaves_embeddings = project(leaves_embeddings).detach().cpu()
        Z = linkage(leaves_embeddings, method='ward', metric='euclidean')
        return Z


    def forward(self, batch):
        if self.dataset == 'shapenet':
            points, label, targets = batch
        elif self.dataset == 'partnet':
            points, targets = batch
            label = torch.zeros(points.size(0), 1)

        trot = None
        rot = self.train_rotation
        if rot == 'z':
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True)
        elif rot == 'so3':
            trot = Rotate(R=random_rotations(points.shape[0]))
        if trot is not None:
            points = trot.transform_points(points.cpu())

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        points, label, targets = points.float().to(device), label.long().to(device), targets.long().to(device)
        points = points.transpose(2, 1)

        if self.dataset == 'shapenet':
            num_parts = self.num_class
            batch_class_vector = []
            for object in targets:
                parts = F.one_hot(remap_labels(torch.unique(object)), num_parts)
                class_vector = parts.sum(dim=0).float()
                batch_class_vector.append(class_vector)
            decode_vector = torch.stack(batch_class_vector)
        elif self.dataset == 'partnet':
            num_parts = self.num_class
            batch_class_vector = []
            for object in targets:
                parts = F.one_hot(torch.unique(object), num_parts)
                class_vector = parts.sum(dim=0).float()
                batch_class_vector.append(class_vector)
            decode_vector = torch.stack(batch_class_vector)

        x_embedding = self.model(points, decode_vector)
        x_poincare = project(self.scale * x_embedding)

        x_poincare_reshape = x_poincare.contiguous().view(-1, self.embedding)
        targets_reshape = targets.view(-1, 1)[:, 0]

        losses = self.triplet_loss.compute_loss(embeddings=x_poincare_reshape,
                                                labels=targets_reshape,
                                                )

        loss_triplet = losses['loss_sim']['losses']
        loss_hyphc = losses['loss_lca']['losses']

        return loss_triplet, loss_hyphc


    def _forward(self, batch):
        if self.dataset == 'shapenet':
            points, label, targets = batch
        elif self.dataset == 'partnet':
            points, targets = batch
            label = torch.zeros(points.size(0), 1)

        trot = None
        rot = self.test_rotation
        if rot == 'z':
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True)
        elif rot == 'so3':
            trot = Rotate(R=random_rotations(points.shape[0]))
        if trot is not None:
            points = trot.transform_points(points.cpu())

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        points, label, targets = points.float().to(device), label.long().to(device), targets.long().to(device)
        points = points.transpose(2, 1)

        if self.dataset == 'shapenet':
            num_parts = self.num_class
            batch_class_vector = []
            for object in targets:
                parts = F.one_hot(remap_labels(torch.unique(object)), num_parts)
                class_vector = parts.sum(dim=0).float()
                batch_class_vector.append(class_vector)
            decode_vector = torch.stack(batch_class_vector)
        elif self.dataset == 'partnet':
            num_parts = self.num_class
            batch_class_vector = []
            for object in targets:
                parts = F.one_hot(torch.unique(object), num_parts)
                class_vector = parts.sum(dim=0).float()
                batch_class_vector.append(class_vector)
            decode_vector = torch.stack(batch_class_vector)

        x_embedding = self.model(points, decode_vector)
        x_poincare = project(self.scale * x_embedding)

        x_poincare_reshape = x_poincare.contiguous().view(-1, self.embedding)
        targets_reshape = targets.view(-1, 1)[:, 0]

        losses = self.triplet_loss.compute_loss(embeddings=x_poincare_reshape,
                                                labels=targets_reshape,
                                                )

        loss_triplet = losses['loss_sim']['losses']
        loss_hyphc = losses['loss_lca']['losses']

        linkage_matrix = []
        for object_idx in range(points.size(0)):
            Z = self._decode_linkage(x_poincare[object_idx])
            linkage_matrix.append(Z)

        return loss_triplet, loss_hyphc, x_embedding, x_poincare, linkage_matrix, points, targets


    def configure_optimizers(self):
        optim = RAdam(self.parameters(), lr=self.lr)
        scheduler = [
            {
                'scheduler': lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=5,
                                                            min_lr=1e-6, verbose=True),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': True,
            },
        ]

        return [optim], scheduler

    def training_step(self, batch, batch_idx):
        loss_triplet, loss_hyphc = self.forward(batch)
        loss = loss_triplet + loss_hyphc

        self.log("train_loss", {"total_loss": loss, "triplet_loss": loss_triplet, "hyphc_loss": loss_hyphc})
        self.log("scale", self.scale)
        self.log("temperature", self.temperature)
        return {'loss': loss, 'progress_bar': {'triplet': loss_triplet, 'hyphc': loss_hyphc}}

    def training_epoch_end(self, outputs):
        if self.current_epoch and self.anneal_step > 0 and self.current_epoch % self.anneal_step == 0:
            print(f"Annealing temperature at the end of epoch {self.current_epoch}")
            self.temperature = self.triplet_loss.anneal_temperature()
            print("Temperature Value: ", self.temperature)

    def validation_step(self, batch, batch_idx):
        val_loss_triplet, val_loss_hyphc = self.forward(batch)
        val_loss = val_loss_triplet + val_loss_hyphc

        self.log("val_loss", val_loss)
        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        test_loss_triplet, test_loss_hyphc, x_embedding, x_poincare, linkage_matrix, points, targets = self._forward(batch)
        test_loss = test_loss_hyphc + test_loss_triplet

        indexes = []
        for object_idx in range(points.size(0)):
            best_pred, best_k, best_score = get_optimal_k(targets[object_idx].cpu(), linkage_matrix[object_idx], 'iou')
            # iou_score, ri_score = eval_clustering(targets[object_idx].cpu(), linkage_matrix[object_idx])

            plot_hyperbolic_eval(x=points[object_idx].T.cpu(),
                                 y=targets[object_idx].cpu(),
                                 y_pred=best_pred,
                                 emb_hidden=x_embedding[object_idx].cpu(),
                                 emb_poincare=self.triplet_loss.normalize_embeddings(x_poincare[object_idx]).cpu(),
                                 linkage_matrix=linkage_matrix[object_idx],
                                 k=best_k,
                                 score=best_score,
                                 show=True)

            indexes.append(best_score)
        score = torch.mean(torch.tensor(indexes))

        self.log("score", score)
        self.log("test_loss", test_loss)
        return {'test_loss': test_loss}
