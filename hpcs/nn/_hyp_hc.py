import scipy.spatial.distance
import wandb
import torch
import numpy as np
import pytorch_lightning as pl
from typing import Optional

from scipy.cluster.hierarchy import fcluster, linkage

from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch_geometric.data import Batch
# from pytorch_lightning.metrics.functional import accuracy, iou
# from torchmetrics.functional import accuracy
from sklearn.metrics.cluster import adjusted_rand_score as ri

from hpcs.utils.viz import plot_hyperbolic_eval, plot_cloud, plot_clustering
from hpcs.utils.scores import eval_clustering, get_optimal_k
from hpcs.loss.ultrametric_loss import TripletHyperbolicLoss
from pytorch_metric_learning.distances import CosineSimilarity
from hpcs.miners.triplet_margin_miner import RandomTripletMarginMiner
from hpcs.distances.poincare import project
from hpcs.distances.lca import hyp_lca

from hpcs.optim import RAdam

from hpcs.utils import provider
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


class SimilarityHypHC(pl.LightningModule):
    """
    Args:
        nn: torch.nn.Module
            model used to do feature extraction

        sim_distance: optional {'cosine', 'hyperbolic'}
            similarity distance to use to compute the triplet loss function in the features' space

        temperature: float
            factor used in the HypHC loss

        margin: float
            margin value used in the triplet loss

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

    def __init__(self, nn: torch.nn.Module,
                 sim_distance: str = 'cosine', temperature: float = 0.05, anneal: float = 0.5, anneal_step: int = 0,
                 embedding: int = 6, margin: float = 1.0, init_rescale: float = 1e-2, max_scale: float = 1. - 1e-3, lr: float = 1e-3,
                 patience: int = 10, factor: float = 0.5, min_lr: float = 1e-4):
        super(SimilarityHypHC, self).__init__()
        self.save_hyperparameters()
        self.model = nn
        self.sim_distance = sim_distance
        self.temperature = temperature
        self.anneal = anneal
        self.anneal_step = anneal_step
        self.margin = margin
        self.max_scale = max_scale
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.embedding = embedding

        # self.temperature = torch.nn.Parameter(torch.Tensor([temperature]), requires_grad=True)
        self.scale = torch.nn.Parameter(torch.Tensor([init_rescale]), requires_grad=True)

        self.triplet_loss = TripletHyperbolicLoss(sim_distance=sim_distance,
                                                  margin=self.margin,
                                                  scale=self.scale,
                                                  max_scale=max_scale,
                                                  temperature=self.temperature,
                                                  anneal=anneal)

    def _decode_linkage(self, leaves_embeddings):
        """Build linkage matrix from leaves' embeddings. Assume points are normalized to same radius."""
        # leaves_embeddings = self.triplet_loss.normalize_embeddings(leaves_embeddings)
        # leaves_embeddings = project(leaves_embeddings).detach().cpu()
        # Z = linkage(leaves_embeddings, method='ward', metric='euclidean')

        leaves_embeddings = self.triplet_loss.normalize_embeddings(leaves_embeddings)
        sim_fn = lambda x, y: np.arccos(np.clip(np.sum(x * y, axis=-1), -1.0, 1.0))
        embeddings = F.normalize(leaves_embeddings, p=2, dim=1).detach().cpu()
        Z = linkage(embeddings, method='average', metric=sim_fn)

        return Z


    def forward(self, batch):
        dataset = 'partnet'
        if dataset == 'shapenet':
            points, label, targets = batch
        elif dataset == 'partnet':
            points, targets = batch
            label = torch.zeros(points.size(0), 1)

        trot = None
        rot = 'so3'
        if rot == 'z':
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True)
        elif rot == 'so3':
            trot = Rotate(R=random_rotations(points.shape[0]))
        if trot is not None:
            points = trot.transform_points(points.cpu())

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        points, label, targets = points.float().to(device), label.long().to(device), targets.long().to(device)
        points = points.transpose(2, 1)

        num_classes = 16
        x_emb = self.model(points, to_categorical(label, num_classes))
        x_poincare = project(self.scale * x_emb)

        x_poincare_reshape = x_poincare.contiguous().view(-1, self.embedding)
        targets_reshape = targets.view(-1, 1)[:, 0]

        # x_poincare = project(self.scale * points)
        # x_emb = self.model(x_poincare, to_categorical(label, num_classes))

        losses = self.triplet_loss.compute_loss(embeddings=x_poincare_reshape,
                                                labels=targets_reshape,
                                                indices_tuple=None,
                                                ref_emb=None,
                                                ref_labels=None,
                                                t_per_anchor=1000)

        loss_triplet = losses['loss_sim']['losses']
        loss_hyphc = losses['loss_lca']['losses']

        return loss_triplet, loss_hyphc


    def _forward(self, batch):
        dataset = 'partnet'
        if dataset == 'shapenet':
            points, label, targets = batch
        elif dataset == 'partnet':
            points, targets = batch
            label = torch.zeros(points.size(0), 1)

        trot = None
        rot = 'so3'
        if rot == 'z':
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True)
        elif rot == 'so3':
            trot = Rotate(R=random_rotations(points.shape[0]))
        if trot is not None:
            points = trot.transform_points(points.cpu())

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        points, label, targets = points.float().to(device), label.long().to(device), targets.long().to(device)
        points = points.transpose(2, 1)

        num_classes = 16
        x_emb = self.model(points, to_categorical(label, num_classes))
        x_poincare = project(self.scale * x_emb)

        x_poincare_reshape = x_poincare.contiguous().view(-1, self.embedding)
        targets_reshape = targets.view(-1, 1)[:, 0]

        # x_poincare = project(self.scale * points)
        # x_emb = self.model(x_poincare, to_categorical(label, num_classes))

        losses = self.triplet_loss.compute_loss(embeddings=x_poincare_reshape,
                                                labels=targets_reshape,
                                                indices_tuple=None,
                                                ref_emb=None,
                                                ref_labels=None,
                                                t_per_anchor=1000)

        loss_triplet = losses['loss_sim']['losses']
        loss_hyphc = losses['loss_lca']['losses']

        linkage_matrix = []
        for object_idx in range(points.size(0)):
            Z = self._decode_linkage(x_poincare[object_idx])
            linkage_matrix.append(Z)

        return x_emb, x_poincare, loss_triplet, loss_hyphc, linkage_matrix, points, targets


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

    def test_step(self, batch, batch_idx, triplet_heat_map=False):
        x_emb, x_poincare, test_loss_triplet, test_loss_hyphc, linkage_matrix, points, targets = self._forward(batch)
        test_loss = test_loss_hyphc + test_loss_triplet

        rand_indexes = []
        for object_idx in range(points.size(0)):
            y_pred_k, k, best_ri = get_optimal_k(targets[object_idx].cpu(), linkage_matrix[object_idx])
            # pu_score, nmi_score, ri_score = eval_clustering(y_true=targets.detach().cpu(), Z=linkage_matrix[0])

            plot_hyperbolic_eval(x=points[object_idx].T.cpu(),
                                 y=targets[object_idx].cpu(),
                                 y_pred=y_pred_k,
                                 emb_hidden=x_emb[object_idx].cpu(),
                                 emb_poincare=self.triplet_loss.normalize_embeddings(x_poincare[object_idx]).cpu(),
                                 linkage_matrix=linkage_matrix[object_idx],
                                 k=k,
                                 show=True)

            rand_indexes.append(best_ri)
        score = torch.mean(torch.tensor(rand_indexes))


        if triplet_heat_map:
            easy_miner = RandomTripletMarginMiner(distance=CosineSimilarity(), margin=0, t_per_anchor=10000, type_of_triplets='easy')
            hard_miner = RandomTripletMarginMiner(distance=CosineSimilarity(), margin=0, t_per_anchor=10000, type_of_triplets='hard')
            easy_indices_tuple = easy_miner(x_poincare, targets)
            hard_indices_tuple = hard_miner(x_poincare, targets)
            anchor_easy, positive_easy, negative_easy = easy_indices_tuple
            anchor_hard, positive_hard, negative_hard = hard_indices_tuple

            scalar = torch.zeros(len(targets))
            base = torch.where(targets == 3)[0]
            outputs = []
            for i in base:
                indices = torch.where(anchor_hard == i)[0]
                outputs.append(indices)
            if outputs:
                triplets = outputs[0]
                anchor = anchor_hard[triplets]
                positive = positive_hard[triplets]
                negative = negative_hard[triplets]
                e1 = x_poincare[anchor]
                e2 = x_poincare[positive]
                e3 = x_poincare[negative]
                e1 = self.triplet_loss.normalize_embeddings(e1)
                e2 = self.triplet_loss.normalize_embeddings(e2)
                e3 = self.triplet_loss.normalize_embeddings(e3)
                dij = hyp_lca(e1, e2, return_coord=False)
                dik = hyp_lca(e1, e3, return_coord=False)
                scalar[anchor] = int(4)
                scalar[positive] = torch.flatten(dij)
                scalar[negative] = torch.flatten(dik)
                points_reshaped = torch.reshape(points, (points.size(1), points.size(2)))
                points_transpose = torch.transpose(points_reshaped, 0, 1)
                plot_cloud(xyz=points_transpose.numpy(), scalars=scalar, point_size=5.0)

        self.log("score", score)
        self.log("test_loss", test_loss)
        return {'test_loss': test_loss}
