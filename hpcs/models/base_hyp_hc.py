from typing import Optional

import torch
import numpy as np
import pytorch_lightning as pl

from torch.nn import functional as F
from torch.optim import lr_scheduler

from scipy.cluster.hierarchy import linkage

from hpcs.optim import RAdam
from hpcs.distances.poincare import project
from hpcs.loss.ultrametric_loss import MetricHyperbolicLoss
from hpcs.loss.ultrametric_loss import HyperbolicHCLoss
from hpcs.utils.viz import plot_hyperbolic_eval
from hpcs.utils.scores import get_optimal_k


class BaseSimilarityHypHC(pl.LightningModule):
    def __init__(self, nn_feat: torch.nn.Module,
                 nn_emb: Optional[torch.nn.Module],
                 lr: float = 1e-3,
                 embedding: int = 6,
                 margin: float = 0.5,
                 t_per_anchor: int = 50,
                 fraction: float = 1.2,
                 temperature: float = 0.05,
                 anneal_factor: float = 0.5,
                 anneal_step: int = 0,
                 num_class: int = 4,
                 trade_off: float = 0.1,
                 radius: float = 1.0,
                 metric_learning: bool = True,
                 cosface: bool = True,
                 hierarchical: bool = False,
                 hierarchy_list: list = [],
                 plot_inference: bool = False):

        super(BaseSimilarityHypHC, self).__init__()
        self.save_hyperparameters()
        self.nn_feat = nn_feat
        self.nn_emb = nn_emb
        self.lr = lr
        self.embedding = embedding
        self.margin = margin
        self.t_per_anchor = t_per_anchor
        self.fraction = fraction
        self.temperature = temperature
        self.anneal_factor = anneal_factor
        self.anneal_step = anneal_step
        self.num_class = num_class
        self.trade_off = trade_off
        self.radius = radius
        self.metric_learning = metric_learning
        self.cosface = cosface
        self.hierarchical = hierarchical
        self.hierarchy_list = hierarchy_list
        self.plot_inference = plot_inference

        if self.metric_learning:
            self.scale = torch.tensor([self.radius - 5e-2])
            self.metric_hyp_loss = MetricHyperbolicLoss(margin=self.margin,
                                                        t_per_anchor=self.t_per_anchor,
                                                        fraction=self.fraction,
                                                        scale=self.scale,
                                                        temperature=self.temperature,
                                                        anneal_factor=self.anneal_factor,
                                                        num_class=self.num_class,
                                                        embedding=self.embedding,
                                                        cosface=self.cosface,
                                                        hierarchical=self.hierarchical,
                                                        hierarchy_list=self.hierarchy_list)
        else:
            self.scale = torch.nn.Parameter(torch.Tensor([1e-3]), requires_grad=True)
            self.hyp_loss = HyperbolicHCLoss(temperature=self.temperature, t_per_anchor=self.t_per_anchor, distance='cosine')


    def _decode_linkage(self, leaves_embeddings):
        """Build linkage matrix from leaves' embeddings. Assume points are normalized to same radius."""
        leaves_embeddings = self.metric_hyp_loss.normalize_embeddings(leaves_embeddings)
        leaves_embeddings = project(leaves_embeddings).detach().cpu()
        Z = linkage(leaves_embeddings, method='ward', metric='euclidean')
        return Z

    def compute_losses(self, x_euclidean, x_poincare, labels):
        labels = labels.view(-1, 1)[:, 0]
        losses = {}
        if self.metric_learning:
            if self.nn_emb is not None:
                loss = self.metric_hyp_loss.compute_loss(x_poincare, labels.long())
                losses['loss_metric'] = loss['loss_metric']['losses']
            else:
                loss = self.metric_hyp_loss.compute_loss(x_euclidean, labels.long())
                losses['loss_metric'] = loss['loss_metric']['losses']
        else:
            loss = self.hyp_loss.compute_loss(x_poincare, self.scale)
            losses['loss_hyp'] = loss['loss_hyp']['losses'] * self.trade_off
        return losses

    def _forward(self, batch, testing: bool):
        raise NotImplemented

    def sum_losses(self, losses: dict):
        total_loss = 0
        for loss in losses.values():
            total_loss += loss
        return total_loss

    def forward(self, batch, testing: bool = False):
        points, x_euclidean, x_poincare, pts_labels = self._forward(batch, testing)
        losses = self.compute_losses(x_euclidean, x_poincare, pts_labels)

        if testing:
            linkage_matrix = []
            for object_idx in range(points.size(0)):
                Z = self._decode_linkage(x_poincare[object_idx])
                linkage_matrix.append(Z)
            return losses, x_euclidean, x_poincare, linkage_matrix, points, pts_labels
        else:
            return losses

    def configure_optimizers(self):
        optim = RAdam(self.parameters(), lr=self.lr)
        scheduler = [
            {
                'scheduler': lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2,
                                                            min_lr=1e-6, verbose=True),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': True,
            },
        ]

        return [optim], scheduler

    def training_step(self, batch, batch_idx):
        losses = self.forward(batch, testing=False)
        losses['total_loss'] = self.sum_losses(losses)

        self.log("train_loss", losses)
        self.log("scale", self.scale)
        self.log("temperature", self.temperature)
        return {'loss': losses['total_loss'], 'progress_bar': losses}

    def training_epoch_end(self, outputs):
        if self.current_epoch and self.anneal_step > 0 and self.current_epoch % self.anneal_step == 0:
            print(f"Annealing temperature at the end of epoch {self.current_epoch}")
            self.temperature = self.metric_hyp_loss.anneal_temperature()
            print("Temperature Value: ", self.temperature)

    def validation_step(self, batch, batch_idx):
        val_losses = self.forward(batch, testing=False)
        val_loss = self.sum_losses(val_losses)

        self.log("val_loss", val_loss)
        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        test_losses, x_euclidean, x_poincare, linkage_matrix, points, targets = self.forward(batch, testing=True)
        test_loss = self.sum_losses(test_losses)

        indexes = []
        for object_idx in range(points.size(0)):
            best_pred, best_k, best_score = get_optimal_k(targets[object_idx], linkage_matrix[object_idx], 'iou')
            if self.plot_inference:
                emb_poincare = self.metric_hyp_loss.normalize_embeddings(x_poincare[object_idx])
                plot_hyperbolic_eval(x=points[object_idx].T.cpu(),
                                     y=targets[object_idx].cpu(),
                                     y_pred=best_pred,
                                     emb_hidden=x_euclidean[object_idx].cpu(),
                                     emb_poincare=emb_poincare.cpu(),
                                     linkage_matrix=linkage_matrix[object_idx],
                                     k=best_k,
                                     score=best_score,
                                     show=True)

            indexes.append(best_score)
        score = torch.mean(torch.tensor(indexes))

        self.log("score", score)
        self.log("test_loss", test_loss)
        return {'test_loss': test_loss}