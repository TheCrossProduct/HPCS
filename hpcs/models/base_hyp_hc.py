import os
from typing import Optional

import torch
import numpy as np
import pytorch_lightning as pl

from torch.nn import functional as F
from torch.optim import lr_scheduler
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassJaccardIndex

from pytorch_metric_learning.utils import common_functions as c_f, loss_and_miner_utils as lmu

from scipy.cluster.hierarchy import linkage

from hpcs.optim import RAdam
from hpcs.distances.poincare import project
from hpcs.loss.ultrametric_loss import MetricHyperbolicLoss
from hpcs.utils.viz import plot_hyperbolic_eval
from hpcs.utils.scores import get_optimal_k
from hpcs.utils.data import to_categorical


class BaseSimilarityHypHC(pl.LightningModule):
    def __init__(self, nn_feat: torch.nn.Module,
                 nn_emb: Optional[torch.nn.Module],
                 euclidean_size: int,
                 hyp_size: int,
                 lr: float = 1e-3,
                 margin: float = 0.5,
                 t_per_anchor: int = 50,
                 fraction: float = 1.2,
                 temperature: float = 0.05,
                 anneal_factor: float = 0.5,
                 anneal_step: int = 0,
                 num_class: int = 4,
                 trade_off: float = 0.1,
                 miner: bool = True,
                 cosface: bool = True,
                 plot_inference: bool = True,
                 notebook: bool = False):

        super(BaseSimilarityHypHC, self).__init__()
        self.nn_feat = nn_feat
        self.nn_emb = nn_emb
        self.lr = lr
        self.euclidean_size = euclidean_size
        self.hyp_size = hyp_size
        self.margin = margin
        self.t_per_anchor = t_per_anchor
        self.fraction = fraction
        self.temperature = temperature
        self.anneal_factor = anneal_factor
        self.anneal_step = anneal_step
        self.num_class = num_class
        self.trade_off = trade_off
        self.miner = miner
        self.cosface = cosface
        self.plot_inference = plot_inference
        self.notebook = notebook
        self.scale = torch.nn.Parameter(torch.Tensor([1e-3]), requires_grad=True)

        self.metric_hyp_loss = MetricHyperbolicLoss(margin=self.margin,
                                                    t_per_anchor=self.t_per_anchor,
                                                    fraction=self.fraction,
                                                    scale=self.scale,
                                                    temperature=self.temperature,
                                                    anneal_factor=self.anneal_factor,
                                                    num_class=self.num_class,
                                                    embedding_size=self.hyp_size,
                                                    miner=self.miner,
                                                    cosface=self.cosface)
        self.accuracy = Accuracy(task='multiclass', num_classes=self.num_class, top_k=1)
        self.iou = MulticlassJaccardIndex(num_classes=self.num_class)
        self.save_hyperparameters()

    def set_category(self, category):
        self.category = category

    def _decode_linkage(self, leaves_embeddings):
        """Build linkage matrix from leaves' embeddings. Assume points are normalized to same radius."""
        leaves_embeddings = self.metric_hyp_loss.normalize_embeddings(leaves_embeddings)
        leaves_embeddings = project(leaves_embeddings).detach().cpu()
        Z = linkage(leaves_embeddings, method='complete', metric='cosine')
        return Z

    def compute_accuracy(self, embeddings, labels):
        # labels_reshape = labels.contiguous().reshape(-1)
        logits = self.metric_hyp_loss.get_logits(embeddings, labels)
        logits = F.softmax(logits)

        return self.accuracy(logits, labels)

    def compute_iou(self, embeddings, labels):
        logits = self.metric_hyp_loss.get_logits(embeddings, labels)
        logits = F.softmax(logits)

        return self.iou(logits, labels)

    def compute_losses(self, x_euclidean, x_poincare, labels):
        labels = labels.view(-1, 1)[:, 0]
        losses = {}

        loss = self.metric_hyp_loss.compute_loss(x_euclidean, x_poincare, labels.long())
        losses['loss_metric'] = loss['loss_metric']['losses']
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
        x_euclidean_reshape = x_euclidean.contiguous().view(-1, x_euclidean.shape[-1])
        x_poincare_reshape = x_poincare.contiguous().view(-1, x_poincare.shape[-1])

        losses = self.compute_losses(x_euclidean_reshape, x_poincare_reshape, pts_labels)
        if hasattr(self.metric_hyp_loss, 'loss_cosface'):
            y_true = pts_labels.contiguous().reshape(-1)
            acc = self.compute_accuracy(x_poincare_reshape, y_true.long())
            iou = self.compute_iou(x_poincare_reshape, y_true.long())
            metrics = {'acc': acc, 'iou': iou}
        else:
            metrics = {}
        if testing:
            linkage_matrix = []
            for object_idx in range(points.size(0)):
                Z = self._decode_linkage(x_poincare[object_idx])
                linkage_matrix.append(Z)
            return losses, metrics, x_euclidean, x_poincare, linkage_matrix, points, pts_labels
        else:
            return losses, metrics

    def configure_optimizers(self):
        optim = RAdam(self.parameters(), lr=self.lr)
        scheduler = [
            {
                'scheduler': lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=4,
                                                            min_lr=1e-6, verbose=True),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': True,
            },
        ]

        return [optim], scheduler

    def training_step(self, batch, batch_idx):
        losses, metrics = self.forward(batch, testing=False)
        losses['total_loss'] = self.sum_losses(losses)

        self.log("train_loss", losses)
        train_metrics = {}
        for key in metrics:
            out_key = "train_" + key
            train_metrics[out_key] = metrics[key]
            self.log(out_key, metrics[key], prog_bar=True)

        self.log("scale", self.scale)
        self.log("temperature", self.temperature)

        return {'loss': losses['total_loss'], **train_metrics}

    def training_epoch_end(self, outputs):
        if self.current_epoch and self.anneal_step > 0 and self.current_epoch % self.anneal_step == 0:
            print(f"Annealing temperature at the end of epoch {self.current_epoch}")
            self.temperature = self.metric_hyp_loss.anneal_temperature()
            print("Temperature Value: ", self.temperature)

    def validation_step(self, batch, batch_idx):
        val_losses, metrics = self.forward(batch, testing=False)
        val_loss = self.sum_losses(val_losses)

        val_metrics = {}
        for key in metrics:
            out_key = "val_" + key
            val_metrics[out_key] = metrics[key]
            self.log(out_key, metrics[key])

        self.log("val_loss", val_loss)
        return {'val_loss': val_loss, **val_metrics}

    def test_step(self, batch, batch_idx):
        test_losses, metrics, x_euclidean, x_poincare, linkage_matrix, points, targets = self.forward(batch, testing=True)
        test_loss = self.sum_losses(test_losses)

        indexes = []
        for object_idx in range(points.size(0)):
            best_pred, best_k, best_score = get_optimal_k(targets[object_idx].cpu(), linkage_matrix[object_idx], 'iou')
            indexes.append(best_score)
            if self.plot_inference:
                emb_poincare = self.metric_hyp_loss.normalize_embeddings(x_poincare[object_idx])
                if self.notebook:
                    if 'shapenet' in str(type(self)):
                        dataset_name = 'shapenet'
                    elif 'partnet' in str(type(self)):
                        dataset_name = 'partnet'
                    else:
                        dataset_name = 'base'

                    category = self.category if hasattr(self, 'category') else ''
                    level = 'level_' + str(self.level) if hasattr(self, 'level') else ''

                    screenshot_basedir = os.path.join(os.getcwd(), dataset_name, category, level)

                    if not os.path.exists(screenshot_basedir):
                        os.makedirs(screenshot_basedir, exist_ok=True)
                    idx = batch[0].shape[0] * batch_idx + object_idx

                    screenshot = os.path.join(screenshot_basedir, str(idx)+'.png')
                else:
                    screenshot = False
                plot_hyperbolic_eval(x=points[object_idx].T.cpu(),
                                     y=targets[object_idx].cpu(),
                                     y_pred=best_pred,
                                     emb_hidden=x_euclidean[object_idx].cpu(),
                                     emb_poincare=emb_poincare.cpu(),
                                     linkage_matrix=linkage_matrix[object_idx],
                                     k=best_k,
                                     score=best_score,
                                     show=True,
                                     notebook=self.notebook,
                                     screenshot=screenshot)

        score = torch.mean(torch.tensor(indexes))

        self.log("score", score)
        self.log("test_loss", test_loss)

        test_metrics = {}
        for key in metrics:
            out_key = "test_" + key
            test_metrics[out_key] = metrics[key]
            self.log(out_key, metrics[key])

        return {'test_loss': test_loss, **test_metrics}