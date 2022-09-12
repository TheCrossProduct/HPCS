import scipy.spatial.distance
import wandb
import torch
import numpy as np
import pytorch_lightning as pl
from typing import Union

from scipy.cluster.hierarchy import fcluster, linkage

from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch_geometric.data import Batch
# from pytorch_lightning.metrics.functional import accuracy, iou
# from torchmetrics.functional import accuracy
from sklearn.metrics.cluster import adjusted_rand_score as ri

from hpcs.utils.viz import plot_hyperbolic_eval, plot_cloud
from hpcs.utils.scores import eval_clustering, get_optimal_k
from hpcs.loss.ultrametric_loss import TripletHyperbolicLoss
from hpcs.distances.poincare import project

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
                 margin: float = 1.0, init_rescale: float = 1e-3, max_scale: float = 1. - 1e-3, lr: float = 1e-3,
                 patience: int = 10, factor: float = 0.5, min_lr: float = 1e-4):
        super(SimilarityHypHC, self).__init__()
        self.save_hyperparameters()
        self.model = nn

        self.scale = torch.nn.Parameter(torch.Tensor([init_rescale]), requires_grad=True)

        self.triplet_loss = TripletHyperbolicLoss(sim_distance=sim_distance,
                                                  margin=margin,
                                                  scale=self.scale,
                                                  max_scale=max_scale,
                                                  temperature=temperature,
                                                  anneal=anneal)

        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.anneal_step = anneal_step

    def _decode_linkage(self, leaves_embeddings):
        """Build linkage matrix from leaves' embeddings. Assume points are normalized to same radius."""
        leaves_embeddings = self.triplet_loss.normalize_embeddings(leaves_embeddings)
        leaves_embeddings = project(leaves_embeddings).detach().cpu()
        Z = linkage(leaves_embeddings, method='ward', metric='euclidean')

        return Z


    def forward(self, data, decode=False):
        points, label, targets = data

        # trot = None
        # rot = 'so3'
        # if rot == 'z':
        #     trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True)
        # elif rot == 'so3':
        #     trot = Rotate(R=random_rotations(points.shape[0]))
        # if trot is not None:
        #     points = trot.transform_points(points.cpu())
        #
        # points = points.detach().numpy()
        # points = provider.random_point_dropout(points)
        # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        # points = torch.Tensor(points)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        points, label, target = points.float().to(device), label.long().to(device), targets.long().to(device)
        points = points.transpose(2, 1)

        num_classes = 16
        x_emb = self.model(points, to_categorical(label, num_classes))
        x_emb_reshape = torch.reshape(x_emb, (x_emb.size(0) * x_emb.size(1), x_emb.size(2)))
        x_poincare = project(self.scale * x_emb)
        x_poincare_reshape = torch.reshape(x_poincare, (x_poincare.size(0) * x_poincare.size(1), x_poincare.size(2)))

        # x_poincare = project(self.scale * points)
        # x_emb = self.model(x_poincare, to_categorical(label, num_classes))
        # x_emb = torch.reshape(x_emb, (x_emb.size(0) * x_emb.size(1), x_emb.size(2)))

        x_feat_samples = x_poincare_reshape
        y_samples = targets.view(-1, 1)[:, 0]

        losses = self.triplet_loss.compute_loss(embeddings=x_feat_samples,
                                                labels=y_samples,
                                                indices_tuple=None,
                                                ref_emb=None,
                                                ref_labels=None,
                                                t_per_anchor=1000)

        loss_triplet = losses['loss_sim']['losses']
        loss_hyphc = losses['loss_lca']['losses']

        linkage_mat = []
        if decode:
            for i in range(points.size(0)):
                Z = self._decode_linkage(x_poincare[i])
                linkage_mat.append(Z)

        return x_emb_reshape, x_poincare_reshape, loss_triplet, loss_hyphc, linkage_mat


    def _forward(self, data, decode=False):
        points, label, targets = data

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
        x_emb_reshape = torch.reshape(x_emb, (x_emb.size(0) * x_emb.size(1), x_emb.size(2)))
        x_poincare = project(self.scale * x_emb)
        x_poincare_reshape = torch.reshape(x_poincare, (x_poincare.size(0) * x_poincare.size(1), x_poincare.size(2)))

        # x_poincare = project(self.scale * points)
        # x_emb = self.model(x_poincare, to_categorical(label, num_classes))
        # x_emb = torch.reshape(x_emb, (x_emb.size(0) * x_emb.size(1), x_emb.size(2)))

        x_feat_samples = x_poincare_reshape
        y_samples = targets.view(-1, 1)[:, 0]

        losses = self.triplet_loss.compute_loss(embeddings=x_feat_samples,
                                                labels=y_samples,
                                                indices_tuple=None,
                                                ref_emb=None,
                                                ref_labels=None,
                                                t_per_anchor=1000)

        loss_triplet = losses['loss_sim']['losses']
        loss_hyphc = losses['loss_lca']['losses']

        linkage_mat = []
        if decode:
            for i in range(points.size(0)):
                Z = self._decode_linkage(x_poincare[i])
                linkage_mat.append(Z)

        return x_emb_reshape, x_poincare_reshape, loss_triplet, loss_hyphc, linkage_mat


    def configure_optimizers(self):
        optim = RAdam(self.parameters(), lr=self.lr)
        scheduler = [
            {
                'scheduler': lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10,
                                                            min_lr=1e-4, verbose=True),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': True,
            },
        ]

        return [optim], scheduler

    def training_step(self, data, batch_idx):
        x, x_poincare, loss_triplet, loss_hyphc, _ = self.forward(data)
        loss = loss_triplet + loss_hyphc

        self.log("train_loss", {"total_loss": loss, "triplet_loss": loss_triplet, "hyphc_loss": loss_hyphc})
        return {'loss': loss, 'progress_bar': {'triplet': loss_triplet, 'hyphc': loss_hyphc}}

    def training_epoch_end(self, outputs):
        if self.current_epoch and self.anneal_step > 0 and self.current_epoch % self.anneal_step == 0:
            print(f"Annealing temperature at the end of epoch {self.current_epoch}")
            max_temp = 0.8
            min_temp = 0.01
            self.temperature = max(min(self.temperature * self.anneal, max_temp), min_temp)
            print("Temperature Value: ", self.temperature)

    def validation_step(self, data, batch_idx):
        x, x_poincare, val_loss_triplet, val_loss_hyphc, linkage_matrix = self.forward(data)
        val_loss = val_loss_triplet + val_loss_hyphc

        self.log("val_loss", val_loss)
        return {'val_loss': val_loss}

    def test_step(self, data, batch_idx):
        x, x_poincare, test_loss_triplet, test_loss_hyphc, linkage_matrix = self.forward(data, decode=True)
        test_loss = test_loss_hyphc + test_loss_triplet

        points, label, targets = data
        points = torch.reshape(points, (points.size(0) * points.size(1), points.size(2)))
        targets = targets.view(-1, 1)[:, 0]

        y_pred_k, k, best_ri = get_optimal_k(targets.detach().cpu().numpy(), linkage_matrix[0])
        pu_score, nmi_score, ri_score = eval_clustering(y_true=targets.detach().cpu(), Z=linkage_matrix[0])

        plot_hyperbolic_eval(x=points.detach().cpu(),
                             y=targets.detach().cpu(),
                             y_pred=y_pred_k,
                             emb_hidden=x.detach().cpu(),
                             emb_poincare=self.triplet_loss.normalize_embeddings(x_poincare).detach().cpu(),
                             linkage_matrix=linkage_matrix[0],
                             k=k,
                             show=True)

        # plot_cloud(xyz=data.pos.numpy(), scalars=y_pred_k, point_size=3.0, notebook=True)
        plot_cloud(xyz=points.detach().cpu().numpy(), scalars=y_pred_k, point_size=3.0, notebook=True)

        n_clusters = targets.max() + 1
        y_pred = fcluster(linkage_matrix[0], n_clusters, criterion='maxclust') - 1
        ri_score = ri(targets.detach().cpu().numpy(), y_pred)

        # self.logger.experiment.add_scalar("Loss/Test", test_loss, batch_idx)
        # self.logger.experiment.add_scalar("RandScore/Test", best_ri, batch_idx)
        # # self.logger.experiment.add_scalar("AccScore@k/Test", acc_score,  batch_idx)
        # self.logger.experiment.add_scalar("PurityScore@k/Test", pu_score, batch_idx)
        # self.logger.experiment.add_scalar("NMIScore@k/Test", nmi_score, batch_idx)
        # self.logger.experiment.add_scalar("RandScore@k/Test", ri_score, batch_idx)

        tag = batch_idx // 10
        step = batch_idx % 10
        # self.logger.experiment.add_figure(f"Plots/Test:{tag}", figure=fig, global_step=step)
        # self.logger.log_metrics({'ari@k': ri_score, 'purity@k': pu_score, 'nmi@k': nmi_score,
        #                          'ari': best_ri, 'best_k': k}, step=batch_idx)


        self.log("test_loss", test_loss, batch_size=points.shape[0], on_step=False, on_epoch=True)
        return {'test_loss': test_loss}
                # 'test_ri@k': torch.tensor(ri_score),
                # 'test_pu@k': torch.tensor(pu_score), 'test_nmi@k': torch.tensor(nmi_score),
                # 'test_ri': torch.tensor(best_ri), 'k': torch.tensor(k, dtype=torch.float)}

    def test_epoch_end(self, outputs):

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        # avg_ri_k = torch.stack([x['test_ri@k'] for x in outputs]).mean()
        # std_ri_k = torch.stack([x['test_ri@k'] for x in outputs]).std()
        # # avg_acc_k = torch.stack([x['test_acc@k'] for x in outputs]).mean()
        # # std_acc_k = torch.stack([x['test_acc@k'] for x in outputs]).std()
        # avg_pu_k = torch.stack([x['test_pu@k'] for x in outputs]).mean()
        # std_pu_k = torch.stack([x['test_pu@k'] for x in outputs]).std()
        # avg_nmi_k = torch.stack([x['test_nmi@k'] for x in outputs]).mean()
        # std_nmi_k = torch.stack([x['test_nmi@k'] for x in outputs]).std()
        # avg_ri = torch.stack([x['test_ri'] for x in outputs]).mean()
        # std_ri = torch.stack([x['test_ri'] for x in outputs]).std()
        # avg_best_k = torch.stack([x['k'] for x in outputs]).mean()
        # std_best_k = torch.stack([x['k'] for x in outputs]).std()

        # predictions = torch.from_numpy(np.stack([x['prediction'] for x in outputs]))

        # metrics = {'ari@k': avg_ri_k, 'ari@k-std': std_ri_k,
        #            # 'acc@k': avg_acc_k, 'acc@k-std': std_acc_k,
        #            'purity@k': avg_pu_k, 'purity@k-std': std_pu_k,
        #            'nmi@k': avg_nmi_k, 'nmi@k-std': std_nmi_k,
        #            'ari': avg_ri, 'ari-std': std_ri,
        #            'best_k': avg_best_k, 'std_k': std_best_k}

        # self.logger.log_metrics(metrics, step=len(outputs))

        return {'test_loss': avg_loss}
                # 'test_ri': avg_ri,
                # 'ari@k': avg_ri_k, 'ari@k-std': std_ri_k,
                # # 'acc@k': avg_acc_k, 'acc@k-std': std_acc_k,
                # 'purity@k': avg_pu_k, 'purity@k-std': std_pu_k,
                # 'nmi@k': avg_nmi_k, 'nmi@k-std': std_nmi_k}

