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

from hpcs.utils.viz import plot_hyperbolic_eval
from hpcs.utils.scores import eval_clustering, get_optimal_k
from hpcs.loss.ultrametric_loss import TripletHyperbolicLoss

from hpcs.optim import RAdam


class SimilarityHypHC(pl.LightningModule):
    """
    Args:
        nn: torch.nn.Module
            model used to do feature extraction

        embedder: Union[torch.nn.Module, None]
            if not None, module used to embed features from initial space to Poincare's Disk

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

        plot_every: int
            plot validation value every #plot_every epoch

    """
    def __init__(self, nn: torch.nn.Module, embedder: Union[torch.nn.Module, None],
                 sim_distance: str = 'cosine', temperature: float = 0.05, anneal: float = 0.5, anneal_step: int = 0,
                 margin: float = 1.0, init_rescale: float = 1e-3, max_scale: float = 1. - 1e-3, lr: float = 1e-3,
                 patience: int = 10, factor: float = 0.5, min_lr: float = 1e-4,
                 plot_every: int = -1):
        super(SimilarityHypHC, self).__init__()
        self.model = nn
        self.embedder = embedder

        self.triplet_loss = TripletHyperbolicLoss(sim_distance=sim_distance,
                                                  margin=margin,
                                                  init_rescale=init_rescale,
                                                  max_scale=max_scale,
                                                  temperature=temperature,
                                                  anneal=anneal)

        # learning rate
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.plot_interval = plot_every
        self.plot_step = 0
        self.anneal_step = anneal_step

    def _decode_linkage(self, leaves_embeddings):
        """Build linkage matrix from leaves' embeddings. Assume points are normalized to same radius."""
        leaves_embeddings = self.triplet_loss._rescale_emb(leaves_embeddings)
        sim_fn = lambda x, y: np.arccos(np.clip(np.sum(x * y, axis=-1), -1.0, 1.0))
        embeddings = F.normalize(leaves_embeddings, p=2, dim=1).detach().cpu()
        Z = linkage(embeddings, method='single', metric=sim_fn)

        return Z

    def forward(self, x, y, pos, labels=None, batch=None, decode=False):

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)

        batch_size = batch.max() + 1

        # feature extractor
        x = self.model(x, pos, batch)

        if isinstance(self.embedder, torch.nn.Module):
            x_emb = self.embedder(x)
        else:
            x_emb = x

        linkage_mat = []

        if labels is not None:
            x_feat_samples = x[labels]
            y_samples = y[labels]
        else:
            x_feat_samples = x
            y_samples = y

        # loss_triplet, loss_hyphc = self._loss(x_feat_samples=x_feat_samples, y_samples=y_samples,
        #                                       x_emb_samples=x_emb_samples, t_per_anchor=1000)
        losses = self.triplet_loss.compute_loss(embeddings=x_feat_samples,
                                                labels=y_samples,
                                                indices_tuple=None,
                                                ref_emb=None,
                                                ref_labels=None,
                                                t_per_anchor=1000)

        loss_triplet = losses['loss_sim']['losses']
        loss_hyphc = losses['loss_lca']['losses']

        if decode:
            for i in range(batch_size):
                Z = self._decode_linkage(x_emb[batch == i])
                linkage_mat.append(Z)

        return x_emb, loss_triplet, loss_hyphc, linkage_mat

    def _forward(self, data, decode=False):
        if isinstance(data, list):
            data = Batch.from_data_list(data, follow_batch=[]).to(self.device)

        x = data.x
        y = data.y
        pos = data.pos
        batch = data.batch
        if hasattr(data, 'labels'):
            labels = data.labels
        else:
            labels = None

        x, loss_triplet, loss_hyphc, link_mat = self(x=x, y=y, pos=pos, labels=labels, batch=batch, decode=decode)

        return x, loss_triplet, loss_hyphc, link_mat

    def _get_optimal_k(self, y, linkage_matrix):
        best_ri = 0.0
        n_clusters = y.max() + 1
        # min_num_clusters = max(n_clusters - 1, 1)
        best_k = 0
        best_pred = None
        for k in range(1, n_clusters + 5):
            y_pred = fcluster(linkage_matrix, k, criterion='maxclust') - 1
            k_ri = ri(y, y_pred)
            if k_ri > best_ri:
                best_ri = k_ri
                best_k = k
                best_pred = y_pred

        return best_pred, best_k, best_ri

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
        x, loss_triplet, loss_hyphc, _ = self._forward(data)
        loss = loss_triplet + loss_hyphc
        self.log("train_loss", {"total_loss": loss, "triplet_loss": loss_triplet, "hyphc_loss": loss_hyphc}, prog_bar=True)

        return {'loss': loss, 'progress_bar': {'triplet': loss_triplet, 'hyphc': loss_hyphc}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # avg_ri = torch.stack([x['ri'] for x in outputs]).mean()
        # self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        # self.logger.experiment.add_scalar("RandScore/Train", avg_ri, self.current_epoch)
        if self.current_epoch and self.anneal_step > 0 and self.current_epoch % self.anneal_step == 0:
            print(f"Annealing temperature at the end of epoch {self.current_epoch}")
            max_temp = 0.8
            min_temp = 0.01
            self.temperature = max(min(self.temperature * self.anneal, max_temp), min_temp)
            print("Temperature Value: ", self.temperature)

    def validation_step(self, data, batch_idx):
        maybe_plot = self.plot_interval > 0 and ((self.current_epoch + 1) % self.plot_interval == 0)
        x, val_loss_triplet, val_loss_hyphc, linkage_matrix = self._forward(data, decode=maybe_plot)
        val_loss = val_loss_triplet + val_loss_hyphc

        fig = None
        best_ri = 0.0
        if maybe_plot:
            y_pred, k, best_ri = get_optimal_k(data.y.detach().cpu().numpy(), linkage_matrix[0])
            pu_score, nmi_score, ri_score = eval_clustering(y_true=data.y.detach().cpu(), Z=linkage_matrix[0])

            fig = plot_hyperbolic_eval(x=data.x.detach().cpu(),
                                       y=data.y.detach().cpu(),
                                       labels=data.labels.detach().cpu(),
                                       y_pred=y_pred,
                                       emb=self.triplet_loss._rescale_emb(x).detach().cpu(),
                                       linkage_matrix=linkage_matrix[0],
                                       emb_scale=self.rescale.item(),
                                       k=k,
                                       show=False)

            # self.logger.experiment.add_scalar("RandScore/Validation", best_ri, self.plot_step)
            # # self.logger.experiment.add_scalar("AccScore@k/Validation", acc_score, self.plot_step)
            # self.logger.experiment.add_scalar("PurityScore@k/Validation", pu_score, self.plot_step)
            # self.logger.experiment.add_scalar("NMIScore@k/Validation", nmi_score, self.plot_step)
            # self.logger.experiment.add_scalar("RandScore@k/Validation", ri_score, self.plot_step)
            self.plot_step += 1

        self.log("val_loss", val_loss, batch_size=data.batch.shape[0])
        return {'val_loss': val_loss, 'figures': fig, 'best_ri': torch.tensor(best_ri)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
        # self.logger.log_metrics({'val_loss': avg_loss}, step=self.current_epoch)

        figures = [x['figures'] for x in outputs if x['figures'] is not None]

        for n, fig in enumerate(figures):
            tag = n // 10
            step = n % 10
            # self.logger.experiment.add_figure(f"Plots/Validation@Epoch:{self.current_epoch}:{tag}", figure=fig,
            #                                   global_step=step)

    def test_step(self, data, batch_idx):
        x, test_loss_triplet, test_loss_hyphc, linkage_matrix = self._forward(data, decode=True)
        test_loss = test_loss_hyphc + test_loss_triplet

        y_pred_k, k, best_ri = get_optimal_k(data.y.detach().cpu().numpy(), linkage_matrix[0])
        pu_score, nmi_score, ri_score = eval_clustering(y_true=data.y.detach().cpu(), Z=linkage_matrix[0])

        # fig = plot_hyperbolic_eval(x=data.x.detach().cpu(),
        #                            y=data.y.detach().cpu(),
        #                            labels=data.labels.detach().cpu(),
        #                            y_pred=y_pred_k,
        #                            emb=self.triplet_loss._rescale_emb(x).detach().cpu(),
        #                            linkage_matrix=linkage_matrix[0],
        #                            emb_scale=self.rescale.item(),
        #                            k=k,
        #                            show=False)

        # n_clusters = data.y.max() + 1
        # y_pred = fcluster(linkage_matrix[0], n_clusters, criterion='maxclust') - 1
        # ri_score = ri(data.y.detach().cpu().numpy(), y_pred)

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

        self.log("test_loss", test_loss, batch_size=data.batch.shape[0])
        return {'test_loss': test_loss, 'test_ri@k': torch.tensor(ri_score),
                'test_pu@k': torch.tensor(pu_score), 'test_nmi@k': torch.tensor(nmi_score),
                'test_ri': torch.tensor(best_ri), 'k': torch.tensor(k, dtype=torch.float)}

    def test_epoch_end(self, outputs):

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_ri_k = torch.stack([x['test_ri@k'] for x in outputs]).mean()
        std_ri_k = torch.stack([x['test_ri@k'] for x in outputs]).std()
        # avg_acc_k = torch.stack([x['test_acc@k'] for x in outputs]).mean()
        # std_acc_k = torch.stack([x['test_acc@k'] for x in outputs]).std()
        avg_pu_k = torch.stack([x['test_pu@k'] for x in outputs]).mean()
        std_pu_k = torch.stack([x['test_pu@k'] for x in outputs]).std()
        avg_nmi_k = torch.stack([x['test_nmi@k'] for x in outputs]).mean()
        std_nmi_k = torch.stack([x['test_nmi@k'] for x in outputs]).std()
        avg_ri = torch.stack([x['test_ri'] for x in outputs]).mean()
        std_ri = torch.stack([x['test_ri'] for x in outputs]).std()
        avg_best_k = torch.stack([x['k'] for x in outputs]).mean()
        std_best_k = torch.stack([x['k'] for x in outputs]).std()

        metrics = {'ari@k': avg_ri_k, 'ari@k-std': std_ri_k,
                   # 'acc@k': avg_acc_k, 'acc@k-std': std_acc_k,
                   'purity@k': avg_pu_k, 'purity@k-std': std_pu_k,
                   'nmi@k': avg_nmi_k, 'nmi@k-std': std_nmi_k,
                   'ari': avg_ri, 'ari-std': std_ri,
                   'best_k': avg_best_k, 'std_k': std_best_k}

        # self.logger.log_metrics(metrics, step=len(outputs))

        # return {'test_loss': avg_loss,
        #         'test_ri': avg_ri,
        #         'ari@k': avg_ri_k, 'ari@k-std': std_ri_k,
        #         # 'acc@k': avg_acc_k, 'acc@k-std': std_acc_k,
        #         'purity@k': avg_pu_k, 'purity@k-std': std_pu_k,
        #         'nmi@k': avg_nmi_k, 'nmi@k-std': std_nmi_k}


# DEPRECATED CLASS
# class HyperbolicSeg(pl.LightningModule):
#     def __init__(self, k: int, hidden_feat: int, num_classes: int, transform: bool = False, aggr='max',
#                  dropout=0.3, negative_slope=0.2,
#                  sim_distance: str = 'cosine', t_per_anchor=100, label_ratio=0.3,
#                  temperature: float = 0.05, margin: float = 1.0, init_rescale: float = 1e-3,
#                  max_scale: float = 1. - 1e-3, lr: float = 1e-3, patience: int = 10,
#                  factor: float = 0.5, min_lr: float = 1e-4, weight_decay=1e-4):
#
#         super(HyperbolicSeg, self).__init__()
#         self.num_classes = num_classes
#         # dgcnn model
#         self.transform = transform
#         if self.transform:
#             self.tnet = TransformNet()
#
#         self.conv1 = DynamicEdgeConv(
#             nn=MLP([2 * 3, hidden_feat, hidden_feat], negative_slope=negative_slope),
#             k=k,
#             aggr=aggr
#         )
#         self.conv2 = DynamicEdgeConv(
#             nn=MLP([2 * hidden_feat, hidden_feat, hidden_feat], negative_slope=negative_slope),
#             k=k,
#             aggr=aggr
#         )
#         self.conv3 = DynamicEdgeConv(
#             nn=MLP([2 * hidden_feat, hidden_feat, hidden_feat], negative_slope=negative_slope),
#             k=k,
#             aggr=aggr
#         )
#
#         # used to embedd hidden features to poincare disk
#         self.embedder = MLP([hidden_feat, hidden_feat, hidden_feat, 2], negative_slope=negative_slope)
#         self.lin1 = MLP([3 * hidden_feat, 1024], bias=False, negative_slope=negative_slope)
#
#         self.mlp = Seq(
#             MLP([1024, 256], negative_slope=0.2),
#             Dropout(dropout),
#             MLP([256, 128], negative_slope=0.2),
#             Dropout(dropout),
#             Linear(128, num_classes)
#         )
#         # parameters for the triplet loss term
#         self.t_per_anchor = t_per_anchor
#         self.label_ratio = label_ratio
#         self.margin = margin
#
#         # parameters for hyperbolic disk projection
#         self.rescale = torch.nn.Parameter(torch.Tensor([init_rescale]), requires_grad=True)
#         self.max_scale = max_scale
#         self.temperature = temperature
#         # least common ancestor on hyperbolic disk
#         self.distance_lca = HyperbolicLCA()
#
#         if sim_distance == 'cosine':
#             self.distace_sim = distances.CosineSimilarity()
#             self.loss_triplet_sim = losses.TripletMarginLoss(distance=self.distace_sim, margin=self.margin)
#         elif sim_distance == 'euclidean':
#             self.distace_sim = distances.LpDistance()
#             self.loss_triplet_sim = losses.TripletMarginLoss(distance=self.distace_sim, margin=self.margin,
#                                                              embedding_regularizer=regularizers.LpRegularizer())
#         else:
#             raise ValueError(
#                 f"The option {sim_distance} is not available for sim_distance. The only available are ['cosine', 'euclidean'].")
#
#         # parameters for optimizers
#         self.lr = lr
#         self.factor = factor
#         self.patience = patience
#         self.min_lr = min_lr
#         self.weight_decay = weight_decay
#
#         # parameter to control manual optimization
#         self.automatic_optimization = False
#
#     def configure_optimizers(self):
#
#         optim_adam = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#
#         optim_radam = RAdam(self.parameters(), lr=self.lr)
#
#         optimizers = [optim_adam, optim_radam]
#         schedulers = [
#             {
#                 'scheduler': lr_scheduler.ReduceLROnPlateau(optimizers[1], mode='min', factor=0.5, patience=10,
#                                                             min_lr=1e-4, verbose=True),
#                 'monitor': 'val_loss',
#                 'interval': 'epoch',
#                 'frequency': 1,
#                 'strict': True,
#             },
#             lr_scheduler.StepLR(optimizers[0], step_size=20, gamma=0.5)
#         ]
#
#         return optimizers, schedulers
#
#     def _rescale_emb(self, embeddings):
#         """Normalize leaves embeddings to have the lie on a diameter."""
#         min_scale = 1e-4  # self.init_size
#         max_scale = self.max_scale
#         return F.normalize(embeddings, p=2, dim=1) * self.rescale.clamp_min(min_scale).clamp_max(max_scale)
#
#     def _loss(self, x_feat_samples, y_samples, x_emb_samples, t_per_anchor=100):
#         # indices_tuple = lmu.convert_to_triplets(None, y_samples, t_per_anchor='all')
#         indices_tuple = lmu.convert_to_triplets(None, y_samples, t_per_anchor=t_per_anchor)
#
#         anchor_idx, positive_idx, negative_idx = indices_tuple
#         # print("Len Anchor: ", len(anchor_idx))
#         if len(anchor_idx) == 0:
#             loss_triplet_sim = self.loss_triplet_sim(x_feat_samples, y_samples)
#             loss_triplet_lca = torch.tensor([0.0], device=loss_triplet_sim.device)
#             # raise ValueError("Anchor idx is empty ")
#             return loss_triplet_lca, loss_triplet_sim
#         # #
#         if isinstance(self.distace_sim, distances.CosineSimilarity):
#             mat_sim = 0.5 * (1 + self.distace_sim(x_feat_samples))
#         else:
#             # mat_sim = 0.5 * (1 + self.distace_sim(project(x_emb_samples)))
#             mat_sim = torch.exp(-self.distace_sim(x_feat_samples))
#             # print("euclidean", mat_sim.max(), mat_sim.min())
#         #
#         # mat_sim = self.distace_sim(x_feat_samples)
#         mat_lca = self.distance_lca(self._rescale_emb(x_emb_samples))
#         # print(f"sim values: max {mat_sim.max()}, min: {mat_sim.min()}")
#         wij = mat_sim[anchor_idx, positive_idx]
#         wik = mat_sim[anchor_idx, negative_idx]
#         wjk = mat_sim[positive_idx, negative_idx]
#
#         dij = mat_lca[anchor_idx, positive_idx]
#         dik = mat_lca[anchor_idx, negative_idx]
#         djk = mat_lca[positive_idx, negative_idx]
#
#         # loss proposed by Chami et al.
#         sim_triplet = torch.stack([wij, wik, wjk]).T
#         lca_triplet = torch.stack([dij, dik, djk]).T
#         weights = torch.softmax(lca_triplet / self.temperature, dim=-1)
#
#         w_ord = torch.sum(sim_triplet * weights, dim=-1, keepdim=True)
#         total = torch.sum(sim_triplet, dim=-1, keepdim=True) - w_ord
#         loss_triplet_lca = torch.mean(total) + mat_sim.mean()
#
#         loss_triplet_sim = self.loss_triplet_sim(x_feat_samples, y_samples)
#
#         return loss_triplet_sim, loss_triplet_lca
#
#     def forward(self, x, y, batch=None, labels=None):
#
#         if self.transform:
#             tr = self.tnet(x, batch=batch)
#
#             if batch is None:
#                 x = torch.matmul(x, tr[0])
#             else:
#                 batch_size = batch.max().item() + 1
#                 x = torch.cat([torch.matmul(x[batch == i], tr[i]) for i in range(batch_size)])
#
#         # feature extractor
#         x1 = self.conv1(x, batch)
#         x2 = self.conv2(x1, batch)
#         x3 = self.conv3(x2, batch)
#
#         if labels is not None:
#             # if labels we compute hyperbolic loss
#             x_emb = self.embedder(x3)
#             x_feat_samples = x3[labels]
#             x_emb_samples = x_emb[labels]
#             y_samples = y[labels]
#
#             loss_triplet, loss_hyphc = self._loss(x_feat_samples=x_feat_samples, y_samples=y_samples,
#                                                   x_emb_samples=x_emb_samples, t_per_anchor=self.t_per_anchor)
#             return x_emb, loss_triplet, loss_hyphc
#         else:
#
#             out = self.lin1(torch.cat([x1, x2, x3], dim=1))
#             out = self.mlp(out)
#
#             return F.log_softmax(out, dim=-1)
#
#     def step(self, data, compute_hier_loss=False):
#         if isinstance(data, list):
#             data = Batch.from_data_list(data, follow_batch=[]).to(self.device)
#
#         y = data.y
#         x = data.pos
#         batch = data.batch
#         if compute_hier_loss:
#             classes = torch.unique(y)
#             labels = torch.zeros(x.size(0), dtype=torch.bool)
#             for c in classes:
#                 # picking at least one label per class
#                 p = torch.zeros(x.size(0))
#                 p[y == c] = 1
#                 num_labels = max(int(p.sum() * self.label_ratio), 1)
#                 idx = p.multinomial(num_samples=num_labels, replacement=False)
#                 labels[idx] = True
#
#             labels = labels.to(x.device)
#
#             # labels = torch.rand_like(x[:,0]) <= self.label_ratio
#             x_emb, loss_triplet, loss_hyphc = self(x=x, y=y, batch=batch, labels=labels)
#
#             return x_emb, loss_triplet, loss_hyphc
#         else:
#             y_hat = self(x=x, y=y, batch=batch)
#             loss = F.nll_loss(y_hat, y)
#
#             _, y_pred = y_hat.max(dim=1)
#             acc_score = accuracy(y_pred, y, num_classes=self.num_classes)
#             iou_score = iou(y_pred, y, num_classes=self.num_classes)
#
#             return loss, acc_score, iou_score
#
#     def training_step(self, data, batch_idx, optimizer_idx):
#         # manual
#         (opt_adam, opt_radam) = self.optimizers()
#
#         loss, acc_score, iou_score = self.step(data, compute_hier_loss=False)
#
#         self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
#         self.log('acc', acc_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#
#         # make sure there are no grads
#         # if batch_idx > 0:
#         #     assert torch.all(self.layer.weight.grad == 0)
#
#         self.manual_backward(loss, opt_adam)
#         opt_adam.step()
#         opt_adam.zero_grad()
#         # assert torch.all(self.layer.weight.grad == 0)
#
#         # hyphc
#         x_emb, loss_triplet, loss_hyphc = self.step(data, compute_hier_loss=True)
#         loss_hc = loss_triplet + loss_hyphc
#         self.log('loss_triplet', loss_triplet, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('loss_hyphc', loss_hyphc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#
#         # ensure we forward the correct params to the optimizer
#         # without retain_graph we can't do multiple backward passes
#         self.manual_backward(loss_hc, opt_radam, retain_graph=True)
#         self.manual_backward(loss_hc, opt_adam, retain_graph=True)
#
#         opt_radam.step()
#         opt_radam.zero_grad()
#
#         # return {'loss': loss, 'loss_triplet': loss_triplet, 'loss_hyphc': loss_hyphc, 'acc': acc_score, 'iou': iou_score}
#
#     def training_epoch_end(self, outputs):
#         self.epoch_end(outputs)
#
#     def epoch_end(self, outputs):
#         """
#         Run at epoch end for training or validation. Can be overriden in models.
#         """
#         return outputs
#
#     def validation_step(self, data, batch_idx):
#
#         val_loss, val_acc_score, val_iou_score = self.step(data, compute_hier_loss=False)
#         x_emb, val_loss_triplet, val_loss_hyphc = self.step(data, compute_hier_loss=True)
#         self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('val_loss_triplet', val_loss_triplet, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('val_loss_hyphc', val_loss_hyphc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('val_acc', val_acc_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('val_iou', val_iou_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#
#         # return {'val_loss': val_loss, 'val_loss_triplet': val_loss_triplet, 'val_loss_hyphc': val_loss_hyphc,
#         #         'val_acc': val_acc_score, 'val_iou': val_iou_score}
#
#     def validation_epoch_end(self, outputs):
#         self.epoch_end(outputs)
#
#     def test_step(self, data, batch_idx):
#         test_loss, test_acc_score, test_iou_score = self.step(data, compute_hier_loss=False)
#         x_emb, test_loss_triplet, test_loss_hyphc = self.step(data, compute_hier_loss=True)
#
#         self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('test_loss_triplet', test_loss_triplet, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('test_loss_hyphc', test_loss_hyphc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('test_acc', test_acc_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('test_iou', test_iou_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#
#         # return {'test_loss': test_loss,
#         #         'test_acc': test_acc_score, 'test_iou': test_iou_score}
#
#     def test_epoch_end(self, outputs):
#         self.epoch_end(outputs)