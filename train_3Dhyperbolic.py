import os
import argparse
import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from hpcs.nn.models._hyp_hc import SimilarityHypHC
from hpcs.nn.models.encoders.dgcnn import DGCNN
from hpcs.nn.models.encoders.euler import EulerFeatExtract
from hpcs.nn.models.encoders.point_transformer import PointTransformer
from hpcs.nn.models.encoders.pointnet2 import PointNet2

from hpcs.nn.models.networks._mlp import MLP


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='hyperbolic', type=str, help='dirname for logs')
    parser.add_argument('--data', default='shapenet', type=str, help='name of dataset to use')
    # parser.add_argument('--train_samples', default=100, type=int, help='number of samples in training set')
    # parser.add_argument('--valid_samples', default=10, type=int, help='number of samples in valid set')
    # parser.add_argument('--test_samples', default=10, type=int, help='number of samples in test set')
    # parser.add_argument('--max_points', default=300, type=int, help='number of points in each sample')
    # parser.add_argument('--num_labels', default=0.3, type=float, help='number/ratio of labels to use in each sample')
    # parser.add_argument('--min_noise', default=0.12, type=float, help='min value of noise to use')
    # parser.add_argument('--max_noise', default=0.15, type=float, help='max value of noise to use')
    # parser.add_argument('--cluster_std', default=0.1, type=float, help='std blobs')
    # parser.add_argument('--num_blobs', default=3, type=int, help='number of blobs in blob/aniso/varied')
    parser.add_argument('--model', default='point_transformer', type=str, help='model to use to extract features')
    parser.add_argument('--embedder', help='if True add a an embedding model from the feature space to B2', action='store_true')
    parser.add_argument('--k', default=10, type=int, help='if model dgcnn, k is the number of neigh to take into account')
    parser.add_argument('--hidden', default=64, type=int, help='number of hidden features')
    parser.add_argument('--negative_slope', default=0.2, type=float, help='negative slope for leaky relu in the feature extractor')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout in the feature extractor')
    parser.add_argument('--cosine', help='if True add use cosine dist in DynamicEdgeConv', action='store_true')
    parser.add_argument('--distance', default='cosine', type=str, help='distance to use to compute triplets')
    parser.add_argument('--margin', default=1.0, type=float, help='margin value to use in triplet loss')
    parser.add_argument('--temperature', default=0.05, type=float, help='rescale softmax value used in the hyphc loss')
    parser.add_argument('--annealing', default=1.0, type=float, help='annealing factor')
    parser.add_argument('--anneal_step', default=0, type=int, help='use annealing each n step')
    parser.add_argument('--batch', default=4, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--patience', default=50, type=int, help='patience value for early stopping')
    parser.add_argument('--plot', default=-1, type=int, help='interval in which we plot prediction on validation batch')
    parser.add_argument('--gpu', default="", type=str, help='use gpu')
    parser.add_argument('--distributed', help='if True run on a cluster machine', action='store_true')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--fixed_points', type=int, default=128)


    args = parser.parse_args()

    logdir = args.logdir
    dataname = args.data
    epochs = args.epochs
    # train_samples = args.train_samples
    # valid_samples = args.valid_samples
    # test_samples = args.test_samples
    # num_labels = args.num_labels
    # max_points = args.max_points
    # min_noise = args.min_noise
    # max_noise = args.max_noise
    # cluster_std = args.cluster_std
    # num_blobs = args.num_blobs
    model_name = args.model
    embedder = args.embedder
    k = args.k
    hidden = args.hidden
    negative_slope = args.negative_slope
    dropout = args.dropout
    cosine = args.cosine
    distance = args.distance
    margin = args.margin
    temperature = args.temperature
    annealing = args.annealing
    anneal_step = args.anneal_step
    batch = args.batch
    lr = args.lr
    patience = args.patience
    plot_every = args.plot
    distr = args.distributed
    num_workers = args.num_workers
    fixed_points = args.fixed_points


    category = 'Airplane'  # Pass in `None` to train on all categories.
    path = osp.join(osp.dirname(osp.realpath("hpcs\data\ShapeNet")), '..', 'data', 'ShapeNet')

    pre_transform, transform = T.NormalizeScale(), T.FixedPoints(fixed_points)
    train_dataset = ShapeNet(path, category, split='train', transform=transform, pre_transform=pre_transform)
    valid_dataset = ShapeNet(path, category, split='val', transform=transform, pre_transform=pre_transform)
    test_dataset = ShapeNet(path, category, split='test', transform=transform, pre_transform=pre_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=num_workers)


    class MyTensorBoardLogger(TensorBoardLogger):
        def __init__(self, *args, **kwargs):
            super(MyTensorBoardLogger, self).__init__(*args, **kwargs)

        def log_hyperparams(self, *args, **kwargs):
            pass

        @rank_zero_only
        def log_hyperparams_metrics(self, params: dict, metrics: dict) -> None:
            from torch.utils.tensorboard.summary import hparams
            params = self._convert_params(params)
            exp, ssi, sei = hparams(params, metrics)
            writer = self.experiment._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)
            # some alternative should be added
            self.hparams.update(params)

    if len(args.gpu):
        gpu = [int(g) for g in args.gpu.split(',')]
    else:
        gpu = 0
    print("Distributed: ", distr)
    print("Gpu: ", gpu)

    # training with multiple gpu-s
    if isinstance(gpu, list) and len(gpu) > 1:
        # check if training on a cluster or not
        distributed_backend = 'ddp' if distr else 'dp'
        replace_sampler_ddp = False if distr else True
    else:
        distributed_backend = None
        replace_sampler_ddp = True


    out_features = hidden if embedder else 2
    # todo parametrize this
    if model_name == 'dgcnn':
        nn = DGCNN(in_channels=3, hidden_features=hidden, out_features=out_features, k=k, transformer=False,
                               dropout=dropout, negative_slope=negative_slope, cosine=cosine)
    elif model_name == 'point_transformer':
        nn = PointTransformer(in_channels=3, out_channels=train_dataset.num_classes, dim_model=[32, 64, 128, 256, 512], k=16)
    elif model_name == 'pointnet2':
        nn = PointNet2(train_dataset.num_classes)
    elif model_name == 'euler':
        nn = EulerFeatExtract(in_channels=3, hidden_features=hidden, dropout=dropout, negative_slope=negative_slope)
    else:
        nn = MLP([3, hidden, hidden, hidden, hidden, out_features], dropout=dropout, negative_slope=negative_slope)

    nn_emb = EulerFeatExtract(in_channels=hidden, hidden_features=hidden, dropout=dropout, negative_slope=negative_slope) if embedder else None

    # nn_emb = MLP([hidden, hidden, 2], dropout=dropout, negative_slope=negative_slope) if embedder else None

    model = SimilarityHypHC(nn=nn,
                            embedder=nn_emb,
                            sim_distance=distance,
                            margin=margin,
                            temperature=temperature,
                            anneal=annealing,
                            anneal_step=anneal_step,
                            plot_every=plot_every)

    logger = MyTensorBoardLogger(logdir, name=dataname)
    model_params = {'dataset': dataname,
                    # 'ratio_labels': num_labels,
                    # 'min_noise': min_noise if dataname in ['moons', 'circles'] else None,
                    # 'max_noise': max_noise if dataname in ['moons', 'circles'] else None,
                    # 'cluster_std': cluster_std if dataname in ['blobs', 'varied', 'aniso'] else None,
                    # 'num_blobs': num_blobs if dataname in ['blobs', 'varied', 'aniso'] else '-1',
                    'model': model_name,
                    'embedder': 'True' if embedder else 'False',
                    'k': k if model_name == 'dgcnn' else -1,
                    'distance': distance,
                    'hidden': hidden,
                    'negative_slope': negative_slope,
                    'dropout': dropout,
                    'cosine': 'True' if cosine else 'False',
                    'margin': margin,
                    'temperature': temperature,
                    'annealing': annealing,
                    'anneal_step': anneal_step,
                    'max_epochs': epochs,
                    'batch': batch,
                    'lr': lr}

    print(model_params)

    metrics = {'ari@k': 0.0, 'ari@k-std': 0.0,
               'acc@k': 0.0, 'acc@k-std': 0.0,
               'purity@k': 0.0, 'purity@k-std': 0.0,
               'nmi@k': 0.0, 'nmi@k-std': 0.0,
               'ari': 0.0, 'ari-std': 0.0,
               'best_k': 0.0, 'std_k': 0.0}

    logger.log_hyperparams_metrics(params=model_params, metrics=metrics)
    savedir = os.path.join(logger.save_dir, logger.name, 'version_' + str(logger.version), 'checkpoints')
    # call backs for trainer
    checkpoint_callback = ModelCheckpoint(dirpath=savedir, verbose=True)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=patience,
        verbose=True,
        mode='min')


    trainer = pl.Trainer(gpus=gpu, max_epochs=epochs,
                         checkpoint_callback=checkpoint_callback,
                         callbacks=early_stop_callback,
                         logger=logger,
                         track_grad_norm=2,
                         distributed_backend=distributed_backend,
                         replace_sampler_ddp=replace_sampler_ddp)


    trainer.fit(model, train_loader, valid_loader)

    print("End Training")

    results = trainer.test(model, test_loader)
