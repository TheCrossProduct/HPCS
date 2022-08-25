import os
import argparse
import os.path as osp
import yaml

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from hpcs.nn.models._hyp_hc import SimilarityHypHC
from hpcs.nn.models.encoders.dgcnn import DGCNN
from hpcs.nn.models.encoders.dgcnn2 import DGCNN2
from hpcs.nn.models.encoders.pointnet import PointNet
from hpcs.nn.models.encoders.vndgcnn_source import VNDGCNN


# def sweep():
#     with wandb.init():
#         config = wandb.config
#         model, trainer, train_loader, valid_loader, test_loader, savedir = configure(config)
#         train(model, trainer, train_loader, valid_loader, test_loader)


def configure(config):

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='logs', type=str, help='dirname for logs')
    parser.add_argument('--data', default=config.dataset, type=str, help='name of dataset to use')
    parser.add_argument('--model', default=config.model, type=str, help='model to use to extract features')
    parser.add_argument('--k', default=10, type=int, help='if model dgcnn, k is the number of neigh to take into account')
    parser.add_argument('--hidden', default=64, type=int, help='number of hidden features')
    parser.add_argument('--negative_slope', default=0.2, type=float, help='negative slope for leaky relu in the feature extractor')
    parser.add_argument('--dropout', default=config.dropout, type=float, help='dropout in the feature extractor')
    parser.add_argument('--cosine', help='if True add use cosine dist in DynamicEdgeConv', action='store_true')
    parser.add_argument('--distance', default='hyperbolic', type=str, help='distance to use to compute triplets')
    parser.add_argument('--margin', default=1.0, type=float, help='margin value to use in triplet loss')
    parser.add_argument('--temperature', default=0.05, type=float, help='rescale softmax value used in the hyphc loss')
    parser.add_argument('--annealing', default=1.0, type=float, help='annealing factor')
    parser.add_argument('--anneal_step', default=0, type=int, help='use annealing each n step')
    parser.add_argument('--batch', default=config.batch, type=int, help='batch size')
    parser.add_argument('--epochs', default=config.epochs, type=int, help='number of epochs')
    parser.add_argument('--lr', default=config.lr, type=float, help='learning rate')
    parser.add_argument('--patience', default=50, type=int, help='patience value for early stopping')
    parser.add_argument('--plot', default=-1, type=int, help='interval in which we plot prediction on validation batch')
    parser.add_argument('--gpu', default=config.gpu, type=str, help='use gpu')
    parser.add_argument('--distributed', help='if True run on a cluster machine', action='store_true')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--fixed_points', type=int, default=config.fixed_points)
    parser.add_argument('--min_scale', type=int, default=config.min_scale)
    parser.add_argument('--embedding', type=int, default=config.embedding)

    args = parser.parse_args()

    logdir = args.logdir
    dataname = args.data
    epochs = args.epochs
    model_name = args.model
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
    min_scale = args.min_scale
    embedding = args.embedding

    category = 'Airplane'  # Pass in `None` to train on all categories.
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ShapeNet')

    pre_transform, transform = T.NormalizeScale(), T.FixedPoints(fixed_points)
    train_dataset = ShapeNet(path, category, split='train', transform=transform, pre_transform=pre_transform)
    valid_dataset = ShapeNet(path, category, split='val', transform=transform, pre_transform=pre_transform)
    test_dataset = ShapeNet(path, category, split='test', transform=transform, pre_transform=pre_transform)
    test_dataset = test_dataset[0:20]

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    

    if len(args.gpu):
        gpu = [int(g) for g in args.gpu.split(',')]
    else:
        gpu = 0
    print("Distributed: ", distr)
    print("Gpu: ", gpu)


    out_features = embedding
    # todo parametrize this
    if model_name == 'dgcnn':
        nn = DGCNN(in_channels=3, hidden_features=hidden, out_features=out_features, k=k, transformer=False,
                   dropout=dropout, negative_slope=negative_slope, cosine=cosine)
    elif model_name == 'dgcnn2':
        nn = DGCNN2(in_channels=6, out_channels=out_features, k=k, dropout=dropout)
    elif model_name == 'pointnet':
        nn = PointNet(in_channels=3, out_features=out_features)
    elif model_name == 'vndgcnn':
        nn = VNDGCNN(in_channels=3, out_features=out_features, k=k, dropout=dropout)


    model = SimilarityHypHC(nn=nn,
                            min_scale=min_scale,
                            sim_distance=distance,
                            margin=margin,
                            temperature=temperature,
                            anneal=annealing,
                            anneal_step=anneal_step,
                            plot_every=plot_every)

    logger = WandbLogger(name=dataname, save_dir=os.path.join(logdir), project="HPCS", log_model=True)
    model_params = {'dataset': dataname,
                    'model': model_name,
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
                    'epochs': epochs,
                    'batch': batch,
                    'lr': lr,
                    'fixed_points': fixed_points,
                    'min_scale': min_scale}
    print(model_params)

    savedir = os.path.join(logger.save_dir, logger.name, 'version_' + str(logger.version), 'checkpoints')
    checkpoint_callback = ModelCheckpoint(dirpath=savedir, verbose=True)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=patience,
        verbose=True,
        mode='min')

    trainer = pl.Trainer(gpus=gpu,
                         max_epochs=epochs,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         logger=logger,
                         # limit_train_batches=100,
                         # limit_test_batches=100,
                         # track_grad_norm=2
                         )

    return model, trainer, train_loader, valid_loader, test_loader


def train(model, trainer, train_loader, valid_loader, test_loader):

    if os.path.exists('model.ckpt'):
        os.remove('model.ckpt')

    if config.resume:
        wandb.restore('model.ckpt', root=os.getcwd(), run_path='pierreoo/HPCS/runs/xj3d1dc9')
        model = model.load_from_checkpoint('model.ckpt')

    trainer.fit(model, train_loader, valid_loader)

    if os.path.exists('model.ckpt'):
        os.remove('model.ckpt')

    print("End Training")

    trainer.save_checkpoint('model.ckpt')
    wandb.save('model.ckpt')

    trainer.test(model, test_loader)



if __name__ == "__main__":
    # with open(r'sweeps/sweep.yaml') as file:
    #     sweep_config = yaml.load(file, Loader=yaml.FullLoader)
    #
    # sweep_id = wandb.sweep(sweep_config, project="HPCS")
    # # sweep_id = 'v7hcnyap'
    # wandb.agent(sweep_id, function=sweep, count=1, project="HPCS")

    config = dict(
        batch=2,
        epochs=1,
        lr=0.0001,
        dropout=0.0,
        fixed_points=400,
        min_scale=0.1,
        embedding=1000,
        model="dgcnn",
        dataset="shapenet",
        gpu="0",
        resume=False,
    )

    wandb.init(project='HPCS', config=config)
    config = wandb.config
    model, trainer, train_loader, valid_loader, test_loader = configure(config)
    train(model, trainer, train_loader, valid_loader, test_loader)