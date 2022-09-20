import os
import argparse
import os.path as osp
import yaml

import torch
from collections import OrderedDict
from torch.utils.data import DataLoader
from data.ShapeNet.ShapeNetDataLoader import PartNormalDataset

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from hpcs.nn._hyp_hc import SimilarityHypHC
from hpcs.nn.dgcnn import DGCNN_simple
from hpcs.nn.dgcnn import DGCNN_partseg
from hpcs.nn.dgcnn import VN_DGCNN_partseg
from hpcs.nn.dgcnn import VN_DGCNN_partseg_encoder


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
    parser.add_argument('--k', default=40, type=int, help='if model dgcnn, k is the number of neigh to take into account')
    parser.add_argument('--hidden', default=64, type=int, help='number of hidden features')
    parser.add_argument('--negative_slope', default=0.2, type=float, help='negative slope for leaky relu in the feature extractor')
    parser.add_argument('--dropout', default=config.dropout, type=float, help='dropout in the feature extractor')
    parser.add_argument('--cosine', help='if True add use cosine dist in DynamicEdgeConv', action='store_true')
    parser.add_argument('--distance', default='cosine', type=str, help='distance to use to compute triplets')
    parser.add_argument('--margin', default=1.0, type=float, help='margin value to use in triplet loss')
    parser.add_argument('--temperature', default=0.01, type=float, help='rescale softmax value used in the hyphc loss')
    parser.add_argument('--annealing', default=1.0, type=float, help='annealing factor')
    parser.add_argument('--anneal_step', default=20, type=int, help='use annealing each n step')
    parser.add_argument('--batch', default=config.batch, type=int, help='batch size')
    parser.add_argument('--epochs', default=config.epochs, type=int, help='number of epochs')
    parser.add_argument('--lr', default=config.lr, type=float, help='learning rate')
    parser.add_argument('--patience', default=50, type=int, help='patience value for early stopping')
    parser.add_argument('--gpu', default=config.gpu, type=str, help='use gpu')
    parser.add_argument('--distributed', help='if True run on a cluster machine', action='store_true')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--fixed_points', type=int, default=config.fixed_points)
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
    distr = args.distributed
    num_workers = args.num_workers
    fixed_points = args.fixed_points
    embedding = args.embedding


    category = 'Airplane'
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ShapeNet/raw')

    train_dataset = PartNormalDataset(root=path, npoints=fixed_points, split='train', class_choice=category)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=num_workers)
    valid_dataset = PartNormalDataset(root=path, npoints=fixed_points, split='val', class_choice=category)
    valid_loader = DataLoader(valid_dataset, batch_size=batch, shuffle=False, num_workers=num_workers)
    test_dataset = PartNormalDataset(root=path, npoints=fixed_points, split='test', class_choice=category)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)


    if len(args.gpu):
        gpu = [int(g) for g in args.gpu.split(',')]
    else:
        gpu = 0
    print("Distributed: ", distr)
    print("Gpu: ", gpu)


    out_features = embedding
    if model_name == 'dgcnn_simple':
        nn = DGCNN_simple(in_channels=3, out_features=out_features, k=k, dropout=dropout)
    elif model_name == 'dgcnn_partseg':
        nn = DGCNN_partseg(in_channels=3, out_features=out_features, k=k, dropout=dropout)
    elif model_name == 'vn_dgcnn_partseg':
        nn = VN_DGCNN_partseg(in_channels=3, out_features=out_features, k=k, dropout=dropout, pooling='mean')
    elif model_name == 'vn_dgcnn_partseg_encoder':
        nn = VN_DGCNN_partseg_encoder(in_channels=3, out_features=out_features, k=k, dropout=dropout, pooling='mean')

    if config.pretrained:
        model_path = osp.realpath('model.partseg.vn_dgcnn.aligned.t7')
        checkpoint = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        nn.load_state_dict(new_state_dict, strict=False)


    model = SimilarityHypHC(nn=nn,
                            sim_distance=distance,
                            margin=margin,
                            temperature=temperature,
                            anneal=annealing,
                            anneal_step=anneal_step)


    logger = WandbLogger(name=dataname, save_dir=os.path.join(logdir), project="HPCS", log_model=True)
    model_params = {'dataset': dataname,
                    'model': model_name,
                    'k': k,
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
                    'fixed_points': fixed_points}
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
                         # limit_train_batches=500,
                         limit_test_batches=25
                         )

    return model, trainer, train_loader, valid_loader, test_loader


def train(model, trainer, train_loader, valid_loader, test_loader):

    if os.path.exists('model.ckpt'):
        os.remove('model.ckpt')

    if config.resume:
        wandb.restore('model.ckpt', root=os.getcwd(), run_path='pierreoo/HPCS/runs/ckz23kji')
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
        batch=6,
        epochs=15,
        lr=0.0001,
        dropout=0.0,
        fixed_points=1024,
        embedding=4,
        model="vn_dgcnn_partseg",
        dataset="shapenet",
        gpu="0",
        resume=False,
        pretrained=True,
    )

    wandb.init(project='HPCS', config=config)
    config = wandb.config
    model, trainer, train_loader, valid_loader, test_loader = configure(config)
    train(model, trainer, train_loader, valid_loader, test_loader)