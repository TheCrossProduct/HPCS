import argparse

import pytorch_lightning as pl

from hpcs.nn.models._hyp_hc import SimilarityHypHC
from hpcs.nn.models.encoders.dgcnn2 import DGCNN2
from hpcs.nn.models.encoders.euler import EulerFeatExtract
from hpcs.nn.models.encoders.point_transformer import PointTransformer
from hpcs.nn.models.encoders.pointnet2 import PointNet2


def configure(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='logs', type=str, help='dirname for logs')
    parser.add_argument('--data', default='shapenet', type=str, help='name of dataset to use')
    parser.add_argument('--model', default='dgcnn2', type=str, help='model to use to extract features')
    parser.add_argument('--embedder', help='if True add a an embedding model from the feature space to B2',
                        action='store_true')
    parser.add_argument('--k', default=10, type=int,
                        help='if model dgcnn, k is the number of neigh to take into account')
    parser.add_argument('--hidden', default=64, type=int, help='number of hidden features')
    parser.add_argument('--negative_slope', default=0.2, type=float,
                        help='negative slope for leaky relu in the feature extractor')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout in the feature extractor')
    parser.add_argument('--cosine', help='if True add use cosine dist in DynamicEdgeConv', action='store_true')
    parser.add_argument('--distance', default='cosine', type=str, help='distance to use to compute triplets')
    parser.add_argument('--margin', default=1.0, type=float, help='margin value to use in triplet loss')
    parser.add_argument('--temperature', default=0.05, type=float, help='rescale softmax value used in the hyphc loss')
    parser.add_argument('--annealing', default=1.0, type=float, help='annealing factor')
    parser.add_argument('--anneal_step', default=0, type=int, help='use annealing each n step')
    parser.add_argument('--batch', default=config['batch']['value'], type=int, help='batch size')
    parser.add_argument('--epochs', default=config['epochs']['value'], type=int, help='number of epochs')
    parser.add_argument('--lr', default=config['lr']['value'], type=float, help='learning rate')
    parser.add_argument('--patience', default=50, type=int, help='patience value for early stopping')
    parser.add_argument('--plot', default=-1, type=int, help='interval in which we plot prediction on validation batch')
    parser.add_argument('--gpu', default="", type=str, help='use gpu')
    parser.add_argument('--distributed', help='if True run on a cluster machine', action='store_true')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--fixed_points', type=int, default=config['fixed_points']['value'])

    args, unknown = parser.parse_known_args()

    logdir = args.logdir
    dataname = args.data
    epochs = args.epochs
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

    if len(args.gpu):
        gpu = [int(g) for g in args.gpu.split(',')]
    else:
        gpu = 0
    print("Distributed: ", distr)
    print("Gpu: ", gpu)


    out_features = 2
    # todo parametrize this
    if model_name == 'dgcnn2':
        nn = DGCNN2(in_channels=6, out_channels=out_features, k=30)
    elif model_name == 'point_transformer':
        nn = PointTransformer(in_channels=3, out_channels=out_features, dim_model=[32, 64, 128, 256, 512], k=30)
    elif model_name == 'pointnet2':
        nn = PointNet2(in_channels=6, out_channels=out_features)

    nn_emb = EulerFeatExtract(in_channels=hidden, hidden_features=hidden, dropout=dropout,
                              negative_slope=negative_slope) if embedder else None

    model = SimilarityHypHC(nn=nn,
                            embedder=nn_emb,
                            sim_distance=distance,
                            margin=margin,
                            temperature=temperature,
                            anneal=annealing,
                            anneal_step=anneal_step,
                            plot_every=plot_every)


    trainer = pl.Trainer(gpus=gpu,
                         max_epochs=epochs,
                         limit_test_batches=2,
                         )

    return model, trainer