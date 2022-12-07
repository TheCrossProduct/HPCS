import argparse

import pytorch_lightning as pl

from hpcs.hyp_hc import SimilarityHypHC
from hpcs.nn.dgcnn import DGCNN_partseg
from hpcs.nn.dgcnn import VN_DGCNN_partseg
from hpcs.nn.pointnet import POINTNET_partseg
from hpcs.nn.pointnet import VN_POINTNET_partseg


def configure(config):

    dataset = config['dataset']['value']
    model_name = config['model_name']['value']
    train_rotation = config['train_rotation']['value']
    test_rotation = config['test_rotation']['value']
    embedding = config['embedding']['value']
    k = config['k']['value']
    margin = config['margin']['value']
    t_per_anchor = config['t_per_anchor']['value']
    temperature = config['temperature']['value']
    lr = config['lr']['value']
    accelerator = 'cpu'
    anneal_factor = config['anneal_factor']['value']
    anneal_step = config['anneal_step']['value']
    num_class = config['num_class']['value']


    out_features = embedding
    if model_name == 'dgcnn_partseg':
        nn = DGCNN_partseg(in_channels=3, out_features=out_features, k=k, dropout=0.5, num_class=num_class)
    elif model_name == 'vn_dgcnn_partseg':
        nn = VN_DGCNN_partseg(in_channels=3, out_features=out_features, k=k, dropout=0.5, pooling='mean', num_class=num_class)
    elif model_name == 'pointnet_partseg':
        nn = POINTNET_partseg(num_part=out_features, normal_channel=False)
    elif model_name == 'vn_pointnet_partseg':
        nn = VN_POINTNET_partseg(num_part=out_features, normal_channel=True, k=k, pooling='mean')

    model = SimilarityHypHC(nn=nn,
                            train_rotation=train_rotation,
                            test_rotation=test_rotation,
                            dataset=dataset,
                            lr=lr,
                            embedding=embedding,
                            margin=margin,
                            t_per_anchor=t_per_anchor,
                            temperature=temperature,
                            anneal_factor=anneal_factor,
                            anneal_step=anneal_step,
                            num_class=num_class
                            )

    trainer = pl.Trainer(accelerator=accelerator,
                         max_epochs=-1,
                         limit_test_batches=10
                         )

    return model, trainer