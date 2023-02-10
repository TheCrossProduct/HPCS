import os
import argparse
import os.path as osp

import torch
from collections import OrderedDict
from torch.utils.data import DataLoader
from hpcs.data import PartNetDataset, ShapeNetDataset
from hpcs.data.hierarchy_list import get_hierarchy_list

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from hpcs.models import ShapeNetHypHC, PartNetHypHC
from hpcs.nn.dgcnn import DGCNN_partseg, VN_DGCNN_partseg

from hpcs.nn.pointnet import POINTNET_partseg, VN_POINTNET_partseg
from hpcs.nn.hyperbolic import ExpMap


def read_configutation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='logs', type=str, help='dirname for logs')
    parser.add_argument('--dataset', '-dataset', default='shapenet', type=str, help='name of dataset to use')
    parser.add_argument('--category', '-category', default='Airplane', type=str, help='category from dataset')
    parser.add_argument('--level', '-level', default=3, type=int, help='granularity level of partnet object')
    parser.add_argument('--fixed_points', '-fixed_points', default=256, type=int, help='points retained from point cloud')
    parser.add_argument('--model', '-model', default='vn_dgcnn_partseg', type=str, help='model to use to extract features')
    parser.add_argument('--train_rotation', '-train_rotation', default='so3', type=str, help='type of rotation augmentation for train')
    parser.add_argument('--test_rotation', '-test_rotation', default='so3', type=str, help='type of rotation augmentation for test')
    parser.add_argument('--embedding', '-embedding', default=6, type=int, help='dimension of poincare space')
    parser.add_argument('--k', '-k', default=10, type=int, help='if model dgcnn, k is the number of neigh to take into account')
    parser.add_argument('--margin', '-margin', default=0.05, type=float, help='margin value to use in miner loss')
    parser.add_argument('--t_per_anchor', '-t_per_anchor', default=50, type=int, help='margin value to use in miner loss')
    parser.add_argument('--fraction', '-fraction', default=1.2, type=float, help='number of triplets for underrepresented classes')
    parser.add_argument('--temperature', '-temperature', default=1, type=float, help='rescale softmax value used in the hyphc loss')
    parser.add_argument('--epochs', '-epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch', '-batch', default=6, type=int, help='batch size')
    parser.add_argument('--lr', '-lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('--accelerator', '-accelerator', default='gpu', type=str, help='use gpu')
    parser.add_argument('--num_workers', '-num_workers', default=10, type=int, help='number of workers')
    parser.add_argument('--dropout', '-dropout', default=0.5, type=float, help='dropout in the feature extractor')
    parser.add_argument('--anneal_factor', '-anneal_factor', default=2, type=float, help='annealing factor')
    parser.add_argument('--anneal_step', '-anneal_step', default=0, type=int, help='use annealing each n step')
    parser.add_argument('--patience', '-patience', default=50, type=int, help='patience value for early stopping')
    parser.add_argument('--trade_off', '-trade_off', default=0.5, type=float, help='control trade-off between two losses')
    parser.add_argument('--miner', '-miner', default=True, type=bool, help='triplet miner for hyperbolic loss')
    parser.add_argument('--cosface', '-cosface', default=True, type=bool, help='cosface / triplet loss')
    parser.add_argument('--class_vector', '-class_vector', default=False, type=bool, help='class vector to decode')
    parser.add_argument('--hierarchical', '-hierarchical', default=True, type=bool, help='hierarchical loss')
    parser.add_argument('--hierarchy_list', '-hierarchy_list', default=[], type=list, help='precomputed hierarchy list')
    parser.add_argument('--plot_inference', '-plot_inference', default=False, type=bool, help='plot visualizations during testing')
    parser.add_argument('--pretrained', '-pretrained', default=False, type=bool, help='load pretrained model')
    parser.add_argument('--resume', '-resume', default=False, type=bool, help='resume training on model')
    parser.add_argument('--wandb', '-wandb', default='online', type=str, help='Online/Offline WandB mode (Useful in JeanZay)')
    args = parser.parse_args()
    return args


def configure_feature_extractor(model_name, embedding_size, num_class, k, dropout, pretrained):
    out_features = embedding_size
    if model_name == 'dgcnn_partseg':
        nn = DGCNN_partseg(in_channels=3, out_features=out_features, k=k, dropout=dropout, num_class=num_class)
    elif model_name == 'vn_dgcnn_partseg':
        nn = VN_DGCNN_partseg(in_channels=3, out_features=out_features, k=k, dropout=dropout, pooling='mean', num_class=num_class)
    elif model_name == 'pointnet_partseg':
        nn = POINTNET_partseg(num_part=out_features, normal_channel=False)
    elif model_name == 'vn_pointnet_partseg':
        nn = VN_POINTNET_partseg(num_part=out_features, normal_channel=True, k=k, pooling='mean')
    else:
        raise ValueError(f"Not implemented for model_name {model_name}")

    if pretrained:
        model_path = osp.realpath('model.partseg.vn_dgcnn.aligned.t7')
        checkpoint = torch.load(model_path)
        new_state_dict = OrderedDict()
        for key, value in checkpoint.items():
            name = key.replace('module.', '')
            new_state_dict[name] = value
        nn.load_state_dict(new_state_dict, strict=False)
    return nn


def configure_embedder(embedder_name, input_features, output_features, radius):
    nn = ExpMap()
    return nn


def configure(args):
    wandb.config.update(args)
    log = args.log
    dataset = args.dataset
    category = args.category
    level = args.level
    fixed_points = args.fixed_points
    model_name = args.model
    train_rotation = args.train_rotation
    test_rotation = args.test_rotation
    embedding = args.embedding
    k = args.k
    margin = args.margin
    t_per_anchor = args.t_per_anchor
    fraction = args.fraction
    temperature = args.temperature
    epochs = args.epochs
    batch = args.batch
    lr = args.lr
    accelerator = args.accelerator
    num_workers = args.num_workers
    dropout = args.dropout
    anneal_factor = args.anneal_factor
    anneal_step = args.anneal_step
    patience = args.patience
    trade_off = args.trade_off
    miner = args.miner
    cosface = args.cosface
    class_vector = args.class_vector
    hierarchical = args.hierarchical
    hierarchy_list = args.hierarchy_list
    plot_inference = args.plot_inference
    pretrained = args.pretrained
    resume = args.resume
    wandb_mode = args.wandb

    if dataset == 'shapenet':
        data_folder = 'data/ShapeNet/raw'
        # data_folder = '/gpfsscratch/rech/qpj/uyn98cq/ShapeNet/raw'
        train_dataset = ShapeNetDataset(root=data_folder, npoints=fixed_points, split='train', class_choice=category)
        valid_dataset = ShapeNetDataset(root=data_folder, npoints=fixed_points, split='val', class_choice=category)
        test_dataset = ShapeNetDataset(root=data_folder, npoints=fixed_points, split='test', class_choice=category)

        if class_vector:
            num_class = len(train_dataset.seg_classes[category])
        else:
            num_class = 16

    elif dataset == 'partnet':
        data_folder = 'data/PartNet/sem_seg_h5/'

        if hierarchical:
            levels = []
            for i in range(3):
                list_train = os.path.join(data_folder, '%s-%d' % (category, i+1), 'train_files.txt')
                if os.path.exists(list_train):
                    levels.append(i+1)
            level = levels[-1]
            hierarchy_list = get_hierarchy_list(category, levels)

        list_train = os.path.join(data_folder, '%s-%d' % (category, level), 'train_files.txt')
        list_val = os.path.join(data_folder, '%s-%d' % (category, level), 'val_files.txt')
        list_test = os.path.join(data_folder, '%s-%d' % (category, level), 'test_files.txt')

        train_dataset = PartNetDataset(list_train, fixed_points)
        valid_dataset = PartNetDataset(list_val, fixed_points)
        test_dataset = PartNetDataset(list_test, fixed_points)

        with open('data/PartNet/after_merging_label_ids/%s-level-%d.txt' % (category, level), 'r') as fin:
            num_class = len(fin.readlines()) + 1
            print('Number of Classes: %d' % num_class)

    else:
        raise KeyError(f"Not available implementation for dataset: {dataset}")

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch, shuffle=False, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=num_workers, drop_last=True)

    nn_feat = configure_feature_extractor(model_name=model_name,
                                          embedding_size=embedding,
                                          num_class=num_class, k=k,
                                          dropout=dropout,
                                          pretrained=pretrained)
    nn_emb = ExpMap()

    if dataset == 'shapenet':
        model = ShapeNetHypHC(nn_feat=nn_feat,
                              nn_emb=nn_emb,
                              train_rotation=train_rotation,
                              test_rotation=test_rotation,
                              lr=lr,
                              embedding=embedding,
                              margin=margin,
                              t_per_anchor=t_per_anchor,
                              fraction=fraction,
                              temperature=temperature,
                              anneal_factor=anneal_factor,
                              anneal_step=anneal_step,
                              num_class=num_class,
                              class_vector=class_vector,
                              trade_off=trade_off,
                              miner=miner,
                              cosface=cosface,
                              hierarchical=hierarchical,
                              hierarchy_list=hierarchy_list,
                              plot_inference=plot_inference)
    elif dataset == 'partnet':
        model = PartNetHypHC(nn_feat=nn_feat,
                             nn_emb=nn_emb,
                             train_rotation=train_rotation,
                             test_rotation=test_rotation,
                             lr=lr,
                             embedding=embedding,
                             margin=margin,
                             t_per_anchor=t_per_anchor,
                             fraction=fraction,
                             temperature=temperature,
                             anneal_factor=anneal_factor,
                             anneal_step=anneal_step,
                             num_class=num_class,
                             class_vector=class_vector,
                             trade_off=trade_off,
                             miner=miner,
                             cosface=cosface,
                             hierarchical=hierarchical,
                             hierarchy_list=hierarchy_list,
                             plot_inference=plot_inference)
    else:
        raise KeyError(f"Not available implementation for dataset: {dataset}")

    logger = WandbLogger(name=dataset, save_dir=os.path.join(log), project='HPCS', log_model=True)
    model_params = {'dataset': dataset,
                    'category': category,
                    'level': level if dataset == 'partnet' else 'coarse',
                    'fixed_points': fixed_points,
                    'model': model_name,
                    'embedding': embedding,
                    'k': k,
                    'margin': margin,
                    't_per_anchor': t_per_anchor,
                    'fraction': fraction,
                    'temperature': temperature,
                    'epochs': epochs,
                    'batch': batch,
                    'lr': lr,
                    'accelerator': accelerator}
    print(model_params)

    savedir = os.path.join(logger.save_dir, logger.name, 'version_' + str(logger.version), 'checkpoints')
    checkpoint_callback = ModelCheckpoint(dirpath=savedir, verbose=True)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=patience,
        verbose=True,
        mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(accelerator=accelerator,
                         max_epochs=epochs,
                         callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
                         logger=logger,
                         limit_test_batches=10
                         )

    return model, trainer, train_loader, valid_loader, test_loader, resume, wandb_mode


def train(model, trainer, train_loader, valid_loader, test_loader, resume):
    if os.path.exists('model.ckpt'):
        os.remove('model.ckpt')

    if resume:
        wandb.restore('model.ckpt', root=os.getcwd(), run_path='princepi/HPCS/runs/1zvcmsdj')
        model = model.load_from_checkpoint('model.ckpt')

    trainer.fit(model, train_loader, valid_loader)

    print("End Training")

    trainer.save_checkpoint('model.ckpt')
    wandb.save('model.ckpt')

    trainer.test(model, test_loader)


if __name__ == "__main__":
    args = read_configutation()

    wandb.init(project='HPCS', mode=args.wandb, config=args)
    model, trainer, train_loader, valid_loader, test_loader, resume, wandb_mode = configure(args)
    train(model, trainer, train_loader, valid_loader, test_loader, resume)
