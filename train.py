import os
import argparse
import os.path as osp

import torch
from collections import OrderedDict
from torch.utils.data import DataLoader
from data.ShapeNet.ShapeNetDataLoader import PartNormalDataset
from data.PartNet.PartNetDataLoader import H5Dataset
from data.PartNet.Hierarchical import H5Dataset_hierarchical

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from hpcs.hyp_hc import SimilarityHypHC
from hpcs.nn.dgcnn import DGCNN_partseg
from hpcs.nn.dgcnn import VN_DGCNN_partseg
from hpcs.nn.dgcnn import VN_DGCNN_expo
from hpcs.nn.pointnet import POINTNET_partseg
from hpcs.nn.pointnet import VN_POINTNET_partseg

def get_wandb_mode():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb','-wandb',default='online',type=str, help='Online/Offline WandB mode (Useful in JeanZay)')
    args = parser.parse_args()
    return args.wandb

def configure():
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
    parser.add_argument('--normalize', '-normalize', default=True, type=bool, help='normalize hyperbolic space')
    parser.add_argument('--class_vector', '-class_vector', default=False, type=bool, help='class vector to decode')
    parser.add_argument('--trade_off', '-trade_off', default=0.5, type=float, help='control trade-off between two losses')
    parser.add_argument('--hierarchical', '-hierarchical', default=False, type=bool, help='hierarchical loss')
    parser.add_argument('--pretrained', '-pretrained', default=False, type=bool, help='load pretrained model')
    parser.add_argument('--resume', '-resume', default=False, type=bool, help='resume training on model')
    parser.add_argument('--wandb','-wandb',default='online',type=str, help='Online/Offline WandB mode (Useful in JeanZay)')
    args = parser.parse_args()

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
    normalize = args.normalize
    class_vector = args.class_vector
    trade_off = args.trade_off
    hierarchical = args.hierarchical
    pretrained = args.pretrained
    resume = args.resume
    wandb_mode=args.wandb


    if dataset == 'shapenet':
        data_folder = 'data/ShapeNet/raw'
        train_dataset = PartNormalDataset(root=data_folder, npoints=fixed_points, split='train', class_choice=category)
        valid_dataset = PartNormalDataset(root=data_folder, npoints=fixed_points, split='val', class_choice=category)
        test_dataset = PartNormalDataset(root=data_folder, npoints=fixed_points, split='test', class_choice=category)

        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=num_workers, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch, shuffle=False, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=num_workers, drop_last=True)

        if class_vector:
            num_class = len(train_dataset.seg_classes[category])
        else:
            num_class = 16

    elif dataset == 'partnet':
        data_folder = 'data/PartNet/sem_seg_h5/'

        if hierarchical:
            train_paths = []
            val_paths = []
            test_paths = []
            for i in range(3):
                list_train = os.path.join(data_folder, '%s-%d' % (category, i+1), 'train_files.txt')
                list_val = os.path.join(data_folder, '%s-%d' % (category, i+1), 'val_files.txt')
                list_test = os.path.join(data_folder, '%s-%d' % (category, i+1), 'test_files.txt')
                train_paths.append(list_train)
                val_paths.append(list_val)
                test_paths.append(list_test)

            train_dataset = H5Dataset_hierarchical(train_paths, fixed_points, i+1)
            val_dataset = H5Dataset_hierarchical(val_paths, fixed_points, i+1)
            test_dataset = H5Dataset_hierarchical(test_paths, fixed_points, i+1)

            with open('data/PartNet/after_merging_label_ids/%s-level-%d.txt' % (category, 3), 'r') as fin:
                num_class = len(fin.readlines()) + 1
                print('Number of Classes: %d' % num_class)
            if not class_vector:
                num_class = 16
        else:
            list_train = os.path.join(data_folder, '%s-%d' % (category, level), 'train_files.txt')
            list_val = os.path.join(data_folder, '%s-%d' % (category, level), 'val_files.txt')
            list_test = os.path.join(data_folder, '%s-%d' % (category, level), 'test_files.txt')

            train_dataset = H5Dataset(list_train, fixed_points)
            val_dataset = H5Dataset(list_val, fixed_points)
            test_dataset = H5Dataset(list_test, fixed_points)

            with open('data/PartNet/after_merging_label_ids/%s-level-%d.txt' % (category, level), 'r') as fin:
                num_class = len(fin.readlines()) + 1
                print('Number of Classes: %d' % num_class)
            if not class_vector:
                num_class = 16



        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=num_workers, drop_last=True)
        valid_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=num_workers, drop_last=True)

    out_features = embedding
    if model_name == 'dgcnn_partseg':
        nn = DGCNN_partseg(in_channels=3, out_features=out_features, k=k, dropout=dropout, num_class=num_class)
    elif model_name == 'vn_dgcnn_partseg':
        nn = VN_DGCNN_partseg(in_channels=3, out_features=out_features, k=k, dropout=dropout, pooling='mean', num_class=num_class)
    elif model_name == 'vn_dgcnn_expo':
        nn = VN_DGCNN_expo(in_channels=3, out_features=out_features, k=k, dropout=dropout, pooling='mean', num_class=num_class)
    elif model_name == 'pointnet_partseg':
        nn = POINTNET_partseg(num_part=out_features, normal_channel=False)
    elif model_name == 'vn_pointnet_partseg':
        nn = VN_POINTNET_partseg(num_part=out_features, normal_channel=True, k=k, pooling='mean')

    if pretrained:
        model_path = osp.realpath('model.partseg.vn_dgcnn.aligned.t7')
        checkpoint = torch.load(model_path)
        new_state_dict = OrderedDict()
        for key, value in checkpoint.items():
            name = key.replace('module.', '')
            new_state_dict[name] = value
        nn.load_state_dict(new_state_dict, strict=False)


    model = SimilarityHypHC(nn=nn,
                            model_name=model_name,
                            train_rotation=train_rotation,
                            test_rotation=test_rotation,
                            dataset=dataset,
                            lr=lr,
                            embedding=embedding,
                            k=k,
                            margin=margin,
                            t_per_anchor=t_per_anchor,
                            fraction=fraction,
                            temperature=temperature,
                            anneal_factor=anneal_factor,
                            anneal_step=anneal_step,
                            num_class=num_class,
                            normalize=normalize,
                            class_vector=class_vector,
                            trade_off=trade_off,
                            hierarchical=hierarchical
                            )


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

    trainer = pl.Trainer(accelerator=accelerator,
                         max_epochs=epochs,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         logger=logger,
                         limit_test_batches=10
                         )

    return model, trainer, train_loader, valid_loader, test_loader, resume,wandb_mode


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
    wandb.init(project='HPCS',mode=get_wandb_mode())
    model, trainer, train_loader, valid_loader, test_loader, resume, wandb_mode = configure()
    train(model, trainer, train_loader, valid_loader, test_loader, resume)
