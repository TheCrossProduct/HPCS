import warnings
from typing import Optional, Union
import argparse
import os
import multiprocessing as mp
import torch
import wandb
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from hpcs.models import PartNetHypHC, ShapeNetHypHC
from hpcs.utils.data import get_partnet_path, get_shapenet_path
from hpcs.data import ShapeNetDataset, PartNetDataset

def check_model_path(model_path) -> str:
  if os.path.exists(model_path):
    return model_path
  else:
    checkpoint_path = os.path.join(os.getcwd(), 'model.ckpt')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    wandb.restore('model.ckpt', run_path=model_path)
    return 'model.ckpt'

def init_pl_trainer(limit_test_batches: Optional[Union[int, float]] = None):
  accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
  trainer = pl.Trainer(accelerator=accelerator,
                       max_epochs=-1,
                       limit_test_batches=limit_test_batches,
                       logger=False)

  return trainer

def infer_shapenet(model_path : str, category: Optional[str] = None, num_points: int = 1024, batch_size: int = 1,
                   test_batches: Optional[Union[int, float]] = None, plot: bool = False):
  num_workers = mp.cpu_count()
  data_folder = get_shapenet_path()

  test_dataset = ShapeNetDataset(root=data_folder, npoints=num_points, split='test', class_choice=category)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

  model = ShapeNetHypHC.load_from_checkpoint(model_path, plot_inference=plot)
  tester = init_pl_trainer(limit_test_batches=test_batches)

  tester.test(model, test_loader)

def infer_partnet(model_path : str, category: str, level: int, num_points: int = 1024, batch_size: int = 1, plot: bool = False,
                  test_batches: Optional[Union[int, float]] = None):
  num_workers = mp.cpu_count()
  data_folder = get_partnet_path()
  if category.lower() not in model_path.lower():
    print(f"Be careful the chosen model {model_path} could not be adapted for this category: {category.title()}")
  path = os.path.realpath(os.path.join('..', data_folder, '%s-%d' % (category.title(), level), 'test_files.txt'))

  test_dataset = PartNetDataset(path, num_points)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

  model = PartNetHypHC.load_from_checkpoint(model_path, plot_inference=plot)
  tester = init_pl_trainer(limit_test_batches=test_batches)

  tester.test(model, test_loader)

def main():
  parser = argparse.ArgumentParser(description='Foo Bar')
  subparsers = parser.add_subparsers(dest='command', help='Dataset on which run inference', required=True)

  parser_shapenet = subparsers.add_parser('shapenet', help='Infer model on Shapenet dataset')
  parser_shapenet.set_defaults(func=infer_shapenet)

  parser_partnet = subparsers.add_parser('partnet', help='Infer model on Partnet dataset')
  parser_partnet.add_argument('--category', type=str, help='Category of object on which infer')
  parser_partnet.add_argument('--level', type=int, help='Level of segmentation on which test')
  parser_partnet.set_defaults(func=infer_partnet)

  for subparser in [parser_shapenet, parser_partnet]:
    subparser.add_argument('--num_points', default=1024, type=int, help='Number of points in each sample')
    subparser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    subparser.add_argument('--test_batches', default=None, type=float, help='Batch Size')
    subparser.add_argument('--model_path', help='Path to pretrained model to test')
    subparser.add_argument('--plot', action='store_true', help='Add this flag to plot results')

  args = parser.parse_args()
  args_ = vars(args).copy()
  args_.pop('command', None)
  args_.pop('func', None)
  args_['test_batches'] = int(args_['test_batches']) if \
    (args_['test_batches'] is not None and args_['test_batches'].is_integer()) else args_['test_batches']
  print(f"Calling {args.func.__name__}, with the folling arguments {args_}")
  args.func(**args_)

if __name__ == '__main__':
  main()