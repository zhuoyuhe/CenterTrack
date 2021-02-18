from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os

import torch
import torch.utils.data
from opts import opts
from model.model import create_model, load_model, save_model
from model.data_parallel import DataParallel
from logger import Logger
from dataset.dataset_factory import get_dataset
from trainer import Trainer
from main import get_optimizer



if __name__ == '__main__':
  opt = opts().parse()
  torch.manual_seed(opt.seed)
  Dataset = get_dataset(opt.dataset)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  path_1 = '/mnt/3dvision-cpfs/zhuoyu/CenterTrack/exp/ddd/nu_3d_det_uni/model_last.pth'
  path_2 = '/mnt/3dvision-cpfs/zhuoyu/CenterTrack/exp/ddd/nu_3d_det_fix_param/model_last.pth'

  model_1 = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
  model_2 = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
  optimizer = get_optimizer(opt, model_1)

  model_1, _, _ = load_model(
    model_1, path_1, opt, optimizer)

  model_2, _, _ = load_model(
    model_2, path_2, opt, optimizer)

  for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
    if p1.data.ne(p2.data).sum() > 0:
      print(False)
    else:
      print(True)