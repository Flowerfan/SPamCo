import os
import torch
import argparse
import model_utils as mu
from util.data import data_process as dp
from config import Config
from util.serialization import load_checkpoint, save_checkpoint
import datasets
import models
import numpy as np
from copy import deepcopy

parser = argparse.ArgumentParser(description='Cotrain args')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-d', '--dataset', type=str, default='cifar10')
parser.add_argument('--iter-steps', type=int, default=6)
parser.add_argument('--num-per-class', type=int, default=400)
parser.add_argument('--tricks',
                    action='store_true',
                    help='draw with replacement')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed(args.seed)


def adjust_config(config, num_examples, iter_step):
  config.epochs = 300 - iter_step * 40
  #  config.epochs = 1
  config.step_size = max(int(config.epochs // 3), 1)
  return config


def cotrain(configs, data, iter_steps=1, train_ratio=0.2, device='cuda:0'):
  """
    cotrain model:
    params:
    model_names: model configs
    data: dataset include train and untrain data
    save_paths: paths for storing models
    iter_steps: maximum iteration steps
    train_ratio: labeled data ratio
    """
  assert iter_steps >= 1
  assert len(configs) == 2
  train_data, untrain_data = dp.split_dataset(data['train'],
                                              seed=args.seed,
                                              num_per_class=args.num_per_class)
  gt_y = data['test'][1]

  new_train_data = deepcopy(train_data)
  add_num = 8000
  for step in range(iter_steps):
    pred_probs = []
    test_preds = []
    add_ids = []
    for view in range(2):
      print('Iter step: %d, view: %d, model name: %s' %
            (step + 1, view, configs[view].model_name))
      configs[view] = adjust_config(configs[view], len(train_data[0]), step)
      net = models.create(configs[view].model_name).to(device)
      mu.train(net, new_train_data, configs[view], device)
      mu.evaluate(net, data['test'], configs[view], device)
      save_checkpoint({
          'state_dict': net.state_dict(),
          'epoch': step + 1,
      },
                      False,
                      fpath=os.path.join('logs/cotrain/%s.epoch%d' %
                                         (configs[view].model_name, step)))
      test_preds.append(
          mu.predict_prob(net, data['test'], configs[view], device))
      if len(untrain_data[0]) > configs[view].batch_size:
        pred_probs.append(
            mu.predict_prob(net, untrain_data, configs[view], device))
        add_ids.append(dp.select_ids(pred_probs[view], train_data, add_num))

    # update training data
    #  import pdb;pdb.set_trace()
    pred_y = np.argmax(sum(pred_probs), axis=1)
    add_id = np.array(sum(add_ids), dtype=np.bool)
    fuse_y = np.argmax(sum(test_preds), axis=1)
    print('Fuse Acc:%0.4f' % np.mean(fuse_y == gt_y))
    if args.tricks:
      new_train_data, _ = dp.update_train_untrain(add_id, train_data,
                                                  untrain_data, pred_y)
      add_num += add_num
    else:
      if len(untrain_data[0]) < 1:
          break
      new_train_data, untrain_data = dp.update_train_untrain(
          add_id, new_train_data, untrain_data, pred_y)


config1 = Config(model_name='shake_drop2', loss_name='weight_softmax')
config2 = Config(model_name='wrn', loss_name='weight_softmax')

dataset = args.dataset
cur_path = os.getcwd()
logs_dir = os.path.join(cur_path, 'logs')
data_dir = os.path.join(cur_path, 'data', dataset)
data = datasets.create(dataset, data_dir)
cotrain([config1, config2], data, args.iter_steps)
