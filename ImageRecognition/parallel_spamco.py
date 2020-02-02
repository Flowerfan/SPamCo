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
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description='soft_spaco')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-r', '--regularizer', type=str, default='hard')
parser.add_argument('-d', '--dataset', type=str, default='cifar10')
parser.add_argument('--gamma', type=float, default=0.3)
parser.add_argument('--iter-steps', type=int, default=7)
parser.add_argument('--num-per-class', type=int, default=400)
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed(args.seed)


def adjust_config(config, num_examples, iter_step):
  config.epochs = 300 - iter_step * 20
  config.step_size = max(int(config.epochs // 3), 1)
  return config


def train_predict(net, train_data, untrain_data, config, device, iteration,
                  pred_probs):
  config = adjust_config(config, len(train_data[0]), iteration)
  mu.train(net, train_data, config, device)
  pred_probs[device] = mu.predict_prob(net, untrain_data, config, device)
  save_checkpoint({
      'state_dict': net.state_dict(),
      'epoch': iteration,
  },
                  False,
                  fpath=os.path.join('logs/spaco/%s.epoch%d' %
                                     (config.model_name, iteration)))


def parallel_train(nets, train_datas, untrain_data, configs, iteration):
  processes = []
  manager = mp.Manager()
  pred_probs = manager.dict()
  for view, net in enumerate(nets):
    p = mp.Process(target=train_predict,
                   args=(net, train_datas[view], untrain_data, configs[view], view,
                         iteration, pred_probs))
    p.start()
    processes.append(p)
  for p in processes:
    p.join()
  return pred_probs.values()


def test_net(net, test_data, config, device, probs):
  probs[device] = mu.predict_prob(net, test_data, config, device)


def parallel_test(nets, test_data, configs):
  processes = []
  manager = mp.Manager()
  pred_probs = manager.dict()
  gt_y = test_data[1]
  for view, net in enumerate(nets):
    p = mp.Process(target=test_net,
                   args=(net, test_data, configs[view], view, pred_probs))
    p.start()
    processes.append(p)
  for p in processes:
    p.join()
  pred_probs = pred_probs.values()
  pred_y1 = np.argmax(pred_probs[0], axis=1)
  pred_y2 = np.argmax(pred_probs[1], axis=1)
  pred_y = np.argmax(sum(pred_probs), axis=1)
  print('view 1: %0.4f;   view 2: %0.4f;   fuse: %0.4f\n' %
        (np.mean(pred_y1 == gt_y), np.mean(pred_y2 == gt_y),
         np.mean(pred_y == gt_y)))


def spaco(configs,
          data,
          iter_steps=1,
          gamma=0,
          train_ratio=0.2,
          regularizer='hard'):
  """
    self-paced co-training model implementation based on Pytroch
    params:
    model_names: model names for spaco, such as ['resnet50','densenet121']
    data: dataset for spaco model
    save_pathts: save paths for two models
    iter_step: iteration round for spaco
    gamma: spaco hyperparameter
    train_ratio: initiate training dataset ratio
    """
  num_view = len(configs)
  train_data, untrain_data = dp.split_dataset(data['train'],
                                              seed=args.seed,
                                              num_per_class=args.num_per_class)
  add_num = 8000
  sel_ids = []
  weights = []
  start_step = 0
  ###########
  # initiate classifier to get preidctions
  ###########
  nets = [
      models.create(configs[view].model_name).to(view)
      for view in range(num_view)
  ]
  train_datas = [train_data, train_data]
  pred_probs = parallel_train(nets,
                              train_datas,
                              untrain_data,
                              configs,
                              iteration=0)
  pred_y = np.argmax(sum(pred_probs), axis=1)
  parallel_test(nets, data['test'], configs)
  # initiate weights for unlabled examples
  for view in range(num_view):
    sel_id, weight = dp.get_ids_weights(pred_probs[view], pred_y, train_data,
                                        add_num, gamma, regularizer)
    sel_ids.append(sel_id)
    weights.append(weight)

  # start iterative training
  for step in range(start_step, iter_steps):
    for view in range(num_view):
      print('Iter step: %d, view: %d, model name: %s' %
            (step + 1, view, configs[view].model_name))
      # update sample weights
      sel_ids[view], weights[view] = dp.update_ids_weights(
          view, pred_probs, sel_ids, weights, pred_y, train_data, add_num,
          gamma, regularizer)
      new_train_data, _ = dp.update_train_untrain(sel_ids[view], train_data,
                                                  untrain_data, pred_y,
                                                  weights[view])
      train_datas[view] = new_train_data
      nets[view] = models.create(configs[view].model_name).to(view)
    # update model parameter
    pred_probs = parallel_train(nets,
                                train_datas,
                                untrain_data,
                                configs,
                                iteration=step + 1)
    pred_y = np.argmax(sum(pred_probs), axis=1)
    parallel_test(nets, data['test'], configs)
    add_num += 8000


if __name__ == '__main__':
  mp.set_start_method('spawn')
  config1 = Config(model_name='shake_drop2', loss_name='weight_softmax')
  config2 = Config(model_name='wrn', loss_name='weight_softmax')

  dataset = args.dataset
  cur_path = os.getcwd()
  logs_dir = os.path.join(cur_path, 'logs')
  data_dir = os.path.join(cur_path, 'data', dataset)
  data = datasets.create(dataset, data_dir)

  spaco([config1, config2],
        data,
        gamma=args.gamma,
        iter_steps=args.iter_steps,
        regularizer=args.regularizer)
