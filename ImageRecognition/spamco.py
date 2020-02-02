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
parser.add_argument('--iter-steps', type=int, default=5)
parser.add_argument('--num-per-class', type=int, default=400)
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


def train_predict(net, train_data, untrain_data, test_data, config, device, pred_probs):
    mu.train(net, train_data, config, device)
    pred_probs.append(mu.predict_prob(net, untrain_data, configs[view], view))


def parallel_train(nets, train_data, data_dir, configs):
    processes = []
    for view, net in enumerate(nets):
        p = mp.Process(target=mu.train, args=(net, train_data, config, view))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()




def adjust_config(config, num_examples, iter_step):
    repeat = 20 * (1.1 ** iter_step)
    #  epochs = list(range(300, 20, -20))
    #  config.epochs = epochs[iter_step]
    #  config.epochs = int((50000 * repeat) // num_examples)
    config.epochs = 200
    config.step_size = max(int(config.epochs // 3), 1)
    return config

def spaco(configs,
          data,
          iter_steps=1,
          gamma=0,
          train_ratio=0.2,
          regularizer='soft'):
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
    train_data, untrain_data = dp.split_dataset(
      data['train'], seed=args.seed, num_per_class=args.num_per_class)
    add_num = 4000
    pred_probs = []
    test_preds = []
    sel_ids = []
    weights = []
    start_step = 0
    ###########
    # initiate classifier to get preidctions
    ###########
    for view in range(num_view):
        configs[view] = adjust_config(configs[view], len(train_data[0]), 0)
        net = models.create(configs[view].model_name).to(view)
        mu.train(net, train_data, configs[view], device=view)
        pred_probs.append(mu.predict_prob(net, untrain_data, configs[view], view))
        test_preds.append(mu.predict_prob(net, data['test'], configs[view], view))
        acc = mu.evaluate(net, data['test'], configs[view], view)
        save_checkpoint(
          {
            'state_dict': net.state_dict(),
            'epoch': 0,
          },
          False,
          fpath=os.path.join(
            'spaco/%s.epoch%d' % (configs[view].model_name, 0)))
    pred_y = np.argmax(sum(pred_probs), axis=1)

    # initiate weights for unlabled examples
    for view in range(num_view):
        sel_id, weight = dp.get_ids_weights(pred_probs[view], pred_y,
                                            train_data, add_num, gamma,
                                            regularizer)
        import pdb;pdb.set_trace()
        sel_ids.append(sel_id)
        weights.append(weight)

    # start iterative training
    gt_y = data['test'][1]
    for step in range(start_step, iter_steps):
        for view in range(num_view):
            print('Iter step: %d, view: %d, model name: %s' % (step+1,view,configs[view].model_name))

            # update sample weights
            sel_ids[view], weights[view] = dp.update_ids_weights(
              view, pred_probs, sel_ids, weights, pred_y, train_data,
              add_num, gamma, regularizer)
            # update model parameter
            new_train_data, _ = dp.update_train_untrain(
              sel_ids[view], train_data, untrain_data, pred_y, weights[view])
            configs[view] = adjust_config(configs[view], len(train_data[0]), step)
            net = models.create(configs[view].model_name).cuda()
            mu.train(net, new_train_data, configs[view], device=view)

            # update y
            pred_probs[view] = mu.predict_prob(model, untrain_data,
                                               configs[view])

            # evaluation current model and save it
            acc = mu.evaluate(net, data['test'], configs[view], device=view)
            predictions = mu.predict_prob(net, data['train'], configs[view], device=view)
            save_checkpoint(
              {
                'state_dict': net.state_dict(),
                'epoch': step + 1,
                'predictions': predictions,
                'accuracy': acc
              },
              False,
              fpath=os.path.join(
                'spaco/%s.epoch%d' % (configs[view].model_name, step + 1)))
            test_preds[view] = mu.predict_prob(model, data['test'], configs[view], device=view)
        add_num +=  4000 * num_view
        fuse_y = np.argmax(sum(test_preds), axis=1)
        print('Acc:%0.4f' % np.mean(fuse_y== gt_y))
    #  print(acc)


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
