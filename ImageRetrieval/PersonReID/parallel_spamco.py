import os
import torch
import argparse
from reid.models import model_utils as mu
from reid.utils.data import data_process as dp
from reid.config import Config
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid import datasets
from reid import models
import numpy as np
import torch.multiprocessing as mp


parser = argparse.ArgumentParser(description='soft_spaco')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-r', '--regularizer', type=str, default='hard')
parser.add_argument('-d', '--dataset', type=str, default='market1501std')
parser.add_argument('--gamma', type=float, default=0.3)
parser.add_argument('--iter_steps', type=int, default=4)
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

def test_acc(model, data, data_dir, config):

    p_b = mu.predict_prob(model, data,
                          data_dir, config)
    p_y = np.argmax(p_b, axis=1)
    t_y = [c for (_, c, _, _) in data]
    print(np.mean(t_y == p_y))

def wrap_train(args):
    return mu.train(*args)

def update_ids_weights(view, probs, sel_ids, weights, pred_y,
                       train_data, add_ratio, gamma, regularizer):
    num_view = len(probs)
    for v in range(num_view):
        if v == view:
            continue
        ov = sel_ids[v]
        probs[view][ov, pred_y[ov]] += gamma * weights[v][ov] / (num_view - 1)
    sel_id, weight = dp.get_ids_weights(probs[view], pred_y, train_data,
                                        add_ratio, gamma, regularizer, num_view)
    return sel_id, weight

def parallel_train(nets, train_datas, data_dir, configs):
    processes = []
    for view, net in enumerate(nets):
        config = configs[view]
        p = mp.Process(target=mu.train, args=(net, train_datas[view], data_dir, config, view))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

def wrap_predict(net, data, data_dir, config, view, probs):
    probs[view] = mu.predict_prob(net, data, data_dir, config, view)
    

def parallel_predict(nets, data, data_dir, configs):
    processes = []
    #  pred_probs = mp.Array('probs', range(len(nets)))
    #  mAPs = mp.Array('mAPs', range(len(nets)))
    manager = mp.Manager()
    pred_probs = manager.dict()
    for view in range(len(nets)):
        p = mp.Process(target=wrap_predict, args=(nets[view], data, data_dir, configs[view], view, pred_probs))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    return pred_probs.values()

def wrap_evaluate(net, data, config, view, mAPs):
    mAPs[view] = mu.evaluate(net, data, config, view)

def parallel_evaluate(nets, data, configs):
    processes = []
    manager = mp.Manager()
    mAPs = manager.dict()
    for view in range(len(nets)):
        p = mp.Process(target=wrap_evaluate, args=(nets[view], data, configs[view], view, mAPs))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print(mAPs.values())
    #  return pred_probs.values()

def spaco(configs, data, iter_steps=1, gamma=0, train_ratio=0.2, regularizer='hard'):
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
        data.trainval, train_ratio, args.seed)
    data_dir = data.images_dir
    query_gallery = list(set(data.query) | set(data.gallery))
    save_dir = os.path.join('logs', 'parallel_spaco', regularizer,
                            'seed_%d' % args.seed)
    ###########
    # initiate classifier to get preidctions
    ###########

    add_ratio = 0.5
    pred_probs = []
    sel_ids = []
    weights = []
    features = []
    nets = []
    train_datas = []
    for view in range(num_view):
        net = models.create(configs[view].model_name,
                              num_features=configs[view].num_features,
                              dropout=configs[view].dropout,
                              num_classes=configs[view].num_classes).to(view)
        nets += [net]
        train_datas += [train_data]
    parallel_train(nets, train_datas, data_dir, configs)
    parallel_evaluate(nets, data, configs)
    pred_probs = parallel_predict(nets, untrain_data, data_dir, configs)
    pred_y = np.argmax(sum(pred_probs), axis=1)

    # initiate weights for unlabled examples
    for view in range(num_view):
        sel_id, weight = dp.get_ids_weights(pred_probs[view], pred_y, train_data, add_ratio, gamma, regularizer, num_view)
        sel_ids.append(sel_id)
        weights.append(weight)

    #  # start iterative training
    for step in range(iter_steps):
        for view in range(num_view):
            # update v_view
            sel_ids[view], weights[view] = update_ids_weights(view, pred_probs, sel_ids, weights, pred_y, train_data, add_ratio, gamma, regularizer)
            #  # update w_view
            new_train_data, _ = dp.update_train_untrain(
                sel_ids[view], train_data, untrain_data, pred_y, weights[view])
            train_datas[view] = new_train_data
            nets[view] = models.create(configs[view].model_name,
                                  num_features=configs[view].num_features,
                                  dropout=configs[view].dropout,
                                  num_classes=configs[view].num_classes).to(view)
        parallel_train(nets, train_datas, data_dir, configs)
        pred_probs = parallel_predict(nets, untrain_data, data_dir, configs)
        ##  update predictions
        pred_y = np.argmax(sum(pred_probs), axis=1)
        add_ratio += 1.2
    parallel_evaluate(nets, data, configs)

    ### save checkpoints
    for view in range(num_view):
        #  test_acc(model, data.trainval, data_dir, configs[view])
        save_checkpoint(
            {
                'state_dict': nets[view].state_dict(),
                'epoch': step + 1,
                'train_data': new_train_data,
                'trainval': data.trainval,
                'predictions': pred_probs[view],
                #  'performance': mAP
            },
            False,
            fpath=os.path.join(
                save_dir,
                '%s.epoch%d' % (configs[view].model_name, step + 1)))
        if step + 1 == iter_steps:
            features += [
                mu.get_feature(nets[view], query_gallery, data.images_dir,
                               configs[view], view)
            ]
    acc = mu.combine_evaluate(features, data)
    #  print(acc)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    config1 = Config(model_name='resnet50', loss_name='weight_softmax')
    config2 = Config(model_name='densenet121', loss_name='weight_softmax',
                     height=224, width=224)
    config3 = Config(model_name='resnet101',
                     loss_name='weight_softmax', img_translation=2)

    dataset = args.dataset
    cur_path = os.getcwd()
    logs_dir = os.path.join(cur_path, 'logs')
    data_dir = os.path.join(cur_path, 'data', dataset)
    data = datasets.create(dataset, data_dir)

    spaco([config1, config2, config3], data, gamma=args.gamma, iter_steps=args.iter_steps, regularizer=args.regularizer)
