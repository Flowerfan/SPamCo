from __future__ import print_function, absolute_import
from reid.models import model_utils as mu
from reid.utils.data import data_process as dp
from reid.utils.serialization import save_checkpoint
from reid import datasets
from reid import models
from reid.config import Config
from copy import deepcopy
import torch
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Cotrain args')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-t', '--tricks', action='store_true', help='updating tricks')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


def cotrain(configs,data,iter_steps=1,train_ratio=0.2, device='cuda:0'):
    """
    cotrain model:
    params:
    model_names: model configs
    data: dataset include train and untrain data
    save_paths: paths for storing models
    iter_steps: maximum iteration steps
    train_ratio: labeled data ratio

    """
    add_ratio = 0.5
    assert iter_steps >= 1
    train_data,untrain_data = dp.split_dataset(data.trainval, train_ratio, args.seed)
    query_gallery = list(set(data.query) | set(data.gallery))
    data_dir = data.images_dir

    new_train_data = deepcopy(train_data)
    features = []
    for step in range(iter_steps):
        pred_probs = []
        add_ids = []
        for view in range(2):
            config = configs[view]
            config.set_training(True)
            net = models.create(config.model_name,
                                  num_features=config.num_features,
                                  dropout=config.dropout,
                                  num_classes=config.num_classes).to(device)
            mu.train(net, new_train_data, data_dir, configs[view], device)
            save_checkpoint({
                'state_dict': net.state_dict(),
                'epoch': step + 1,
                'train_data': new_train_data}, False,
                fpath = os.path.join('logs/cotrain/seed_%d/%s.epoch%d' % (args.seed, config.model_name, step)
            ))
            if len(untrain_data) == 0:
                continue
            pred_probs.append(mu.predict_prob(
                net,untrain_data,data_dir,configs[view], device))
            add_ids.append(dp.sel_idx(pred_probs[view], train_data, add_ratio))

            # calculate predict probility on all data
            p_b = mu.predict_prob(net, data.trainval, data_dir, configs[view], device)
            p_y = np.argmax(p_b, axis=1)
            t_y = [c for (_,c,_,_) in data.trainval]
            print(np.mean(t_y == p_y))
            ### final evaluation
            if step + 1 == iter_steps:
                features += [
                    mu.get_feature(net, query_gallery, data.images_dir,
                                   configs[view], device)
                ]


        # update training data
        pred_y = np.argmax(sum(pred_probs), axis=1)
        add_id = sum(add_ids)
        if args.tricks:
            add_ratio += 1.2
            new_train_data, _ = dp.update_train_untrain(
                add_id,train_data,untrain_data,pred_y)
        else:
            if len(untrain_data) == 0:
                break
            new_train_data, untrain_data = dp.update_train_untrain(
                add_id,new_train_data,untrain_data,pred_y)
    acc = mu.combine_evaluate(features, data)
    print(acc)



config1 = Config()
config2 = Config(model_name='densenet121', height=224, width=224)
config3 = Config(model_name='resnet101', img_translation=2)
dataset = 'market1501std'
cur_path = os.getcwd()
logs_dir = os.path.join(cur_path, 'logs')
data_dir = os.path.join(cur_path,'data',dataset)
data = datasets.create(dataset, data_dir)


cotrain([config2,config3], data, 5)
