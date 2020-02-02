import torch
from torch import nn
from torch.autograd import Variable
from reid import models
from torch.optim import lr_scheduler
from reid.trainers import Trainer
from reid.evaluators import extract_features, Evaluator
from reid.loss import TripletLoss, SoftCrossEntropyLoss
from reid.dist_metric import DistanceMetric
from reid.utils.data import data_process as dp
from reid.utils import to_torch, to_numpy
from reid.evaluators import pairwise_distance, evaluate_all
import numpy as np
from collections import defaultdict


def get_optim_params(config, params, device):

    if config.loss_name not in ['softmax', 'triplet', 'weight_softmax']:
        raise ValueError('wrong loss name')
    if config.loss_name is 'triplet':
        criterion = TripletLoss(margin=config.margin)
        optimizer = torch.optim.Adam(params, lr=config.lr,
                                     weight_decay=config.weight_decay)
        return criterion, optimizer
    optimizer = torch.optim.SGD(params,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay,
                                nesterov=True)
    if config.loss_name is 'softmax':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = SoftCrossEntropyLoss()
    criterion = criterion.to(device)
    return criterion, optimizer


def train_model(model, dataloader, config, device):
    """
    train model given the dataloader the criterion,
    stop when epochs are reached
    params:
        model: model for training
        dataloader: training data
        config: training configuration
        criterion
    """
    if hasattr(model, 'base'):
        base_param_ids = set(map(id, model.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.base.parameters(), 'lr': 0.01},
            {'params': new_params, 'lr': 0.1}]
    else:
        param_groups = model.parameters()

    criterion, optimizer = get_optim_params(config, param_groups, device)

    trainer = Trainer(model, criterion, device)

    # schedule learning rate
    def adjust_lr(epoch):
        step_size = 40
        lr = config.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    for epoch in range(config.epochs):
        #adjust_lr(epoch)
        trainer.train(epoch, dataloader, optimizer, print_freq=config.print_freq)
        scheduler.step()


def train(net, train_data, data_dir, config, device):
    config.set_training(True)
    #  net = models.create(config.model_name,
                          #  num_features=config.num_features,
                          #  dropout=config.dropout,
                          #  num_classes=config.num_classes).to(device)
    #  model = nn.DataParallel(model).cuda(device)
    dataloader = dp.get_dataloader(train_data, data_dir, config)
    train_model(net, dataloader, config, device)
    return net


def get_feature(model, data, data_dir, config, device):
    config.set_training(False)
    dataloader = dp.get_dataloader(data, data_dir, config)
    features, _ = extract_features(model, dataloader, device)
    #features = {k:nn.functional.softmax(Variable(v), dim=1).values()
    #            for k,v in features.items()}
    return features


def predict_prob(model, data, data_dir, config, device):
    config.set_training(False)
    model.eval()
    dataloader = dp.get_dataloader(data, data_dir, config)
    probs = []
    for i, (imgs, _, _, _, _) in enumerate(dataloader):
        with torch.no_grad():
            inputs = to_torch(imgs).to(device)
            #  inputs = to_torch(imgs).cuda()
            #  inputs = Variable(inputs).to(device)
            output = model(inputs)
        prob = nn.functional.softmax(output, dim=1)
        probs += [prob.data.cpu().numpy()]
    probs = np.concatenate(probs)
    return probs


def train_predict(train_data, untrain_data, data_dir, config):
    model = train(train_data, data_dir, config)
    pred_prob = predict_prob(model, untrain_data, data_dir, config)
    return pred_prob


def get_clusters(model, data_loader, num_classes):
    features, labels = extract_features(model, data_loader)
    assert np.unique(labels.values()) == num_classes
    class_features = defaultdict(list)
    for k, v in labels.items():
        class_features[v].append(features[k].numpy())
    clusters = [np.mean(class_features[i], axis=0)
                for i in range(num_classes)]
    clusters = torch.from_numpy(np.array(clusters, dtype='float32'))
    return torch.autograd.Variable(clusters)


def combine_evaluate(features, dataset):
    metric = DistanceMetric(algorithm='euclidean')
    distmats = [pairwise_distance(feature, dataset.query, dataset.gallery, metric=None)\
            for feature in features]
    distmats = np.array([dist.numpy() for dist in distmats])
    distmat = np.sum(distmats, axis=0)
    return evaluate_all(distmat, dataset.query, dataset.gallery)


def evaluate(model, dataset, config, device):
    config.set_training(False)
    query, gallery = dataset.query, dataset.gallery
    dataloader = dp.get_dataloader(
        list(set(dataset.query) | set(dataset.gallery)),
        dataset.images_dir, config)
    #  metric = DistanceMetric(algorithm=config.dist_metric)
    #  metric.train(model, dataloader)
    evaluator = Evaluator(model, device)
    return evaluator.evaluate(dataloader, query, gallery, metric=None, print_freq=config.batch_size)
