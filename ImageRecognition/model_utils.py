import torch
from torch import nn
import models
from util.data import data_process as dp
import numpy as np
from collections import OrderedDict
from loss import SoftCrossEntropyLoss


def get_optim_params(config, params):

    if config.loss_name not in ['softmax', 'weight_softmax']:
        raise ValueError('wrong loss name')
    optimizer = torch.optim.SGD(params,
                                lr=config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay,
                                nesterov=True)
    if config.loss_name == 'softmax':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = SoftCrossEntropyLoss()
    return criterion, optimizer


def train_model(model, dataloader, config, device):
    """
    train model given the dataloader the criterion,
    stop when epochs are reached
    params:
        model: model for training
        dataloader: training data
        config: training config
        criterion
    """
    param_groups = model.parameters()
    criterion, optimizer = get_optim_params(config, param_groups)
    criterion = criterion.to(device)

    def adjust_lr(epoch, step_size=20):
        lr = 0.1 * (0.1**(epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    for epoch in range(config.epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        adjust_lr(epoch, config.step_size)
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, weights) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            weights = weights.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets, weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
              (train_loss /
               (batch_idx + 1), 100. * correct / total, correct, total))


def train(model, train_data, config, device):
    #  model = models.create(config.model_name)
    #  model = nn.DataParallel(model).cuda()
    dataloader = dp.get_dataloader(train_data, config, is_training=True)
    train_model(model, dataloader, config, device)
    #  return model


def predict_prob(model, data, config, device):
    model.eval()
    dataloader = dp.get_dataloader(data, config)
    probs = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            prob = nn.functional.softmax(output, dim=1)
            probs += [prob.data.cpu().numpy()]
    return np.concatenate(probs)


def evaluate(model, data, config, device):
    model.eval()
    correct = 0
    total = 0
    dataloader = dp.get_dataloader(data, config)
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print('Accuracy on Test data: %0.5f' % acc)
    return acc
