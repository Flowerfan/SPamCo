import numpy as np
from util.data import transforms as T
from torch.utils.data import DataLoader
from .preprocessor import Preprocessor


def get_augmentation_func_list(aug_list, config):
    if aug_list is None: return []
    assert isinstance(aug_list, list)
    aug_func = []
    for aug in aug_list:
        if aug == 'rf':
            aug_func += [T.RandomHorizontalFlip()]
        elif aug == 'rc':
            aug_func += [T.RandomCrop(config.height, padding=config.padding)]
        elif aug == 're':
            aug_func += [T.RandomErasing(probability=0.5, sh=0.4, r1=0.3)]
        else:
            raise ValueError('wrong augmentation name')
    return aug_func



def get_transformer(config, is_training=False):
    normalizer = T.Normalize(mean=config.mean, std=config.std)
    base_transformer = [T.ToTensor(), normalizer]
    if not is_training:
        return T.Compose(base_transformer)
    aug1 = T.RandomErasing(probability=0.5, sh=0.4, r1=0.3)
    early_aug = get_augmentation_func_list(config.early_transform, config)
    later_aug = get_augmentation_func_list(config.later_transform, config)
    aug_list = early_aug + base_transformer + later_aug
    return T.Compose(aug_list)


def get_dataloader(dataset, config, is_training=False):
    transformer = get_transformer(config, is_training=is_training)
    sampler = None
    if is_training and config.sampler:
        sampler = config.sampler(dataset, config.num_instances)
    data_loader = DataLoader(Preprocessor(dataset, transform=transformer),
                             batch_size=config.batch_size,
                             num_workers=config.workers,
                             shuffle=is_training,
                             sampler=sampler,
                             pin_memory=True,
                             drop_last=is_training)
    return data_loader


def update_train_untrain(sel_idx,
                         train_data,
                         untrain_data,
                         pred_y,
                         weights=None):
    #  assert len(train_data) == len(untrain_data)
    if weights is None:
        weights = np.ones(len(untrain_data[0]), dtype=np.float32)
    add_data = [untrain_data[0][sel_idx], pred_y[sel_idx], weights[sel_idx]]
    new_untrain = [
      untrain_data[0][~sel_idx], pred_y[~sel_idx], weights[~sel_idx]
    ]
    new_train = [
      np.concatenate((d1, d2)) for d1, d2 in zip(train_data, add_data)
    ]
    return new_train, new_untrain




def select_ids(score, train_data, max_add):
    y = train_data[1]
    add_indices = np.zeros(score.shape[0])
    clss = np.unique(y)
    assert score.shape[1] == len(clss)
    pred_y = np.argmax(score, axis=1)
    ratio_per_class = [sum(y == c)/len(y) for c in clss]
    for cls in range(len(clss)):
        indices = np.where(pred_y == cls)[0]
        cls_score = score[indices, cls]
        idx_sort = np.argsort(cls_score)
        add_num = min(int(np.ceil(ratio_per_class[cls] * max_add)),
                      indices.shape[0])
        add_indices[indices[idx_sort[-add_num:]]] = 1
    return add_indices.astype('bool')


def get_lambda_class(score, pred_y, train_data, max_add):
    y = train_data[1]
    lambdas = np.zeros(score.shape[1])
    add_ids = np.zeros(score.shape[0])
    clss = np.unique(y)
    assert score.shape[1] == len(clss)
    ratio_per_class = [sum(y == c)/len(y) for c in clss]
    for cls in range(len(clss)):
        indices = np.where(pred_y == cls)[0]
        if len(indices) == 0:
            continue
        cls_score = score[indices, cls]
        idx_sort = np.argsort(cls_score)
        add_num = min(int(np.ceil(ratio_per_class[cls] * max_add)),
                      indices.shape[0])
        add_ids[indices[idx_sort[-add_num:]]] = 1
        lambdas[cls] = cls_score[idx_sort[-add_num]] - 0.1
    return add_ids.astype('bool'), lambdas


def get_ids_weights(pred_prob, pred_y, train_data, max_add, gamma, regularizer='hard'):
    '''
    pred_prob: predicted probability of all views on untrain data
    pred_y: predicted label for untrain data
    train_data: training data
    max_add: number of selected data
    gamma: view correlation hyper-parameter
    '''

    add_ids, lambdas = get_lambda_class(pred_prob, pred_y, train_data, max_add)
    weight = np.array([(pred_prob[i, l] - lambdas[l]) / (gamma + 1e-5)
                       for i, l in enumerate(pred_y)],
                      dtype='float32')
    weight[~add_ids] = 0
    if regularizer == 'hard' or gamma == 0:
        weight[add_ids] = 1
        return add_ids, weight
    weight[weight < 0] = 0
    weight[weight > 1] = 1
    return add_ids, weight

def update_ids_weights(view, probs, sel_ids, weights, pred_y, train_data,
                       max_add, gamma, regularizer='hard'):
    num_view = len(probs)
    for v in range(num_view):
        if v == view:
            continue
        ov = sel_ids[v]
        probs[view][ov, pred_y[ov]] += gamma * weights[v][ov] / (num_view - 1)
    sel_id, weight = get_ids_weights(probs[view], pred_y, train_data,
                                     max_add, gamma, regularizer)
    return sel_id, weight

def get_weights(pred_prob, pred_y, train_data, max_add, gamma, regularizer):
    lamb = get_lambda_class(pred_prob, pred_y, train_data, max_add)
    weight = np.array([(pred_prob[i, l] - lamb[l]) / gamma
                       for i, l in enumerate(pred_y)],
                      dtype='float32')
    if regularizer is 'hard':
        weight[weight > 0] = 1
        return weight
    weight[weight > 1] = 1
    return weight


def split_dataset(dataset, train_ratio=0.2, seed=0, num_per_class=400):
    """
    split dataset to train_set and untrain_set
    """
    assert 0 <= train_ratio <= 1
    np.random.seed(seed)
    pids = np.array(dataset[1])
    clss = np.unique(pids)
    sel_ids = np.zeros(len(dataset[0]), dtype=bool)
    for cls in clss:
        indices = np.where(pids == cls)[0]
        np.random.shuffle(indices)
        if num_per_class:
            sel_id = indices[:num_per_class]
        else:
            train_num = int(np.ceil((len(indices) * train_ratio)))
            sel_id = indices[:train_num]
        sel_ids[sel_id] = True
    train_set = [d[sel_ids] for d in dataset]
    untrain_set = [d[~sel_ids] for d in dataset]
    ### add sample weight
    train_set += [np.full((len(train_set[0])), 1.0)]
    return train_set, untrain_set
