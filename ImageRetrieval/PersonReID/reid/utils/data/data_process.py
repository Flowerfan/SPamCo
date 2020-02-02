import numpy as np
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from reid.utils.data.preprocessor import Preprocessor


def get_transformer(config):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    base_transformer = [T.ToTensor(), normalizer]
    if config.training is False:
        return T.Compose([T.Resize((config.height, config.width))] + base_transformer)
    if config.img_translation is None:
        return T.Compose([T.RandomSizedRectCrop(config.height, config.width),
                          T.RandomHorizontalFlip()] + base_transformer)
    return T.Compose([T.RandomTranslateWithReflect(config.img_translation),
                      T.RandomSizedRectCrop(config.height, config.width),
                      T.RandomHorizontalFlip()] + base_transformer)


def get_dataloader(dataset, data_dir, config):
    if len(dataset[0]) == 3:
        dataset = add_sample_weights(dataset)
    transformer = get_transformer(config)
    sampler = None
    if config.training and config.sampler:
        sampler = config.sampler(dataset, config.num_instances)
    data_loader = DataLoader(
        Preprocessor(dataset, root=data_dir,
                     transform=transformer),
        batch_size=config.batch_size,
        num_workers=config.workers,
        shuffle=config.shuffle,
        sampler=sampler,
        pin_memory=True,
        drop_last=config.training)
    return data_loader


def add_sample_weights(data, weights=None):
    assert isinstance(data[0], tuple)
    if weights is None:
        weights = np.ones(len(data), dtype='float32')
    assert len(data) == len(weights)
    new_data = [(*sample, weight) for sample, weight in zip(data, weights)]
    return new_data


def update_train_untrain(sel_idx, train_data, untrain_data, pred_y,
                         weights=None):
    assert len(train_data[0]) == len(untrain_data[0])
    if weights is None:
        weights = [1.0 for i in range(len(untrain_data))]
    add_data = [(untrain_data[i][0], int(pred_y[i]),
                 untrain_data[i][2], weights[i])
                for i, flag in enumerate(sel_idx) if flag]
    data1 = [untrain_data[i]
             for i, flag in enumerate(sel_idx) if not flag]
    data2 = train_data + add_data
    return data2, data1


def sel_idx(score, train_data, ratio=0.5):
    y = np.array([label for _, label, _, _ in train_data])
    add_indices = np.zeros(score.shape[0])
    clss = np.unique(y)
    assert score.shape[1] == len(clss)
    count_per_class = [sum(y == c) for c in clss]
    pred_y = np.argmax(score, axis=1)
    for cls in range(len(clss)):
        indices = np.where(pred_y == cls)[0]
        cls_score = score[indices, cls]
        idx_sort = np.argsort(cls_score)
        add_num = min(int(np.ceil(count_per_class[cls] * ratio)),
                      indices.shape[0])
        add_indices[indices[idx_sort[-add_num:]]] = 1
    return add_indices.astype('bool')


def get_lambda_class(score, pred_y, train_data, ratio=0.5):
    y = np.array([label for _, label, _, _ in train_data])
    lambdas = np.zeros(score.shape[1])
    add_ids = np.zeros(score.shape[0])
    clss = np.unique(y)
    assert score.shape[1] == len(clss)
    count_per_class = [sum(y == c) for c in clss]
    for cls in range(len(clss)):
        indices = np.where(pred_y == cls)[0]
        if len(indices) == 0:
            continue
        cls_score = score[indices, cls]
        idx_sort = np.argsort(cls_score)
        add_num = min(int(np.ceil(count_per_class[cls] * ratio)),
                      indices.shape[0])
        add_ids[indices[idx_sort[-add_num:]]] = 1
        lambdas[cls] = cls_score[idx_sort[-add_num]] - 0.1
    return add_ids.astype('bool'), lambdas


def get_ids_weights(pred_prob, pred_y, train_data,
                    add_ratio, gamma, regularizer, num_view):
    add_ids, lambdas = get_lambda_class(
        pred_prob, pred_y, train_data, add_ratio)
    #  weight = np.array([(pred_prob[i, l] - lambdas[l]) / (gamma + 1e-5) / (num_view - 1)
                       #  for i, l in enumerate(pred_y)], dtype='float32')
    weight = np.array([(pred_prob[i, l] - lambdas[l]) / (gamma + 1e-5) 
                       for i, l in enumerate(pred_y)], dtype='float32')
    weight[~add_ids] = 0
    if regularizer == 'hard' or gamma == 0:
        weight[add_ids] = 1
        return add_ids, weight
    weight[weight < 0] = 0
    weight[weight > 1] = 1
    return add_ids, weight



def get_weights(pred_prob, pred_y, train_data,
                add_ratio, gamma, regularizer):
    lamb = get_lambda_class(pred_prob, pred_y, train_data, add_ratio)
    weight = np.array([(pred_prob[i, l] - lamb[l]) / gamma
                       for i, l in enumerate(pred_y)], dtype='float32')
    if regularizer is 'hard':
        weight[weight > 0] = 1
        return weight
    weight[weight > 1] = 1
    return weight


def split_dataset(dataset, train_ratio=0.2, seed=0):
    """
    split dataset to train_set and untrain_set
    """
    assert 0 <= train_ratio <= 1
    train_set = []
    untrain_set = []
    np.random.seed(seed)
    pids = np.array([data[1] for data in dataset])
    clss = np.unique(pids)
    assert len(clss) == 751
    for cls in clss:
        indices = np.where(pids == cls)[0]
        np.random.shuffle(indices)
        train_num = int(np.ceil((len(indices) * train_ratio)))
        train_set += [dataset[i] for i in indices[:train_num]]
        untrain_set += [dataset[i] for i in indices[train_num:]]
    cls1 = np.unique([d[1] for d in train_set])
    cls2 = np.unique([d[1] for d in untrain_set])
    assert len(cls1) == len(cls2) and len(cls1) == 751
    return train_set, untrain_set
