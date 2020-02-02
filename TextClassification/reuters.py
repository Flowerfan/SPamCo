import numpy as np
import sklearn.datasets as datasets
import os
from scipy.sparse import vstack
from sklearn.svm import LinearSVC
import joblib
import copy
#  from sklearn.externals import joblib
import argparse


def write_data(data_dir, save_dir='./rcv'):
    sd = './rcv/'
    clss_factory = {
        'C15': '0',
        'CCAT': '1',
        'E21': '2',
        'ECAT': '3',
        'GCAT': '4',
        'M11': '5'
    }

    for subdir in os.listdir(data_dir):
        d = os.path.join(data_dir, subdir)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.startswith('.'):
                    continue
                print('process file%s' % fn)
                fp = os.path.join(d, fn)
                sp = os.path.join(sd, subdir, fn)
                os.mkdirs(os.path.join(sd, subdir), exist_ok=True)
                with open(sp, 'w') as fw:
                    for con in open(fp, 'r').readlines():
                        strp = con.split(' ')
                        strp[0] = clss_factory[strp[0]]
                        new_line = ' '.join(strp)
                        fw.write(new_line)


def update_train(sel_id, train_data, train_labels, untrain_data, pred_y,
                 weight):
    new_train_data = vstack([train_data, untrain_data[sel_id]])
    new_train_y = np.concatenate([train_labels, pred_y[sel_id]])
    new_weight = np.concatenate([np.ones(train_data.shape[0]), weight[sel_id]])
    return new_train_data, new_train_y, new_weight

def update_train_untrain(sel_ids, train_data, train_labels, untrain_data, pred_y):
    sel_id = np.array(sel_ids[0] + sel_ids[1], dtype=bool)
    new_train_data = [vstack([d1, d2[sel_id]]) for d1,d2 in zip(train_data, untrain_data)]
    new_untrain_data = [d[~sel_id] for d in untrain_data]
    new_train_y = np.concatenate([train_labels, pred_y[sel_id]])
    return new_train_data, new_train_y, new_untrain_data 


def get_ids_lambdas(score, pred_y, labels, add_samples):
    clss = np.unique(labels)
    sel_ids = np.zeros(score.shape[0])
    lambdas = np.zeros(score.shape[1])
    pred_y = np.argmax(score, axis=1)
    for cls in range(len(clss)):
        indices = np.where(pred_y == cls)[0]
        if len(indices) == 0:
            continue
        cls_score = score[indices, cls]
        idx_sort = np.argsort(cls_score)
        add_num = min(indices.shape[0], add_samples)
        sel_ids[indices[idx_sort[-add_num:]]] = 1
        lambdas[cls] = cls_score[idx_sort[-add_num]] - 0.01
    return sel_ids.astype('bool'), lambdas


def get_ids_weights(score, pred_y, labels, add_num, gamma=0, regularizer='hard'):
    sel_ids, lambdas = get_ids_lambdas(score, pred_y, labels, add_num)
    weight = np.array([(score[i, l] - lambdas[l]) / (gamma + 1e-5)
                       for i, l in enumerate(pred_y)])
    weight[~sel_ids] = 0
    if regularizer == 'hard' or gamma == 0:
        weight[sel_ids] = 1
        return sel_ids, weight
    weight[weight > 1] = 1
    sel_ids = weight > 0
    return sel_ids, weight


def update_ids_weights(view,
                       scores,
                       sel_ids,
                       weights,
                       pred_y,
                       labels,
                       add_num,
                       gamma,
                       regularizer,
                       mode='serial'):
    num_view = len(scores)
    for v in range(num_view):
        if gamma == 0:
            break
        if v == view and mode == 'serial':
            continue
        sv = sel_ids[v]
        scores[view][sv, pred_y[sv]] += gamma * weights[v][sv] / (num_view - 1)
    sel_id, weight = get_ids_weights(scores[view], pred_y, labels, add_num,
                                     gamma, regularizer)
    # if regularizer == 'soft':
    # weight = weight / (num_view - 1)
    return sel_id, weight


def test_model(clfs, X, y):
    preds = []
    for idx, clf in enumerate(clfs):
        preds.append(clf.decision_function(X[idx]))
    preds = sum(preds)
    pred_y = np.argmax(preds, axis=1)
    return np.mean(pred_y == y)

def cotrain(l_data,
           lbls,
           u_data,
           iter_steps=1,
           addn=10,
           data_name='EN',
           seed=0,
           tricks=False):

    num_view = len(l_data)
    # initiate classifier
    clfs = [LinearSVC(max_iter=10000) for _ in range(num_view)]
    #  clfs = [LinearSVC() for _ in range(num_view)]
    scores = [None for _ in range(num_view)]
    sel_ids = [None for _ in range(num_view)]
    add_num = addn * num_view
    train_data = [d for d in l_data]
    train_labels = copy.deepcopy(lbls)
    untrain_data = [d for d in u_data]
    for step in range(iter_steps):
        for view in range(num_view):
            ## train classifier
            clfs[view].fit(train_data[view], train_labels)
            # save_model
            joblib.dump(
                clfs[view],
                './logs/%s/cotrain_addnum%0.3d_seed%d_view%d_step%0.3d.pkl' %
                (data_name, addn, seed, view, step + 1))
            ##predict
            scores[view] = clfs[view].decision_function(untrain_data[view])
            sel_id, _ = get_ids_weights(scores[view], np.argmax(scores[view], axis=1), train_labels, add_num)
            sel_ids[view] = sel_id
        pred_y = np.argmax(sum(scores), axis=1)
        if tricks:
            ### with repalcement
            add_num += addn * num_view
            train_data, train_labels, _ = update_train_untrain(sel_ids, l_data, lbls, untrain_data, pred_y)
        else:
            ### without repalcement
            train_data, train_labels, untrain_data = update_train_untrain(sel_ids, train_data, train_labels, untrain_data, pred_y)
            if untrain_data[0].shape[0] < add_num:
                break

    return clfs


def spaco(l_data,
          lbls,
          u_data,
          iter_steps=1,
          addn=10,
          gamma=0.1,
          regularizer='hard',
          data_name='EN',
          seed=0):

    # initiate classifier
    clfs = []
    scores = []
    sel_ids = []
    weights = []
    add_num = addn
    num_view = len(l_data)
    # initiate classifier
    for view in range(num_view):
        clfs.append(LinearSVC(max_iter=10000))
        clfs[view].fit(l_data[view], lbls)
        scores.append(clfs[view].decision_function(u_data[view]))
        joblib.dump(clfs[view],
                    './logs/%s/%s_addnum%0.3d_seed%d_view%d_step000.pkl' %
                    (data_name, regularizer, addn, seed, view))

    pred_y = np.argmax(sum(scores), axis=1)
    # initiate weiths for unlabled smaples
    for view in range(num_view):
        sel_id, weight = get_ids_weights(scores[view], pred_y, lbls,
                                         add_num, gamma, regularizer)
        sel_ids.append(sel_id)
        weights.append(weight)
    for step in range(iter_steps):
        for view in range(num_view):
            #update v
            sel_ids[view], weights[view] = update_ids_weights(
                view, scores, sel_ids, weights, pred_y, lbls, add_num, gamma,
                regularizer)

            #update w
            data = update_train(sel_ids[view], l_data[view], lbls,
                                u_data[view], pred_y, weights[view])
            new_train_X, new_train_y, new_train_weight = data
            #  import pdb;pdb.set_trace()
            clfs[view].fit(new_train_X, new_train_y)
            #  clfs[view].fit(new_train_X, new_train_y, sample_weight=new_train_weight)

            #update y
            scores[view] = clfs[view].decision_function(u_data[view])
            pred_y = np.argmax(sum(scores), axis=1)

            # update v
            add_num += addn
            sel_ids[view], weights[view] = update_ids_weights(
                view, scores, sel_ids, weights, pred_y, lbls, add_num, gamma,
                regularizer)

            # save_model
            joblib.dump(
                clfs[view],
                './logs/%s/%s_addnum%0.3d_seed%d_view%d_step%0.3d.pkl' %
                (data_name, regularizer, addn, seed, view, step + 1))

    return clfs


def spaco_parallel(l_data,
                   lbls,
                   u_data,
                   iter_steps=1,
                   addn=10,
                   gamma=0.3,
                   regularizer='hard',
                   data_name='EN',
                   seed=0):

    # initiate classifier
    clfs = []
    scores = []
    sel_ids = []
    weights = []
    add_num = addn
    num_view = len(l_data)
    # initiate classifier
    for view in range(num_view):
        clfs.append(LinearSVC(max_iter=10000))
        clfs[view].fit(l_data[view], lbls)
        scores.append(clfs[view].decision_function(u_data[view]))
        joblib.dump(clfs[view],
                    './logs/%s/%s_addnum%0.3d_seed%d_view%d_step000.pkl' %
                    (data_name, regularizer, view, seed, addn))

    pred_y = np.argmax(sum(scores), axis=1)
    # initiate weiths for unlabled smaples
    for view in range(num_view):
        sel_id, weight = get_ids_weights(scores[view], pred_y, lbls,
                                         add_num, gamma, regularizer)
        sel_ids.append(sel_id)
        weights.append(weight)
    for step in range(iter_steps):
        for view in range(num_view):
            #update v
            sel_ids[view], weights[view] = update_ids_weights(
                view,
                scores,
                sel_ids,
                weights,
                pred_y,
                lbls,
                add_num,
                gamma,
                regularizer,
                mode='parallel')

            #update w
            data = update_train(sel_ids[view], l_data[view], lbls,
                                u_data[view], pred_y, weights[view])
            new_train_X, new_train_y, new_train_weight = data
            clfs[view].fit(new_train_X, new_train_y, new_train_weight)

            #update y
            scores[view] = clfs[view].decision_function(u_data[view])

            #save_model
            joblib.dump(
                clfs[view],
                './logs/%s/%s_addnum%0.3d_seed%d_view%d_step%0.3d.pkl' %
                (data_name, regularizer, addn, seed, view, step + 1))

        add_num += addn * num_view

        pred_y = np.argmax(sum(scores), axis=1)
    return clfs


def get_data(files, dire, num_train=500, seed=0):
    np.random.seed(seed=seed)
    X = [[], [], [], [], []]
    y = [[], [], [], [], []]
    clss = [0, 1, 2, 3, 4, 5]
    numfs = [21531, 24893, 34279, 15506, 11547]
    for i, filename in enumerate(files):
        idx = i % 5
        d = datasets.load_svmlight_file(
            os.path.join(dire, filename), n_features=numfs[idx])

        X[idx].append(d[0])
        y[idx].append(d[1])
    X = [vstack(data) for data in X]
    y = [np.concatenate(labels) for labels in y]

    ### sel labeled and unlabeled

    label_num = 14
    clss_ids = [np.where(y[0] == c)[0] for c in clss]
    ids = np.array([np.random.choice(c_id, num_train) for c_id in clss_ids])
    l_ids = np.concatenate(ids[:, :label_num])
    u_ids = np.concatenate(ids[:, label_num:])

    sel_X = [data[l_ids] for data in X]
    test_ids = np.ones(X[0].shape[0], dtype='bool')
    test_ids[l_ids] = 0
    test_X = [data[test_ids] for data in X]
    test_y = y[0][test_ids]
    sel_U = [data[u_ids] for data in X]
    sel_y = y[0][l_ids]
    return sel_X, sel_y, sel_U, test_X, test_y


def run_all(data_dir,
            gamma=0.3,
            iter_steps=30,
            regularizer='hard',
            add_num=10,
            seed=0,
            mode='serial',
            num_train=500,
            tricks=False):

    sub_dirs = ['EN', 'FR', 'GR', 'IT', 'SP']
    numes = [18758, 26648, 29953, 24039, 12342]
    all_ins = sum(numes)
    avg_acc = 0
    for idx, sub_dir in enumerate(sub_dirs):
        dire = os.path.join(data_dir, sub_dir)
        files = os.listdir(os.path.join(data_dir, sub_dir))
        files = np.sort(files)
        data = get_data(files, dire, seed=seed, num_train=num_train)
        sel_X, sel_y, sel_U, test_X, test_y = data
        if mode == 'serial':
            clfs = spaco(
                sel_X,
                sel_y,
                sel_U,
                iter_steps=iter_steps,
                addn=add_num,
                gamma=gamma,
                regularizer=regularizer,
                data_name=sub_dir,
                seed=seed)
        elif mode == 'parallel':
            clfs = spaco_parallel(
                sel_X,
                sel_y,
                sel_U,
                iter_steps=iter_steps,
                addn=add_num,
                gamma=gamma,
                regularizer=regularizer,
                data_name=sub_dir,
                seed=seed)
        elif mode == 'cotrain':
            clfs = cotrain(
                sel_X,
                sel_y,
                sel_U,
                iter_steps=iter_steps,
                addn=add_num,
                data_name=sub_dir,
                seed=seed,
                tricks=tricks)
        else:
            raise ValueError('wrong training mode')

        acc = test_model(clfs, test_X, test_y)
        print('language: %s, acc: %f' % (sub_dir, acc))
        avg_acc += acc * numes[idx] / all_ins
    print('Overall average acc: %f' % avg_acc)
    return avg_acc


def main(args):
    dire = os.path.join('/Users/flowerfan/Downloads/rcv1rcv2aminigoutte/')
    save_dir = './rcv'
    if not os.path.exists('./rcv/EN'):
        write_data(dire, save_dir=save_dir)
    np.random.seed(args.seed)
    run_all(
        save_dir,
        args.gamma,
        args.iter_steps,
        args.regularizer,
        seed=args.seed,
        num_train=args.num_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='spaco arguments')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-r', '--regularizer', type=str, default='hard')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--add_num', type=int, default=10)
    parser.add_argument('--num_train', type=int, default=500)
    parser.add_argument('--mode', type=str, default='serial')
    parser.add_argument(
        '--iter_steps', type=int, default=20, help='maximum iteration steps')
    args = parser.parse_args()
    main(args)
