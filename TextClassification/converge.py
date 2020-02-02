import numpy as np
import reuters as sp
import argparse
import os
from sklearn.externals import joblib


def converge_analysis(args):
    steps = (args.iter_steps-1) * 5
    regularizer = args.regularizer
    add_num = args.add_num
    sub_dirs = ['EN', 'FR', 'GR', 'IT', 'SP']
    numes = [18758, 26648, 29953, 24039, 12342]
    all_ins = sum(numes)
    acc5 = []

    for seed in range(5):
        avg_accs = np.zeros((steps, 6))

        for idx, data_name in enumerate(sub_dirs):
            dire = os.path.join(args.data_dir, data_name)
            files = os.listdir(dire)
            files = np.sort(files)
            data = sp.get_data(files, dire, seed=seed)
            sel_X, sel_y, sel_U, test_X, test_y = data
            for step in range(steps):
                clfs = []
                for view in range(5):
                    s = step//5 + int((step % 5)>(view))
                    pth = os.path.join('./logs/%s' % data_name,
                                       '%s_addnum%0.3d_seed%d_view%d_step%0.3d.pkl' %
                                       (regularizer, add_num, seed, view, s))
                    clfs.append(joblib.load(pth))
                acc = sp.test_model(clfs, test_X, test_y)
                print('step %d, language: %s, acc %f' % (step, data_name, acc))
                avg_accs[step, idx] = acc
        avg_accs[:, 5] = np.sum(avg_accs[:, :5] * numes / all_ins, axis=1)
        acc5.append(avg_accs[:, 5])
        #  np.savetxt(
            #  './logs/converge/converge_%s_%d.txt' % (regularizer, seed),
            #  avg_accs,
            #  fmt='%0.5f')
    np.savetxt('./logs/converge/converge_%s_addn%0.3d.txt' % (regularizer, args.add_num),
               np.mean(acc5, axis=0),
               fmt='%0.5f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='spaco arguments')
    parser.add_argument('-r', '--regularizer', type=str, default='hard')
    parser.add_argument('--iter_steps', type=int, default=30)
    parser.add_argument('--data_dir', type=str, default='./rcv')
    parser.add_argument('--add_num', type=int, default=10)
    args = parser.parse_args()
    converge_analysis(args)
