import numpy as np
import reuters as sp
import argparse
from converge import converge_analysis as ca

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='serial', help='training strategy, default is serial training')
parser.add_argument('--regularizer', type=str, default='hard', help='regularizer type hard or soft')
parser.add_argument('--gamma', type=float, default=0.3)
parser.add_argument('--num_train', type=int, default=500)
parser.add_argument('--iter_steps', type=int, default=15)
parser.add_argument('--data_dir', type=str, default='./rcv')
parser.add_argument('--add_num', type=int, default=10)
parser.add_argument('--tricks', action='store_true', help='cotraining replacement trick')
args = parser.parse_args()

avg_accs = []
for seed in range(5):
    avg_accs.append(
        sp.run_all(
            './rcv/',
            gamma=args.gamma,
            regularizer=args.regularizer,
            seed=seed,
            mode=args.mode,
            add_num=args.add_num,
            num_train=args.num_train,
            iter_steps=args.iter_steps,
            tricks=args.tricks))

print(avg_accs)
print(np.mean(avg_accs))
#  ca(args)
