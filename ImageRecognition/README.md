# Image Recognition Experiment


## Prerequisites
- Python 3.6
- PyTorch 1.3



Run cotrain without replacement on CIFAR10 with 4000 labeled examples
```
python cotrain.py --dataset cifar10 --num-per-class 400 --seed 0
```

Run cotrain with replacement on CIFAR10 with 4000 labeled examples
```
python cotrain.py --dataset cifar10 --num-per-class 400 --tricks --seed 0
```

Run spamco with hard regularizer ($\gamma$=0.3) on CIFAR10 with 4000 labeled examples
```
python spamco.py --dataset cifar10 --num-per-class 400 --regularizer hard --gamma 0.3 --seed 0
```

Run spamco with soft regularizer ($\gamma$=0.3) on CIFAR10 with 4000 labeled examples
```
python spamco.py --dataset cifar10 --num-per-class 400 --regularizer soft --gamma 0.3 --seed 0
```


Run spamco parallel with hard regularizer ($\gamma$=0.3) on CIFAR10 with 4000 labeled examples
```
python parallel_spamco.py --dataset cifar10 --num-per-class 400 --regularizer hard --gamma 0.3 --seed 0
```

