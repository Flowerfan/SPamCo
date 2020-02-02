## Person-ReID Experiment

## Prerequisites
- Python 3.6
- PyTorch 1.3

Run self train with seed 0
```
python self_train.py --seed 0
```

Run cotrain without replacement
```
python cotrain.py --seed 0
```

Run cotrain with replacement
```
python cotrain.py --tricks --seed 0
```

Run spamco with hard regularizer ($\gamma$=0.3)
```
python spamco.py --regularizer hard --gamma 0.3 --seed 0
```

Run spamco with soft regularizer ($\gamma$=0.3)
```
python spamco.py --regularizer soft --gamma 0.3 --seed 0
```

Run spamco with freeze first prediction ($\gamma$=0.3)
```
python spamco_freeze.py --regularizer hard --gamma 0.3 --seed 0
```

Run spamco parallel with hard regularizer ($\gamma$=0.3)
```
python parallel_spamco.py --regularizer hard --gamma 0.3 --seed 0
```




The Market1501 dataset is stored in [google drive](https://drive.google.com/file/d/1QnRJJxtzQzt_mrOi392vSoYlRrb1bt6z/view?usp=sharing)

