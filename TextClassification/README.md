## Text Classification

## Prerequisites
- Python 3.6
- Scikit-learn 


Run cotrain without replacement, add 10 example each iteration, total 20 iterations 
```
python main.py --mode cotrain --add_num 10 --iter_steps 20 
```

Run cotrain with replacement, add 10 example each iteration, total 20 iterations 
```
python main.py --mode cotrain --tricks --add_num 10 --iter_steps 20
```

Run spamco with hard regularizer ($\gamma$=0.3), add 10 example each iteration, total 20 iterations 
```
python main.py  --regularizer hard --gamma 0.3 --add_num 10 --iter_steps 20
```

Run spamco with soft regularizer ($\gamma$=0.3), add 10 example each iteration, total 20 iterations 
```
python main.py  --regularizer soft --gamma 0.3 --add_num 10 --iter_steps 20
```

The dataset is stored in [google drive](https://drive.google.com/file/d/18gre4ZicnnHcEx5bHw3wXBuYjMFVdKKr/view?usp=sharing)

