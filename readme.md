# Instructions

## Setup

Clone [cleverhans](https://github.com/openai/cleverhans) and follow directions to install.

* [My fork](https://github.com/holdenlee/cleverhans)

Make a virtual environment with `keras` and `tensorflow`. Edit `start` with path to `cleverhans`.

Run every time:
```
. ./start
```

## Control

Control: adversarially train net for 100 epochs.

```
# control
mkdir single_epochs100_ep0.3 
nohup python train_single.py --train_dir=single_epochs100_ep0.3 --max_steps=60000 --epsilon=0.3 > single_epochs100_ep0.3/log.txt 2>&1 &

mkdir single_epochs100_ep0.5 
nohup python train_single.py --train_dir=single_epochs100_ep0.5 --max_steps=60000 --epsilon=0.5 > single_epochs100_ep0.5/log.txt 2>&1 &
```

## Pretraining

Pretrain 100 nets for 1 epoch.

```
mkdir pretrain
nohup python train_many_run.py --t=100 > pretrain/log.txt 2>&1 &
```

