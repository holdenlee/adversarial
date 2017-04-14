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

Python 3 seems to have a problem; I used python 2.

Control: adversarially train net for 100 epochs.

```
# 0.3
mkdir single_epochs100_ep0.3 
nohup python2 train_single.py --train_dir=single_epochs100_ep0.3 --max_steps=60000 --epsilon=0.3 > single_epochs100_ep0.3/log.txt 2>&1 &
# 0.5
mkdir single_epochs100_ep0.5 
nohup python2 train_single.py --train_dir=single_epochs100_ep0.5 --max_steps=60000 --epsilon=0.5 > single_epochs100_ep0.5/log.txt 2>&1 &
# 1.0 (sanity check)
mkdir single_epochs100_ep1
nohup python2 train_single.py --train_dir=single_epochs100_ep1 --max_steps=60000 --epsilon=1.0 > single_epochs100_ep1/log.txt 2>&1 &
```

## Pretraining

Pretrain 100 nets for 1 epoch. (NOTE: use model1, it has less parameters.)

```
mkdir pretrain
nohup python2 train_many_run.py --t=100 > pretrain/log.txt 2>&1 &

#python train_many_run2.py --train_dir=pretrain5_model2_smooth0/ --label_smooth=0 --t=100 --nb_epochs=5 --filename=nets

#mkdir pretrain5_model2_smooth0
#nohup python train_many_run2.py --train_dir=pretrain5_model2_smooth0/ --label_smooth=0 --t=100 --nb_epochs=5 --filename=nets > pretrain5_model2_smooth0/log.txt 2>&1 &

mkdir pretrain5_model1_smooth0
nohup python train_many_run.py --train_dir=pretrain5_model1_smooth0/ --label_smooth=0 --t=100 --nb_epochs=5 --filename=nets > pretrain5_model1_smooth0/log.txt 2>&1 &
```

## Train mixture

Training a mixture of 10 nets. 
```
# 0.3
mkdir mix10_pretrain1_epochs100_ep0.3_reg0.5 
nohup python train_mix_run.py --train_dir=mix10_pretrain1_epochs100_ep0.3_reg0.5/ --t=10 --max_steps=60000 --epsilon=0.3 --reg_weight=0.5 > mix10_pretrain1_epochs100_ep0.3_reg0.5/log.txt 2>&1 &
# 0.5
mkdir mix10_pretrain1_epochs100_ep0.5_reg0.5 
nohup python train_mix_run.py --train_dir=mix10_pretrain1_epochs100_ep0.5_reg0.5/ --t=10 --max_steps=60000 --epsilon=0.5 --reg_weight=0.5 > mix10_pretrain1_epochs100_ep0.5_reg0.5/log.txt 2>&1 &
# 1
mkdir mix10_pretrain1_epochs100_ep1_reg0.5 
nohup python train_mix_run.py --train_dir=mix10_pretrain1_epochs100_ep1_reg0.5/ --t=10 --max_steps=60000 --epsilon=1.0 --reg_weight=0.5 > mix10_pretrain1_epochs100_ep1_reg0.5/log.txt 2>&1 &
```

Train mixture of 100.
```
# 0.3
mkdir mix100_pretrain1_epochs100_ep0.3_reg0.5 
nohup python train_mix.py --train_dir=mix100_pretrain1_epochs100_ep0.3_reg0.5/ --t=100 --max_steps=60000 --epsilon=0.3 --reg_weight=0.5 > mix10_pretrain1_epochs100_ep0.3_reg0.5/log.txt 2>&1 &
# 0.5
mkdir mix100_pretrain1_epochs100_ep0.5_reg0.5 
nohup python train_mix.py --train_dir=mix100_pretrain1_epochs100_ep0.5_reg0.5/ --t=100 --max_steps=60000 --epsilon=0.5 --reg_weight=0.5 > mix10_pretrain1_epochs100_ep0.5_reg0.5/log.txt 2>&1 &
# 1
mkdir mix100_pretrain1_epochs100_ep1_reg0.5 
nohup python train_mix.py --train_dir=mix100_pretrain1_epochs100_ep1_reg0.5/ --t=100 --max_steps=60000 --epsilon=1.0 --reg_weight=0.5 > mix10_pretrain1_epochs100_ep1_reg0.5/log.txt 2>&1 &
```

## Plotting

```
mkdir plots
python plots.py
```

# L1 regularization

Using just L1 regularization (without adversarial training) doesn't help. (I applied L1-L2 regularization to all layers and activation regularization.)
```
mkdir experiment_l1
nohup python experiment_l1.py --train_dir=experiment_l1/ --label_smooth=0 --nb_epochs=5 --filename=nets > experiment_l1/log.txt 2>&1 &
mkdir experiment_l1_test
nohup python experiment_l1_test.py --train_dir=experiment_l1/ --filename=nets --ep=0.3 > experiment_l1_test/log.txt 2>&1 &
```

# Transferring examples

Examples transfer very well (90+% success rate).
```
mkdir experiment_adv_transfer
nohup python experiment_adv_transfer.py --train_dir=pretrain5_model1_smooth0/ --label_smooth=0 --t=100 --filename=nets --ep=0.3 --save_as=experiment_adv_transfer/transfer.txt > experiment_adv_transfer/log.txt 2>&1 &
```
