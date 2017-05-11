# Setup

Clone [cleverhans](https://github.com/openai/cleverhans) and follow directions to install.

* [My fork](https://github.com/holdenlee/cleverhans)

Make a virtual environment with `keras` and `tensorflow`. Edit `start` with path to `cleverhans`.

Note: This is set up with a old version of cleverhans and keras; may need fiddling to work with newer versions.

Run every time:
```
. ./start
```

# Pipeline

* Additional attacks (besides those in `cleverhans.attacks_tf`) are defined in `attacks.py` and re-exported in `adv_model.py`.
* `mnist_util` has utilities
	* `get_data_mnist`
	* `get_bf_mnist` (create batch feeders)
	* `make_phs` (make placeholders)
	* `start_keras_session` 
* `adv_model` has the function `make_adversarial_model` which takes a [`model` which takes inputs x (and y, epsilon,...) and outputs a dictionary with `loss`, `inference`, and `accuracy`].
* `mnist_models` has models. The standard one (as baseline) is `model1_logits`. `model1_logits` keeps outputs in logits, while `model1_no_dropout` does a softmax. Typically use the `logit` version, as calculating cross-entropy is more stable with the logits (says tensorflow).
	* `make_model_from_logits` takes a model like `model1_logits()` and outputs a function taking x, y to a dictionary with `loss`, `inference`, and `accuracy` (the input expected by `make_adversarial_model`).
* `mwu` defines the MWU (EG) optimizer and the MW analogue of `model1`, which is `mnist_mwu_model(u=50,u2=50)`.
* `mwu_adv` defines the MW analogue of `make_adv_model`, `mnist_mwu_model_adv`. (TODO: this is done with cross-entropy right now. Change so that it uses `tf.nn.softmax_cross_entropy_with_logits`.)
	
## Training

`train_single_run` trains `model1` adversarially and saves. The code looks like:
```
train_data, test_data = get_bf_mnist(FLAGS.label_smooth, FLAGS.testing)
sess = start_keras_session(1)
model = model1_logits()
adv_model, ph_dict, epsilon = make_adversarial_model_from_logits_with_inputs(model, fgsm2_clip)
evals = [Eval(test_data, FLAGS.batch_size, ['adv_accuracy'], eval_feed={'epsilon': i*0.1}, eval_steps = FLAGS.eval_steps, name="test (adversarial %f)" % (i*0.1)) for i in range(1,6)]
addons = [GlobalStep(),
	Train(lambda gs: tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate, rho=0.95, epsilon=1e-08), FLAGS.batch_size, train_feed={'epsilon' : FLAGS.epsilon}, loss = 'combined_loss', print_steps=FLAGS.print_steps),
	Saver(save_steps = FLAGS.save_steps, checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.filename)),
	SummaryWriter(summary_steps = FLAGS.summary_steps, feed_dict = {}),
	Eval(test_data, FLAGS.batch_size, ['accuracy'], eval_feed={}, eval_steps = FLAGS.eval_steps, name="test (real)")] + evals
trainer = Trainer(adv_model, FLAGS.max_steps, train_data, addons, ph_dict, train_dir = FLAGS.train_dir, verbosity=FLAGS.verbosity, sess=sess)
trainer.init_and_train()
trainer.finish()
```
`train_adv_mwu` does the same thing for the MWU model.

```
mkdir single_epochs6_ep0.3
python train_single_run.py --train_dir=single_epochs6_ep0.3/ --max_steps=3600 --epsilon=0.3 > single_epochs6_ep0.3/log.txt
mkdir train_adv_mwu
python train_adv_mwu.py --train_dir=train_adv_mwu/ --max_steps=3600 --epsilon=0.3 > single_epochs6_ep0.3/log.txt
```

## Paths

* Output of `train_single_run` is in
	* `single_epochs6_ep0.3/`
* Output of `train_adv_mwu` is in 
	* `train_adv_mwu/`
	
## Evaluation and transfer

To evaluate using the `cleverhans` framework, use `cleverhans_eval`. Use it as follows.
```
cleverhans_evaluate(model1_logits, 
                        tf.train.latest_checkpoint('./single_epochs6_ep0.3/'), 
                        adv_f,
                        FLAGS.epsilon,
                        FLAGS.batch_size)
```

To check transfer, use the functions in `transfer`. Example of use (transfer from the Keras model saved in `filepath` (`pretrain5_model1_smooth0/nets0.hdf5`) to the latest checkpoint in `train_adv_mwu/`:
```
model = model1_logits
def model2():
	d2, ph_dict, epvar = mnist_mwu_model_adv(fgsm, lambda: mnist_mwu_model(u=20,u2=20))
	return d2, ph_dict
mnist_adv_transfer(filepath, model, model2, ep = 0.3, adv_f = fgsm2_clip, keras=True, folder=FLAGS.to_file, label_smooth=FLAGS.label_smooth)
```

# Instructions (other experiments)

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

# Train on adversarial examples for a given net


```
mkdir experiment_train_on_adv
nohup python experiment_train_on_adv.py --nb_epochs=6 > experiment_train_on_adv/log.txt 2>&1 &
mkdir experiment_train_on_adv2
nohup python experiment_train_on_adv2.py --nb_epochs=6 > experiment_train_on_adv2/log.txt 2>&1 &
mkdir experiment_train_on_bogus
nohup python experiment_train_on_bogus.py --nb_epochs=6 > experiment_train_on_bogus/log.txt 2>&1 &
```

Results: (accuracy over epochs)

* Train normally: .9377, .9785, .9848, .9874, .9893, .9905. 2nd run: .9334, .9781, .9844, .9865, .9887, .9906.
* Train with adversarial only: .9320. .9773, .9828,.9848, .9859, .9875
* Train with normal and adversarial (120000 samples): (`experiment_train_on_adv2`) .9330, .9853, .9875, .9887, .9893, ,.9896

# Sigmoids

How do sigmoids do?

```
mkdir experiment_sigmoid
nohup python experiment_sigmoid.py --train_dir=experiment_sigmoid/ --max_steps=6000 > experiment_sigmoid/log.txt 2>&1 & 
mkdir experiment_sigmoid_ep0
nohup python experiment_sigmoid.py --train_dir=experiment_sigmoid_ep0/ --max_steps=6000 --epsilon=0 > experiment_sigmoid_ep0/log.txt 2>&1 & 
```

Accuracy is much worse. It also is weak to adversarial examples.

