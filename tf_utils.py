import tensorflow as tf
from functools import *
import itertools
import inspect
from datetime import datetime
import re
import os.path
import time
import numpy as np
#from six.moves import xrange
from utils import *

DEFAULT_NAME = "tmp/model.ckpt"

"""
Saving and loading. 

See https://www.tensorflow.org/programmers_guide/variables#checkpoint_files
"""
def save_model(sess, name = DEFAULT_NAME, saver=None, verbosity=1):
    if saver == None:
        saver = tf.train.Saver()
    save_path = saver.save(sess, name)
    printv("Model saved in file: %s" % save_path, verbosity, 1)
    return saver

def load_model(sess, name = DEFAULT_NAME, saver=None, verbosity=1): #, return_saver=False
    if saver == None:
        saver = tf.train.Saver()
    saver.restore(sess, name)
    printv("Model restored from: %s" % name, verbosity, 1)
    return saver

"""Summaries"""
def add_scalar_summaries(vs, name_fn = id):
    for v in vs:
        tf.summary.scalar(name_fn(v.op.name), v)

def add_trainable_summaries(vs = None):
    if vs == None:
        vs = tf.trainable_variables()
    for var in vs:
        tf.summary.histogram(var.op.name, var)

#suggestion: grads is output of `opt.compute_gradients(total_loss)`
def add_gradient_summaries(grads):
    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)


def add_avg(vs, averages=None, decay=0.9, name_fn = lambda s: s+' (average)'):
    if averages==None:
        averages = tf.train.ExponentialMovingAverage(decay, name='avg')
    averages_op = averages.apply(vs)
    add_scalar_summaries(vs, name_fn)
    return averages_op

"""Batch feeder"""
class BatchFeeder(object):
  """
  A BatchFeeder has a next_batch_fun, which takes the batch_size as an argument and returns a *dictionary* (?) "variable" : batch_for_that_variable.
  It keeps index and epochs_completed as internal state.
  `args` can keep the entire array in memory (the most basic batch feeder keeps the entire array in memory).
  """
  def __init__(self, args, num_examples, next_batch_fun):
      self.args = args
      self.index = 0
      self.num_examples = num_examples
      self.epochs_completed = 0
      self.next_batch_fun = next_batch_fun

  def next_batch(self, batch_size, *args):
      return self.next_batch_fun(self, batch_size, *args)

"""
Takes a batch feeder and shuffles its args (data).
"""
def shuffle_refresh(bf):
    perm = np.arange(bf.num_examples)
    np.random.shuffle(perm)
    bf.args = {k : v[perm] for (k,v) in bf.args.items()}

"""
Creates the most basic batch feeder function:

* goes through examples, returns [index, index+batch_size) batch
* increments index by batch_size
* refresh every epoch (default: shuffle)
"""
def batch_feeder_f(bf, batch_size, refresh_f = shuffle_refresh):
    #if bf.xs == None:
    #    refresh_f(bf)
    start = bf.index
    bf.index += batch_size
    # for simplicity, discard those at end.
    if bf.index > bf.num_examples:
      # Finished epoch
      bf.epochs_completed += 1
      # Shuffle the data
      refresh_f(bf)
      # Start next epoch
      start = 0
      bf.index = batch_size
      assert batch_size <= bf.num_examples
    end = bf.index
    return { k : v[start:end] for (k,v) in bf.args.items()}

"""
Make the simplest batch feeder, with the batch feeder function above.
"""
def make_batch_feeder(args, refresh_f=shuffle_refresh, num_examples = None):
    if num_examples==None:
        l = len(list(args.values())[0]) #fix for python 3
    else:
        l = num_examples
    return BatchFeeder(args, l, (lambda bf, batch_size: batch_feeder_f(bf, batch_size, refresh_f)))

"""
AddOn for Trainer
"""
class AddOn:
    def __init__(self):
        pass
    def init(self, trainer):
        return True
    def run(self, trainer):
        return True

class GlobalStep(AddOn):
    def __init__(self):
        pass
    def init(self, trainer):
        global_step = tf.Variable(0, trainable=False)
        trainer.data['global_step'] = global_step
        return True
    def run(self, trainer):
        return True

class SummaryWriter(AddOn):
    def __init__(self, summary_steps = 100, feed_dict = {}):
        self.summary_steps = summary_steps
        self.feed_dict = feed_dict
    def init(self, trainer):
        self.summary_op = tf.summary.merge_all()
        trainer.data['summary_op'] = self.summary_op
        self.summary_writer = tf.summary.FileWriter(trainer.train_dir, trainer.sess.graph)
        if trainer.verbosity>=2:
            print("All nodes:")
            for n in tf.get_default_graph().as_graph_def().node: # trainer.sess.graph?
                print(n.name)
        printv("Summary writer initialized.", trainer.verbosity, 1)
        #Note: doing this every step might be slower... Also it gets the dropout wrong...
        #trainer.gets_dict['summary'] = self.summary_op
        return True
    def run(self, trainer):
        if valid_pos_int(self.summary_steps) and trainer.step % self.summary_steps == 0:
            printv("Running summary...", trainer.verbosity, 1)
            #feed_dict??
            mod_feed_dict = merge_two_dicts(trainer.feed_dict, fill_feed_dict(self.feed_dict))
            summary_str = trainer.sess.run(self.summary_op, feed_dict=mod_feed_dict)
            trainer.outputs['summary'] = summary_str
            #summary_str = trainer.sess.run(self.summary_op, feed_dict=self.feed_dict) #???
            #summary_str = trainer.outputs['summary']
            self.summary_writer.add_summary(summary_str, trainer.step)
        return True

#Save every `save_steps`
class Saver(AddOn):
    def __init__(self, save_steps = 1000, checkpoint_path = 'model.ckpt'):
        self.save_steps = save_steps
        self.checkpoint_path = checkpoint_path
    def init(self, trainer):
        self.saver = tf.train.Saver()
        trainer.data['saver'] = self.saver
        return True
    def run(self, trainer):
        step = trainer.step 
        if (valid_pos_int(self.save_steps) and step % self.save_steps == 0) or (step + 1) == trainer.max_step:
            checkpoint_path = os.path.join(trainer.train_dir, self.checkpoint_path)
            printv("Saving as %s" % checkpoint_path, trainer.verbosity, 1)
            self.saver.save(trainer.sess, self.checkpoint_path, global_step=step)
        return True

class Train(AddOn):
    def __init__(self, optimizer, batch_size, loss = 'loss', train_feed={}, print_steps = 1):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.loss = loss
        self.train_feed = train_feed
        self.print_steps=print_steps
    def init(self, trainer):
        trainer.gets_dict['loss'] = trainer.data[self.loss]
        deps = get_with_default(trainer.data, 'grad_deps', [])
        # Compute gradients.
        with tf.control_dependencies(deps):
            #must compute loss_averages_op before executing this---Why?
            opt = self.optimizer(trainer.data['global_step']) if 'global_step' in trainer.data else self.optimizer()
            grads = opt.compute_gradients(trainer.data[self.loss])
        trainer.data['grads']=grads
        apply_gradient_op = opt.apply_gradients(grads, global_step = trainer.data['global_step']) if 'global_step' in trainer.data else opt.apply_gradients(grads)
        trainer.data['apply_gradient_op'] = apply_gradient_op
        trainer.deps.append(apply_gradient_op)
        return True
    def run(self, trainer):
        start_time = time.time()
        feed_dict = fill_feed_dict2(trainer.train_data, self.batch_size, trainer.args_pl, self.train_feed)
        trainer.feed_dict = feed_dict
        #map_feed_dict(merge_two_dicts(
        #  fill_feed_dict(trainer.train_data, self.batch_size, 
        #                 trainer.args_pl), self.train_feed))
        #print(feed_dict)
        #print(trainer.args_pl)
        #print(self.train_feed)
        #_, loss_value = trainer.sess.run([trainer.train_op, trainer.data[self.loss]], feed_dict=feed_dict)
        trainer.outputs = trainer.sess.run(trainer.gets_dict, feed_dict=feed_dict)
        loss_value = trainer.outputs['loss']
        duration = time.time() - start_time
        if (valid_pos_int(self.print_steps) and trainer.step % self.print_steps == 0):
            printv("Step %d took %f seconds. Loss: %f" % (trainer.step, duration, loss_value), trainer.verbosity, 1)
        if np.isnan(loss_value):
            print('Model diverged with loss = NaN')
            return False
        return True
"""
if ((valid_pos_int(eval_steps) and (step + 1) % (eval_steps) == 0)) or (eval_steps!=None and (step + 1) == max_steps):
        for (data, name) in zip([train_data,validation_data,test_data], ["Training", "Validation", "Test"]):
            if data!=None:
                printv('%s Data Eval:' % name, verbosity, 1)
                do_eval(sess,
                        funcs["accuracy"],
                        data,
                        batch_size,
                        args_pl,
                        batch_feeder_args,
                        eval_feed)
"""

class Eval(AddOn):
    def __init__(self, batch_feeder, batch_size, metrics, eval_feed={}, eval_steps = 1000, name=""):
        self.batch_feeder = batch_feeder
        self.batch_size = batch_size
        #self.loss = loss
        self.metrics = metrics
        self.eval_feed = eval_feed
        self.eval_steps = eval_steps
        self.name = name
    def init(self, trainer):
        pass
    def run(self, trainer):
        step = trainer.step
        if (valid_pos_int(self.eval_steps) and 
            valid_pos_int(step) and
            ((step % self.eval_steps == 0) or (step + 1) == trainer.max_step)):
            printv("Doing evaluation for %s" % self.name, trainer.verbosity, 1)
            do_eval(trainer.sess, 
                    trainer.data,
                    self.metrics, 
                    self.batch_feeder,
                    self.batch_size,
                    args_pl = trainer.args_pl,
                    eval_feed = self.eval_feed,
                    verbosity = trainer.verbosity)
        return True

def do_eval(sess,
            data,
            metrics,
            batch_feeder, 
            batch_size,
            args_pl={},
            args=[],
            eval_feed={},
            verbosity=1):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    metrics: Dictionary of tensors that hold metrics to be evaluated
    batch_feeder: Feeder that gives e.g. set of xs and ys to evaluate.
    args_pl: Mapping from keys in batch_feeder to variable placeholder names
    args: Extra arguments required by batch_feeder
    eval_feed: Further arguments 
  """
  # And run one epoch of eval.
  l = len(metrics)
  ms = [data[m] for m in metrics]
  metrics_total = l*[0]
  steps_per_epoch = batch_feeder.num_examples // batch_size
  #FIX this requires whole number now...
  #num_examples = steps_per_epoch * batch_size
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict2(batch_feeder, batch_size, args_pl, feed = eval_feed, args = args)
    #feed_dict = fill_feed_dict(batch_feeder, batch_size, args_pl, args = args)
    #feed_dict.update(map_feed_dict(eval_feed))
    curs = sess.run(ms, feed_dict=feed_dict)
    for i in range(l):
        metrics_total[i] += curs[i]
  metrics_avg = [float(x)/float(steps_per_epoch) for x in metrics_total]
  for (name, m) in zip(metrics, metrics_avg):
      printv('%s: %f' % (name, m), verbosity, 1)
  return metrics_avg
  #printv('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
  #       (num_examples, true_count, precision), verbosity, 1)


class Trainer:
    def __init__(self, model, max_step, train_data, addons, args_pl, train_dir = "train/", verbosity=1, sess=None): #test_bf=None, valid_bf=None, verbosity=1):
        self.addons = addons
        #self.model = model
        self.data = model
        self.max_step = max_step
        self.step = 0
        self.verbosity = verbosity
        self.train_data = train_data
        self.train_dir = train_dir
        #self.test_bf = test_bf
        #self.valid_bf = valid_bf
        self.deps = []
        self.train_op = None
        self.args_pl = args_pl
        self.sess = sess
        self.gets_dict = {}
        self.outputs = {}
    def init(self):
        # create session
        if self.sess == None:
            self.sess = tf.Session()
        for addon in self.addons:
            addon.init(self)
            #ex. create global step
        printv("Initializing variables.", self.verbosity, 1)
        #init = tf.initialize_all_variables()
        with tf.control_dependencies(self.deps):
            train_op = tf.no_op(name='train')
        self.train_op = train_op #deprecated - access using gets_dict instead 
        self.gets_dict['train_op'] = train_op
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op) #do this last
    def train_step(self):
        for addon in self.addons:
            t = addon.run(self)
            if t == False:
                return False
        self.step = self.step + 1
        return True
    def train(self):
        while self.step < self.max_step:
            t = self.train_step()
            if t==False:
                return 
    def init_and_train(self):
        self.init()
        self.train()

class Histograms(AddOn):
    def __init__(self):
        pass
    def init(self, trainer):
        add_trainable_summaries()
        add_gradient_summaries(trainer.data['grads'])
        return True
    def run(self, trainer):
        return True

class TrackAverages(AddOn): #fix: add in vars
    def __init__(self, averages=None, decay=0.9, name_fn = lambda s: s+' (average)'): #vs=[]
        #self.vs = vs
        self.averages = averages
        self.decay = decay
        self.name_fn = name_fn
    def init(self, trainer):
        op = add_avg(tf.get_collection("losses"), self.averages, self.decay, self.name_fn)
        if 'grad_deps' in trainer.data:
            trainer.data['grad_deps'].append(op)
        else:
            trainer.data['grad_deps'] = [op]
        return True
    def run(self, trainer):
        return True

class AddOnFromFun(AddOn):
    def __init__(self, init_fn, run_fn):
        self.init_fn = init_fn
        self.run_fn = run_fn
    def init(self, trainer):
        return self.init_fn(trainer)
    def run(self, trainer):
        return self.run_fn(trainer)

def fill_feed_dict2(batch_feeder, batch_size, args_pl=None, feed = {}, args = []):
    feed_dict = fill_feed_dict(batch_feeder, batch_size, args_pl, args)
    for k in feed:
        if k in args_pl:
            feed_dict[args_pl[k]] = feed[k]
    #feed_dict.update({args_pl[k] : feed[k] for k in feed if k in args_pl.items()})
    return feed_dict

def fill_feed_dict(batch_feeder, batch_size=None, args_pl=None, args = []):
  """Fills the feed_dict for training the given step. Args should be a list.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }"""
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  b = batch_feeder.next_batch(batch_size, *args)  if batch_size != None else {}
  if args_pl != None:
      return {args_pl[k] : b[k] for (k,v) in args_pl.items() if k in b}
  else:
      return b

def map_feed_dict(feed_dict):
    #print(feed_dict)
    return map_keys(lambda x: tf.get_default_graph().get_tensor_by_name(x) if isinstance(x,str) else x, feed_dict)

#y_ is actual labels, y is predicted probabilities
def accuracy(y_, y):
    correct_prediction = tf.equal(tf.argmax(y,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def accuracy2(y_,y):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

"""
BATCH_SIZE = 100

sample_addon_list = [GlobalStep(),
                     TrackAverages(), #do this before train (why?)
                     Train(lambda gs: tf.train.AdamOptimizer(1e-4), 100),
                     Histograms(), #includes gradients, so has to be done after train
                     Saver(save_steps = 1000, checkpoint_path = 'model.ckpt'),
                     SummaryWriter(summary_steps = 100),
                     Eval(train_data, BATCH_SIZE, ['accuracy'], eval_feed={}, eval_steps = 1000, name="training"),
                     Eval(test_data, BATCH_SIZE, ['accuracy'], eval_feed={}, eval_steps = 1000, name="test")]
"""                  

"""
Misc
"""
def valid_pos_int(n):
    return n!=None and n>0

"""
https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
"""
