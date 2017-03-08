
* `mnist_tutorial_tf.py` is copied from [Cleverhans tutorial](https://github.com/openai/cleverhans/blob/master/tutorials/mnist_tutorial_tf.md).
* `mnist_tuturial_tf2.py`
    * `mnist_tuturial_tf2.py` is hacky. I can't figure out how to use tf or tf-slim instead of keras for the adversarial training because of variable reuse problems. I don't know why I have to do this hack here but not in `adversarial.py` 
