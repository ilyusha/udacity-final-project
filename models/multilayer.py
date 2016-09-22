import tensorflow as tf
import numpy      as np
import base

class MLP(base.Model):
  '''
  Class implementing a Multilayer Perceptron with optional dropout.

  The number and sizes of hidden layers is configurable via the hidden_layer_sizes parameter,
  which is a list of integers. For example, [10, 20, 10] would result in an MLP with three
  hidden layers, each with 10, 20 and 10 neurons, respectively.
  '''

  def __init__(self, input_size, num_labels, hidden_layer_sizes, **kwargs):
    base.Model.__init__(self, input_size, num_labels, **kwargs)
    self.layer_sizes  = hidden_layer_sizes
    self.keep_prob    = kwargs.get("keep_prob", 0.7)
    self.vars_to_save = []
    self.dropout = None

  def _build_nn(self, input_layer):
    cur_layer = input_layer
    for idx, size_in, size_out, is_output in self._iterate_layer_sizes():
      layer_name = "layer_%s"%idx
      with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
          weights = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
          self.add_summaries(weights, layer_name + '/weights')
        with tf.name_scope("biases"):
          biases  = tf.Variable(tf.constant(0.1, shape=[size_out]))
          self.add_summaries(biases, layer_name + '/biases')
        if idx > 1 and self.dropout is not None:
          with tf.name_scope('dropout'):
            cur_layer = tf.nn.dropout(cur_layer, self.dropout)
        with tf.name_scope('linear'):
          linear = tf.matmul(cur_layer, weights) + biases
          tf.histogram_summary(layer_name + '/linear', linear)
        if is_output:
          cur_layer = linear
        else:
          with tf.name_scope("activated"):
            activated = self.activation(linear)
            tf.histogram_summary(layer_name + '/activated', activated)
          cur_layer = activated

      self.vars_to_save.extend([weights, biases])

    return cur_layer

  def activation(self, layer):
    return tf.nn.relu(layer)

  def _iterate_layer_sizes(self):
    layer_idx = 1
    #first layer
    yield (layer_idx, self.input_size, self.layer_sizes[0], False)
    prev_layer_size = self.layer_sizes[0]
    #hidden layers
    for i, layer_size in enumerate(self.layer_sizes[1:], 1):
      layer_idx += 1
      yield (layer_idx, prev_layer_size, layer_size, False)
      prev_layer_size = layer_size
    #output layer
    layer_idx += 1
    yield (layer_idx, self.layer_sizes[-1], self.num_labels, True)

  def init_variables(self):
    with tf.name_scope("input"):
      self.X = tf.placeholder(tf.float32, [None, self.input_size])

    with tf.name_scope("output"):
      self.Y = tf.placeholder(tf.float32, [None, self.num_labels])

    if self.keep_prob < 1.0:
      with tf.name_scope("dropout"):
        self.dropout = tf.placeholder(tf.float32)

    self.predict_tensor = self._build_nn(self.X)

    with tf.name_scope("cross_entropy"):
      self.cross_entropy  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predict_tensor, self.Y))
      tf.scalar_summary('cross entropy', self.cross_entropy)

    
  def get_training_tensor(self):
    return self.train_tensor

  def get_predict_tensor(self):
    return self.predict_tensor

  def get_cost_tensor(self):
    return self.cross_entropy

  def get_vars_to_save(self):
    return self.vars_to_save

  def get_train_feed_dict(self, batch_x, batch_y):
    d = {self.X: batch_x, self.Y: batch_y}
    if self.dropout is not None: d[self.dropout] = self.keep_prob
    return d

  def get_test_feed_dict(self, x_test, y_test):
    d = {self.X: x_test, self.Y: y_test}
    if self.dropout is not None: d[self.dropout] = 1.0
    return d

  def get_classify_feed_dict(self, x_classify):
    d= {self.X: x_classify}
    if self.dropout is not None: d[self.dropout] = 1.0
    return d
