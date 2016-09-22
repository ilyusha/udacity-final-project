import tensorflow as tf
from base import Model

class LogisticClassifier(Model):
  '''
  Class implementing a simple logistic classifier
  '''

  def __init__(self, *args, **kwargs):
    Model.__init__(self, *args, **kwargs)

  def init_variables(self):
    with tf.name_scope("weight"):
      self.weight_tensor = tf.Variable(tf.zeros([self.input_size, self.num_labels], dtype="float32"))
    with tf.name_scope("bias"):
      self.bias_tensor = tf.Variable(tf.zeros([self.num_labels], dtype="float32"))
    with tf.name_scope("linear"):
      self.prediction_tensor = tf.add(tf.matmul(self.X, self.weight_tensor), self.bias_tensor)
    with tf.name_scope("cross_entropy"):
      self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction_tensor, self.Y))
    
  def get_training_tensor(self):
    return self.train_tensor

  def get_predict_tensor(self):
    return self.prediction_tensor

  def get_cost_tensor(self):
    return self.cross_entropy

  def get_vars_to_save(self):
    return [self.weight_tensor, self.bias_tensor]

