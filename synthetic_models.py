import numpy            as np
import tensorflow       as tf
from models.multilayer  import MLP
from models.convolution import ConvNet, MultiLogitDigitRecognizer
from flags              import FLAGS


def get_synthetic_digit_MLP(digit_pos):
  class MLPDigitModel(MLP):
    def __init__(self, digit_pos, **kwargs):
      self.position = str(digit_pos)
      MLP.__init__(self, input_size=FLAGS.synthetic_img_width * FLAGS.synthetic_img_height,
                         num_labels=10,
                         hidden_layer_sizes=[10000],
                         keep_prob = 1.0,
                         save_file="digit_%s"%digit_pos,
                         **kwargs)

    def __str__(self):
      return "[position %s classifier]"%self.position
  
  return MLPDigitModel()


def get_synthetic_digit_model(digit_pos):
  return ConvNet(input_size        = FLAGS.synthetic_img_width * FLAGS.synthetic_img_height,
                 num_labels        = 10,
                 fc_layer_sizes    = [1000],
                 fc_keep_prob      = 0.5,
                 filter_sizes      = [10,10,1],
                 filter_depths     = [32,64,64],
                 pooling_layers    = [1,2,4],
                 optimizer         = tf.train.AdamOptimizer(),
                 cutoff            = 0,
                 L2                = True,
                 batch_size        = 100,
                 save_file         = "digit_%s"%digit_pos,
                 name              = "synthetic digit classifier")


def get_synthetic_length_model():
  return ConvNet(input_size        = FLAGS.synthetic_img_width * FLAGS.synthetic_img_height,
                 num_labels        = 5,
                 fc_layer_sizes    = [1000],
                 fc_keep_prob      = 0.5,
                 filter_sizes      = [10,10,1],
                 filter_depths     = [32,64,64],
                 pooling_layers    = [1,2,4],
                 optimizer         = tf.train.AdamOptimizer(),
                 cutoff            = 0,
                 L2                = True,
                 batch_size        = 100,
                 save_file         = "synthetic_length",
                 name              = "synthetic length classifier")
  
def get_synthetic_joint_v1():
  return MultiLogitDigitRecognizer(fc_layer_sizes    = [1000],
                                    fc_keep_prob      = 0.33,
                                    filter_sizes      = [10,10,1],
                                    filter_depths     = [32,64,32],
                                    pooling_layers    = [1,2,3],
                                    optimizer         = tf.train.AdamOptimizer(),
                                    cutoff            = 0,
                                    L2                = True,
                                    batch_size        = 100,
                                    test_batch_size   = 500,
                                    max_epochs        = 25,
                                    save_file         = "synthetic_joint_v1",
                                    name              = "MultiLogitConvNet classifier")
