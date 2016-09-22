import tensorflow       as tf
from models.multilayer  import MLP
from models.convolution import ConvNet, MultiLogitDigitRecognizer
from flags              import FLAGS

'''
Various configurations of models used to learn from the SVHN dataset
'''


def get_svhn_digit_model(digit):
  model = ConvNet(input_size        = FLAGS.img_height * FLAGS.img_width,
                  num_labels        = 10,
                  fc_layer_sizes    = [1000],
                  fc_keep_prob      = 0.33,
                  filter_sizes      = [5,5,1],
                  filter_depths     = [16,32,32],
                  pooling_layers    = [1,2,3],
                  optimizer         = tf.train.AdamOptimizer(),
                  cutoff            = 0,
                  max_epochs        = 10,
                  L2                = False,
                  batch_size        = 500,
                  save_file         = "svhn_digit_%s"%digit,
                  name              = "SVHN digit %s classifier"%digit)
  return model

def get_svhn_length_model():
  model = ConvNet(input_size        = FLAGS.img_height * FLAGS.img_width,
                  num_labels        = 5,
                  fc_layer_sizes    = [1000],
                  fc_keep_prob      = 0.33,
                  filter_sizes      = [5,5,1],
                  filter_depths     = [16,32,32],
                  pooling_layers    = [1,2,3],
                  optimizer         = tf.train.AdamOptimizer(),
                  cutoff            = 0,
                  max_epochs        = 10,
                  L2                = False,
                  batch_size        = 500,
                  save_file         = "svhn_length",
                  name              = "SVHN length classifier")
  return model


def get_svhn_joint_v1():
  model = MultiLogitDigitRecognizer(
                            fc_layer_sizes    = [1000],
                            fc_keep_prob      = 0.5,
                            filter_sizes      = [10,10,1],
                            filter_depths     = [32,64,32],
                            pooling_layers    = [1,2,3],
                            optimizer         = tf.train.AdamOptimizer(),
                            cutoff            = 0,
                            max_epochs        = 25,
                            L2                = True,
                            test_batch_size   = 500,
                            batch_size        = 250,
                            save_file         = "svhn_joint_v1",
                            name              = "SVHN MultiLogitConvNet classifier")
  return model

def get_svhn_joint_v2():
  model = MultiLogitDigitRecognizer(
                            fc_layer_sizes    = [1000],
                            fc_keep_prob      = 0.5,
                            filter_sizes      = [3,3,1,3,3,1],
                            filter_depths     = [8,16,32,32,64,64],
                            pooling_layers    = [3,4,5,6], #4 pooling layers get it down to 4x4, leading to 16*64=1024 output nodes
                            optimizer         = tf.train.AdamOptimizer(),
                            cutoff            = 0,
                            max_epochs        = 25,
                            L2                = True,
                            test_batch_size   = 500,
                            batch_size        = 250,
                            save_file         = "svhn_joint_v2",
                            name              = "SVHN MultiLogitConvNet classifier")
  return model


def get_svhn_joint_v3():
  model = MultiLogitDigitRecognizer(
                            fc_layer_sizes    = [1000],
                            fc_keep_prob      = 0.5,
                            filter_sizes      = [3,3,1,3,3,1],
                            filter_depths     = [8,16,32,32,64,64],
                            pooling_layers    = [3,4,5,6], #4 pooling layers get it down to 4x4, leading to 16*64=1024 output nodes
                            optimizer         = tf.train.AdamOptimizer(),
                            cutoff            = 0,
                            max_epochs        = 25,
                            L2                = True,
                            test_batch_size   = 500,
                            batch_size        = 250,
                            save_file         = "svhn_joint_v3",
                            name              = "SVHN MultiLogitConvNet classifier")
  return model

def get_svhn_joint_v4():
  model = MultiLogitDigitRecognizer(
                            fc_layer_sizes    = [5000],
                            fc_keep_prob      = 0.33,
                            conv_keep_prob    = 0.8,
                            filter_sizes      = [3,3,1,3,3,1],
                            filter_depths     = [8,16,32,32,64,64],
                            pooling_layers    = [3,4,5,6], #4 pooling layers get it down to 4x4, leading to 16*64=1024 output nodes
                            dropout_layers    = [1,2,3,4,5,6],
                            optimizer         = tf.train.AdamOptimizer(),
                            cutoff            = 0,
                            max_epochs        = 25,
                            L2                = True,
                            test_batch_size   = 500,
                            batch_size        = 250,
                            save_file         = "svhn_joint_v4",
                            name              = "SVHN MultiLogitConvNet classifier")
  return model


def get_svhn_joint_v5():
  '''this is the final model used'''
  model = MultiLogitDigitRecognizer(
                            fc_layer_sizes    = [1000, 1000],
                            fc_keep_prob      = 0.5,
                            filter_sizes      = [5,5,5,5,1],
                            filter_depths     = [8,16,32,64,32],
                            pooling_layers    = [2,3,4,5], #4 pooling layers get it down to 4x4, leading to 16*64=1024 output nodes
                            optimizer         = tf.train.AdamOptimizer(),
                            cutoff            = 0,
                            max_epochs        = 25,
                            L2                = True,
                            test_batch_size   = 500,
                            batch_size        = 250,
                            save_file         = "svhn_5",
                            name              = "SVHN MultiLogitConvNet classifier")
  return model  
