import tensorflow as tf
from inputs import MNISTDataSource
from classifier.convolution import ConvNet


def train_mnist():
  datasource = MNISTDataSource()
  conv = ConvNet(input_size     = 28*28,
                 num_labels     = 10,
                 fc_layer_sizes = [512],
                 fc_keep_prob   = 0.5,
                 filter_sizes   = [5, 5, 1],
                 filter_depths  = [32, 64, 64],
                 pooling_layers = [1, 2],
                 save_dir       = "classifiers",
                 optimizer      = tf.train.AdamOptimizer(),
                 cutoff         = 0,
                 batch_size     = 100,
                 L2             = True,
                 save_file      = "mnist")

  conv.train(datasource)

if __name__ == "__main__":
  train_mnist()
