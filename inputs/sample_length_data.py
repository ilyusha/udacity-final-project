import numpy      as np
from functools    import partial
from PIL          import Image
from flags        import FLAGS
import random
import os

'''

This module is used for loading MNIST-based synthetic data, by number length.

TODO: reduce code duplication with sample_digit_data.py

'''
def _iter_dir(d):
  return map(partial(os.path.join, d), os.listdir(d))

def _valid_img_path(path):
  filename = os.path.split(path)[1]
  return not filename.startswith(".")

def get_image_data(filepath):
  image = Image.open(filepath).convert('L')
  data = np.fromiter(iter(image.getdata()), np.float32)
  return data

def get_label_data(filepath, num_labels):
  path = os.path.split(filepath)[0]
  length = path.split("/")[-1]
  arr = np.zeros(num_labels)
  arr[int(length) - 1] = 1
  return arr

def add_directory_to_arrays(directory, data_list, label_list, num_labels):
  paths = _iter_dir(directory)
  num_images = len(paths)
  for img_path in paths:
    if not _valid_img_path(img_path): continue
    label_list.append(get_label_data(img_path, num_labels))
    data_list.append(get_image_data(img_path))

def _shuffle(data, labels):
  zipped = zip(data, labels)
  random.shuffle(zipped)
  return zip(*zipped)

def _partition_train_test(data, labels, percent_test):
  data, labels = _shuffle(data, labels)
  idx = int(len(data) * (1-percent_test))
  data_train, data_test = data[:idx], data[idx:]
  labels_train, labels_test = labels[:idx], labels[idx:]
  return data_train, labels_train, data_test, labels_test

def load_sample_length_dataset(percent_test=0.2):

  data_list  = []
  label_list = []
  dirs = _iter_dir(os.path.join(FLAGS.generated_img_dir, "by_length"))
  num_labels = len(dirs)
  for d in dirs:
    add_directory_to_arrays(d, data_list, label_list, num_labels)

  X_train, Y_train, X_test, Y_test = _partition_train_test(data_list, label_list, percent_test)

  data_train  = np.array(X_train, dtype="float32") / 255
  label_train = np.array(Y_train, dtype="float32")
  data_test   = np.array(X_test, dtype="float32") / 255
  label_test  = np.array(Y_test, dtype="float32")

  return data_train, label_train, data_test, label_test
