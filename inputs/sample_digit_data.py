import numpy         as np
from functools       import partial
from PIL             import Image
from flags           import FLAGS
import random
import os


'''

This module is used for loading MNIST-based synthetic data, by digit position.

TODO: reduce code duplication with sample_length_data.py

'''

def _iter_dir(d, filter_func=None):
  if not filter_func: filter_func = lambda path: True
  return filter(filter_func, map(partial(os.path.join, d), os.listdir(d)))

def _valid_img_path(path):
  filename = os.path.split(path)[1]
  return not filename.startswith(".")

def get_image_data(filepath):
  image = Image.open(filepath).convert('L')
  data = np.fromiter(iter(image.getdata()), np.float32)
  return data

def get_label_data(filepath):
  filename = os.path.split(filepath)[1]
  digit = int(filename.split("_")[0])
  arr = np.zeros(10)
  arr[digit] = 1
  return arr

def _shuffle(data, labels):
  zipped = zip(data, labels)
  random.shuffle(zipped)
  return zip(*zipped)

def _partition_train_test(data, labels, percent_test):
  print "partitioning data into train and test sets"
  data, labels = _shuffle(data, labels)
  idx = int(len(data) * (1-percent_test))
  data_train, data_test = data[:idx], data[idx:]
  labels_train, labels_test = labels[:idx], labels[idx:]
  return data_train, labels_train, data_test, labels_test

def load_sample_digit_dataset(digit_pos, percent_test=0.2):
  img_dir = os.path.join(FLAGS.generated_img_dir, "by_digit", str(digit_pos))
  data_list  = []
  label_list = []
  img_paths = _iter_dir(img_dir, filter_func = _valid_img_path)
  num_images = len(img_paths)
  for idx, img_path in enumerate(img_paths, 1):
    label_list.append(get_label_data(img_path))
    data_list.append(get_image_data(img_path))
    if not idx%500:
      print "loaded %s/%s sample images"%(idx, num_images)

  X_train, Y_train, X_test, Y_test = _partition_train_test(data_list, label_list, percent_test)

  data_train  = np.array(X_train, dtype="float32") / 255
  label_train = np.array(Y_train, dtype="float32")
  data_test   = np.array(X_test, dtype="float32") / 255
  label_test  = np.array(Y_test, dtype="float32")

  return data_train, label_train, data_test, label_test


def get_label(filepath):
  filename = os.path.split(filepath)[1]
  _, length, number, _ = filename.split("_")
  length = int(length)
  digits = map(int, number)
  labels = np.full([6], 10, dtype=np.int32)
  labels[0] = length
  for i, digit in enumerate(digits, 1):
    labels[i] = digit
  return labels

def normalize_image(image_data):
  return (image_data - np.mean(image_data)) / 255

def shuffle(data, labels):
  conc = np.concatenate([data, labels], axis=1) #shuffle together
  np.random.shuffle(conc)
  return conc[:,:-6], conc[:,-6:]

def train_test_split(data, labels, percent_test=0.2):
  idx = int(len(data) * (1-percent_test))
  data_train, data_test = data[:idx], data[idx:]
  labels_train, labels_test = labels[:idx], labels[idx:]
  return data_train, labels_train, data_test, labels_test

def load_combined_dataset(percent_test=0.2):
  all_data   = []
  all_labels = []
  for digit_dir in _iter_dir(os.path.join(FLAGS.generated_img_dir, "by_digit")):
    img_paths = _iter_dir(digit_dir, filter_func = _valid_img_path)
    num_images = len(img_paths)
    print "loading %s images from %s"%(num_images, digit_dir)
    data = np.zeros([num_images, 56*56])
    labels = np.zeros([num_images, 6], dtype=np.int32)
    for idx, img_path in enumerate(img_paths):
      data[idx] = normalize_image(get_image_data(img_path))
      labels[idx] = get_label(img_path)
    all_data.append(data)
    all_labels.append(labels)
  data, labels = np.concatenate(all_data), np.concatenate(all_labels)
  data, labels = shuffle(data, labels)
  return train_test_split(data, labels, percent_test = percent_test)



