from os            import path, listdir
from PIL           import Image
from random        import random, randrange
from itertools     import chain, izip
from collections   import defaultdict
from generate      import generate_random_svhn_composite
from image         import generate_image_patches, extract_data_from_image
from flags         import FLAGS
from batch         import *
import numpy       as np
import tensorflow  as tf
import svhn


class DataSource(object):
  '''Abstract class used to access train and test BatchIterators'''
  
  def get_train_data(self, batch_size):
    '''Return a BatchIterator that yields training data'''
    raise NotImplementedError

  def get_test_data(self, batch_size):
    '''Return a BatchIterator that yields test data'''
    raise NotImplementedError


class MNISTDataSource(DataSource):
  '''DataSource that loads MNIST data into memory and passes it to the InMemoryBatchIterator'''

  def __init__(self):
    DataSource.__init__(self)
    from tensorflow.examples.tutorials.mnist import input_data
    self._mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
  def get_train_data(self, batch_size):
    return InMemoryBatchIterator(self._mnist.train.images, self._mnist.train.labels, batch_size)

  def get_test_data(self, batch_size):
    return InMemoryBatchIterator(self._mnist.test.images, self._mnist.test.labels, batch_size)


class SyntheticPerDigitDataSource(DataSource):
  '''DataSource that loads synthetic data to be used by the per-digit classifier'''

  def __init__(self, digit):
    DataSource.__init__(self)
    from inputs import load_sample_digit_dataset
    self._train_x, self._train_y, self._test_x, self._test_y = load_sample_digit_dataset(digit)
    
  def get_train_data(self, batch_size):
    return InMemoryBatchIterator(self._train_x, self._train_y, batch_size)
  
  def get_test_data(self, batch_size):
    return InMemoryBatchIterator(self._test_x, self._test_y, batch_size)


class SyntheticLengthDataSource(DataSource):
  '''DataSource that loads synthetic data to be used by the length classifier'''

  def __init__(self):
    DataSource.__init__(self)
    from inputs import load_sample_length_dataset
    self._train_x, self._train_y, self._test_x, self._test_y = load_sample_length_dataset()
    
  def get_train_data(self, batch_size):
    return InMemoryBatchIterator(self._train_x, self._train_y, batch_size)
  
  def get_test_data(self, batch_size):
    return InMemoryBatchIterator(self._test_x, self._test_y, batch_size)


class SyntheticJointDataSource(DataSource):
  '''DataSource that loads synthetic data to be used by the joint classifier'''

  def __init__(self):
    DataSource.__init__(self)
    from sample_digit_data import load_combined_dataset
    self._train_x, self._train_y, self._test_x, self._test_y = load_combined_dataset()

  def get_train_data(self, batch_size):
    return InMemoryBatchIterator(self._train_x, self._train_y, batch_size)
  
  def get_test_data(self, batch_size):
    return InMemoryBatchIterator(self._test_x, self._test_y, batch_size)


class SVHNDataSource(DataSource):
  '''
  Primary DataSource used by the number classifier.
  Capable of encoding data for a single-digit classifier, length classifier, or multi-logit classifier.
  Can undersample/oversample by the number of digits.'''

  def __init__(self, img_dirs, digit=None, length_only=False, undersample=None, oversample=None):
    self.digit       = digit
    self.length_only = length_only
    self._filter     = lambda f: "png" in f and "_processed" not in f
    self.img_dirs    = img_dirs
    self.oversample_specs  = oversample
    self.undersample_specs = undersample

  def get_train_data(self, batch_size):
    iterators = []
    for img_dir in self.img_dirs:
      file_iterator = SVHNBatchIterator(img_dir,
                                        batch_size, 
                                        digit            = self.digit, 
                                        length_only      = self.length_only,
                                        undersample_dict = self.undersample_specs,
                                        file_filter      = self._filter)
      iterators.append(file_iterator)

    if self.oversample_specs:
      iterators.append(CompositeSVHNBatchIterator(batch_size, self.oversample_specs))

    return JointBatchIterator(batch_size, iterators)  
    

  def get_test_data(self, batch_size):
    return SVHNBatchIterator(svhn.TEST_IMAGE_DIR, batch_size, 
                             digit       = self.digit, 
                             length_only = self.length_only,
                             file_filter = self._filter)


class SyntheticSVHNDataSource(DataSource):
  '''DataSource that yields synthetic SVHN images based on the given specs'''

  def __init__(self, train_specs, test_specs):
    self.train_specs = train_specs
    self.test_specs  = test_specs

  def get_train_data(self, batch_size):
    return CompositeSVHNBatchIterator(batch_size, self.train_specs, cache_id=1)

  def get_test_data(self, batch_size):
    return CompositeSVHNBatchIterator(batch_size, self.test_specs, cache_id=2)


class NumberRecognizerDataSource(DataSource):
  '''DataSource that yields train/test data for training the number locator'''

  def __init__(self, train_dataset, train_range, test_dataset, test_range):
    self.train_dataset = train_dataset
    self.test_dataset  = test_dataset
    self.train_range   = train_range
    self.test_range    = test_range

  def get_train_data(self, batch_size):
    return ImagePatchIterator(batch_size, self.train_dataset, self.train_range)

  def get_test_data(self, batch_size):
    return ImagePatchIterator(batch_size, self.test_dataset, self.test_range)

