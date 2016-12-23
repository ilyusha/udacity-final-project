from os            import path, listdir
from PIL           import Image
from random        import random, randrange
from itertools     import chain, izip
from collections   import defaultdict
from generate      import generate_random_svhn_composite
from image         import generate_image_patches, extract_data_from_image
from flags         import FLAGS
import numpy      as np
import tensorflow as tf
import svhn

'''This module contains BatchIterators, which implement the iterator protocol to generate batched data'''

class BatchIterator(object):
  """
  This class is used to yield data in batches of specified size.
  Subclasses implement the next_item() method, as well as set the
  data_size attribute.
  """

  def __init__(self, batch_size):
    self._batch_size = batch_size
    self.data_size = None

  @property
  def batch_size(self):
    if self._batch_size is None:
      if self.data_size is None:
        raise Exception("unknown batch size and data size")
      else:
        return self.data_size
    else:
      return self._batch_size

  def get_num_batches(self):
    if not self.data_size: return None
    num_batches = self.data_size / self.batch_size
    if self.data_size%self.batch_size > 0:
      num_batches+=1 #remainder
    return num_batches

  def __iter__(self):
    self.item_iter  = self.next_item()
    self.batch_iter = self.next_batch()
    self.cur_batch  = []
    return self
  
  def __len__(self):
    return self.data_size

  def next(self):
    return self.batch_iter.next()

  def next_item(self):
    raise NotImplementedError

  def next_batch(self):
    try:
      while True:
        next_item = self.item_iter.next()
        self.cur_batch.append(next_item)
        if len(self.cur_batch) == self.batch_size:
          yield self.finalize_batch()
          self.cur_batch = []
    except StopIteration:
      if self.cur_batch:
        yield self.finalize_batch()
        self.cur_batch = []
      raise StopIteration

  def finalize_batch(self):
    data, labels = zip(*self.cur_batch)
    return np.array(data), np.array(labels)


class JointBatchIterator(BatchIterator):
  '''Batch iterator that chains together multiple iterators'''

  def __init__(self, batch_size, iterators):
    BatchIterator.__init__(self, batch_size)
    self.iterators = iterators
    self.chained_iter = chain(*[i.next_item() for i in iterators])
    self.data_size = sum(len(i) for i in iterators)

  def next_item(self):
    for item in self.chained_iter:
      yield item


class InMemoryBatchIterator(BatchIterator):
  '''Batch iterator that yields from a pre-loaded dataset'''

  def __init__(self, data, labels, batch_size):
    BatchIterator.__init__(self, batch_size)
    self.data      = data
    self.labels    = labels
    self.data_size = len(self.data)
    
    if not batch_size:
      self._batch_size  = len(data)
    else:
      self._batch_size  = batch_size
    
  def next_item(self):
    for data, labels in izip(self.data, self.labels):
      yield data, labels


class FileBatchIterator(BatchIterator):
  '''abstract batch iterator that loads files from a specified directory'''

  def __init__(self, directory, batch_size, file_filter=None):
    BatchIterator.__init__(self, batch_size)
    self.directory = directory
    if not file_filter:
      self.filter = lambda f: True
    else:
      self.filter = file_filter
    self.files = filter(self.filter, listdir(self.directory))
    self.data_size = len(self.files)
    self.cur_batch = None


class SVHNEncoder(object):
  '''helper class that implements methods to encode SVHN labels into vectors'''

  def __init__(self, digit_pos=None, length_only=False):
    self.digit_pos   = digit_pos
    self.length_only = length_only

  def encode_onehot_digit(self, number, digit_pos):
    if len(number) < digit_pos:
      return
    digit_encoding = np.zeros([10], dtype="int")
    digit = number[digit_pos - 1]
    #dataset has zeros labeled as '10'
    if digit == 10: digit = 0 
    digit_encoding[int(digit)] = 1
    return digit_encoding

  def encode_onehot_length(self, length):
    length_encoding = np.zeros([5], dtype="int")
    length_encoding[length - 1] = 1
    return length_encoding

  def encode_joint(self, number):
    joint_encoding = np.full([6], 10, dtype="int")
    joint_encoding[0] = len(number)
    for i, digit in enumerate(number, 1):
      if int(digit) == 10:
        digit = 0
      joint_encoding[i] = int(digit)
    return joint_encoding

  def encode(self, number):
    if self.length_only:
      return self.encode_onehot_length(len(number))
    elif self.digit_pos is not None:
      return self.encode_onehot_digit(number, self.digit_pos)
    else:
      return self.encode_joint(number)


class SVHNBatchIterator(FileBatchIterator, SVHNEncoder):
  '''Batch iterator that yields image data from one of the three SVHN data directories (train, test, extra).

     The data can encode SVHN data in three ways:
      1. length-only: encodes the number of digits in the number in a one-hot vector
      2. by digit: encodes the number in the given digit position in a one-hot vector
      3. joint: encode the length and each of the five digit positions in a single vector

     This iterator also has the ability to undersample data based on the length of the number'''

  def __init__(self, directory, batch_size, digit=None, length_only=False, file_filter=None, undersample_dict=None):
    FileBatchIterator.__init__(self, directory, batch_size, file_filter=file_filter)
    SVHNEncoder.__init__(self, digit_pos=digit, length_only = length_only)
    self.dataset_label = path.split(directory)[1] #train, test, extra
    self.digit         = digit
    self.length_only   = length_only
    #undersampling specification. {2: 0.9} would mean that numbers with 2 digits are kept with 90% probability
    self.undersample_dict = {} if not undersample_dict else undersample_dict

  def next_item(self):
    for filename in self.files:
      try:
        img_num = int(filename.split(".")[0])
      except ValueError:
        continue #invalid filename
      digits = svhn.get_label(self.dataset_label, img_num)
      if len(digits) > 5: 
        #skip outliers with more than 5 digits
        continue 
      #if the number length is listed in the undersampling dict, drop the number with the given probability
      keep_prob = self.undersample_dict.get(len(digits), 1)
      if keep_prob < 1 and random() > keep_prob:
        continue
      labels = self.encode(digits)
      if labels is None: 
        # could happen if one-hot encoding a digit of a number that is too short
        continue

      image = Image.open(path.join(self.directory, filename))
      processed = svhn.process_image(image, img_num, self.dataset_label, 
                                     padding=0.3, resize=(FLAGS.img_width,FLAGS.img_height))
      data = extract_data_from_image(processed)

      yield data, labels


class SyntheticSVHNBatchIterator(BatchIterator, SVHNEncoder):
  '''Class that generates composite images from SVHN data on the fly.
     This is used to oversample house numbers of length 4 and 5, as data on them
     is sparse..
  
     Generated image data is cached under the specified cache_id, so that multiple
     runs will yield the same results
  '''
  
  generated_img_cache = {}
  
  def __init__(self, batch_size, generate_specs, cache_id=1):
    
    BatchIterator.__init__(self, batch_size)
    SVHNEncoder.__init__(self)
    self.generate_specs = generate_specs  
    self.data_size      = sum(self.generate_specs.values())
    self.cache_id       = cache_id

    if self.cache_id not in CompositeSVHNBatchIterator.generated_img_cache:
      CompositeSVHNBatchIterator.generated_img_cache[self.cache_id] = defaultdict(dict)
    self.cache = CompositeSVHNBatchIterator.generated_img_cache[self.cache_id]

  def next_item(self):
    for n_digits, n_to_generate in self.generate_specs.iteritems():
      for i in range(n_to_generate):
        cached = self.cache[n_digits].get(i)
        if not cached:
          image, number = generate_random_svhn_composite(n_digits)
          converted = image.resize((FLAGS.img_height,FLAGS.img_width)).convert("L")
          data   = extract_data_from_image(converted)
          labels = self.encode(number)
          self.cache[n_digits][i] = (data, labels)
        else:
          data, labels = cached
        yield data, labels


class ImagePatchIterator(BatchIterator):
  '''iterator class used to iterate over patches of SVHN images to generate positive/negative
     samples for the classifier meant to locate numbers in an arbitrary image'''

  def __init__(self, batch_size, img_dataset, img_range):
    BatchIterator.__init__(self, batch_size)
    self.dataset = img_dataset
    self.img_range = img_range

  def next_item(self):
    for img_num in range(*self.img_range):
      img  = Image.open(path.join(svhn.directories[self.dataset], "%s.png"%img_num))
      bbox = svhn.get_bbox(self.dataset, img_num)
      #expand the bounding box by 30%, since that's what the number classifier is trained on
      bbox = svhn.expand_bbox(bbox, 0.3)
      left, upper, width, height = bbox
      #generate negative examles by sliding a window of the same size as the bounding box
      #over the image, avoiding the known number location
      skip_range = ((left, width), (upper, height))
      window_dim = width - left, height - upper
      step_x     = window_dim[0]
      step_y     = window_dim[1]
      num_negative_samples = 0
      for _ , no_number in generate_image_patches(img, window_dim=window_dim, step_x=step_x, step_y=step_y, skip_range=skip_range):
        num_negative_samples += 1
        no_number = no_number.resize((FLAGS.img_width, FLAGS.img_height)).convert("L")
        data  = extract_data_from_image(no_number)
        #one-hot encoding of "image does not have number"
        label = np.array([1, 0])
        yield data, label

      has_number = img.crop(bbox).resize((FLAGS.img_width, FLAGS.img_height)).convert("L")
      data  = extract_data_from_image(has_number)
      #one-hot encoding of "image has number"
      label = np.array([0, 1])
      yield data, label