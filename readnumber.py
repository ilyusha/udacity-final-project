import numpy            as np
import tensorflow       as tf
from models.logistic    import LogisticClassifier
from models.multilayer  import MLP
from models.convolution import ConvNet, MultiLogitDigitRecognizer
from inputs.datasource  import *
from numberlocator      import find_number_in_image, get_locator_model
from flags              import FLAGS
from image              import draw_bbox
from PIL                import Image
from svhn_models        import *
from synthetic_models   import *
import sys

def get_best_svhn_model():
  '''manually updated - returns the current best SVHN joint model'''
  return get_svhn_joint_v5()

def train_synthetic_joint_classifier():
  model = get_synthetic_joint_v1()
  data  = SyntheticJointDataSource()
  model.train(data)

def train_synthetic_digit_classifier(digit):
  model = get_synthetic_digit_model(digit)
  data  = SyntheticPerDigitDataSource(digit)
  model.train(data)

def train_synthetic_length_classifier():
  model = get_synthetic_length_model()
  data  = SyntheticLengthDataSource()
  model.train(data)

def train_svhn_digit_classifier(digit):
  model = get_svhn_digit_model(pos)
  data = SVHNDataSource(digit=digit)
  model.train(data)

def train_svhn_length_classifier():
  model = get_svhn_length_model()
  data = SVHNDataSource(length_only=True)
  model.train(data)

def train_svhn_joint_classifier():
  model = get_best_svhn_model()
  data = SVHNDataSource(img_dirs    = [FLAGS.svhn_extra_dir, FLAGS.svhn_train_dir],
                        #adjust dataset to 50k samples of each length
                        #undersample = {2: 0.5, 3: 0.5},
                        #oversample  = {1: 25000, 2: 12000, 3:22000, 4: 29000, 5:30000}
                        )

  model.train(data)


def locate_and_read_number(image, locator_model, classifier_model, bbox=False):
  '''
  This function locates all potential numbers in an image, run them through the specified
  classifier, and returns the longest result with the highest confidence
  '''
  found_numbers = find_number_in_image(locator_model, image, draw_steps=False)
  if not found_numbers:
      return
  predictions = []
  for bbox in found_numbers:
      patch = image.crop(bbox)
      prediction, prediction_confidence = classifier_model.classify_image(patch)
      predictions.append((prediction, bbox, prediction_confidence))
  max_prediction_length = len(max(predictions, key = lambda x: len(x[0]))[0])
  acceptable_lengths = (max_prediction_length, max_prediction_length - 1)
  sorted_by_confidence = sorted([i for i in predictions if len(i[0]) in acceptable_lengths], key = lambda x: x[2], reverse=True)
  best_prediction, bbox, _ = sorted_by_confidence[0]
  if bbox:
    draw_bbox(image, bbox)
  return best_prediction


if __name__ == "__main__":

  if FLAGS.train:
    svhn = FLAGS.svhn
    synthetic = FLAGS.synthetic
    if (svhn and synthetic) or (not svhn and not synthetic):
      print "must specify one of 'svhn' or 'synthetic'"
      sys.exit(0)
    if svhn:
      if FLAGS.joint:
        train_svhn_joint_classifier()
      elif FLAGS.digit:
        train_svhn_digit_classifier(int(FLAGS.digit))
      elif FLAGS.length:
        train_svhn_length_classifier()
    else:
      if FLAGS.joint:
        train_synthetic_joint_classifier()
      elif FLAGS.digit:
        train_synthetic_digit_classifier(FLAGS.digit)
      elif FLAGS.length:
        train_synthetic_length_classifier()
  elif FLAGS.classify:
    locator_model = get_locator_model()
    number_model  = get_best_svhn_model()
    image = Image.open(FLAGS.classify)
    print locate_and_read_number(image, locator_model, number_model, bbox=True)
    image.show()

