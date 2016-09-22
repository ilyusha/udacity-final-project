import h5py, os, sys, random
from collections  import defaultdict
from PIL          import Image 
import cPickle    as pickle
from flags        import FLAGS

directories = {
  "train":      FLAGS.svhn_train_dir,
  "test":       FLAGS.svhn_test_dir,
  "extra":      FLAGS.svhn_extra_dir
}

pickle_files = {
  "train":       "train_metadata.pickle",
  "test":        "test_metadata.pickle",
  "extra":       "extra_metadata.pickle",
}

_data = {}
_data_by_digit = {}

def _load_data(dataset):
  pickle_file = pickle_files[dataset]
  with open(pickle_file) as f:
    data = pickle.load(f)
  return data

def _get_data(dataset):
  if dataset not in _data:
    print 'loading SVHN data from dataset "%s"'%dataset
    d = _load_data(dataset)
    _data[dataset] = d
  return _data[dataset]
  
def get_image_data(dataset, img_num):
  d = _get_data(dataset)
  labels = ('top', 'left', 'height', 'width', 'label')
  return [d[label][img_num - 1] for label in labels]

def get_label(dataset, img_num):
  return get_image_data(dataset, img_num)[-1]

def get_digit_bboxes(dataset, img_num):
  bboxes = []
  for top,left,height,width, _ in zip(*get_image_data(dataset, img_num)):
    bboxes.append((left, top, left+width, top+height))
  return bboxes

def get_bbox(dataset, img_num):
  tops,lefts,heights,widths,_ = get_image_data(dataset, img_num)
  uppermost = min(tops)
  leftmost  = min(lefts)
  rightmost = max((l+w) for l,w in zip(lefts, widths))
  tallest   = max((t+h) for t,h in zip(tops, heights))
  return leftmost, uppermost, rightmost, tallest

def expand_bbox(bbox, percent_expand):
  left, upper, width, height = bbox
  x_pad  = width * percent_expand
  y_pad  = height * percent_expand
  left   = max(0, left - x_pad / 2)
  upper  = max(0, upper - y_pad / 2)
  width  = width + x_pad / 2
  height = height + x_pad / 2
  return left, upper, width, height
  
def process_image(image, img_num, dataset, padding, resize):
  image = image.convert("L")
  bbox = get_bbox(dataset, img_num)
  if padding > 0:
    bbox = expand_bbox(bbox, padding)
  cropped = image.crop(bbox)
  if resize:
    return cropped.resize(resize)
  else:
    return cropped

def process_image_by_number(dataset, image_dir, img_num, padding=0.3, resize=(64,64)):
  img_path = os.path.join(image_dir, str(img_num)+".png")
  img = Image.open(img_path)
  return process_image(img, img_num, dataset, padding, resize)

def process_image_by_file(dataset, path, padding=0.3, resize=(64,64)):
  img_dir, filename = os.path.split(path)
  img_num = int(filename.split(".")[0])
  return process_image_by_number(dataset, img_dir, img_num, padding=padding, resize=resize)

def _get_data_by_digit(dataset):
  '''populate an index of instances of each digit in an SVHN dataset'''
  if dataset not in _data_by_digit:
    by_digit = defaultdict(list)
    data = _get_data(dataset)
    for img_idx, image_data in enumerate(zip(*data.values()),1):
      bbox = image_data[:4]
      digits = image_data[4]
      for digit_idx, digit in enumerate(digits):
        if digit == 10: digit = 0
        by_digit[int(digit)].append((img_idx, digit_idx))
    _data_by_digit[dataset] = by_digit
  return _data_by_digit[dataset]


def get_random_digit(digit,dataset="train"):
  '''pick a random version of a digit from a dataset and return its cropped image'''
  data = _get_data(dataset)
  digit_data = _get_data_by_digit(dataset)
  img_idx, digit_idx = random.choice(digit_data[digit])
  image_dir = directories[dataset]
  image = Image.open(os.path.join(image_dir, "%s.png"%img_idx))
  bbox = get_digit_bboxes(dataset, img_idx)[digit_idx]
  cropped = image.crop(bbox)
  return cropped

if __name__ == "__main__":
  cmd, dataset = sys.argv[1:]
  
  image_dir = directories[dataset]
  
  if cmd == "parse":
    datafile  = os.path.join(image_dir, "digitStruct.mat")
    matfile = h5py.File(pickle_files[dataset])

    data = defaultdict(list)

    def parse(name, obj):
      vals = []
      if obj.shape[0] == 1:
        vals.append(obj[0][0])
      else:
        for k in range(obj.shape[0]):
          vals.append(matfile[obj[k][0]][0][0])
      data[name].append(vals)

    print "reading data from",datafile

    for struct in matfile["/digitStruct/bbox"]:
      group = matfile[struct[0]]
      group.visititems(parse)

    print "pickling data...",
    pck = open(pickle_file, 'wb')
    pickle.dump(data, pck, pickle.HIGHEST_PROTOCOL)
    pck.close()
    print "done"
