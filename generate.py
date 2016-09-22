import sys, os, random
import numpy      as np
from PIL          import Image
from functools    import partial
from shutil       import rmtree
from itertools    import chain
from inputs       import svhn
from flags        import FLAGS
from tensorflow.examples.tutorials.mnist import input_data


'''
Module used for generating synthetic images.
'''

class MNISTImageGenerator(object):
  '''
  This class generates images of multi-digit numbers by stitching together
  digits from the MNIST dataset.
  '''

  def __init__(self):
    self._digit_dict = self._build_digit_dict()
    self.directory = FLAGS.generated_img_dir

  @staticmethod
  def load_mnist():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return mnist

  def _build_digit_dict(self):
    nmist = self.load_mnist()
    digit_dict = {}
    for digit, image in zip(nmist.train.labels, nmist.train.images):
      image = (image * 255).astype(np.uint8)
      image.resize(28, 28)
      digit_dict.setdefault(np.argmax(digit),[]).append(image)
    return digit_dict

  def _join_images(self, image_array_list):

    images = map(partial(Image.fromarray), image_array_list)
    width = height = FLAGS.synthetic_img_width / len(images)
    starting_height = ((FLAGS.synthetic_img_width / 2) - (width / 2))
    mode = images[0].mode
    widths, heights = zip(*(i.size for i in images))
    combined = Image.new(mode, (FLAGS.synthetic_img_width, FLAGS.synthetic_img_height))
    for idx, im in enumerate(images):
      resized = im.resize((width, height), Image.NEAREST)
      combined.paste(resized, (idx*width, starting_height))
    return combined

  def build_image(self, number_str, rotate=False):
    img_data_array = []
    for s in number_str:
      img_data = random.choice(self._digit_dict[int(s)])
      img_data_array.append(img_data)
    image = self._join_images(img_data_array)
    if rotate: image = image.rotate(random.randrange(0, 360))
    return image

  @staticmethod
  def _gen_number(digit, position, length):
    assert(0 <= digit <= 9)
    assert(position > 0)
    if position > length: raise ValueError("position can't be greater than length")
    numbers = [random.choice(range(10)) for i in range(length)]
    numbers[position - 1] = digit
    return "".join(map(str, numbers))

  def _reset_directory(self, subdir):
    d = os.path.join(self.directory, subdir)
    if os.path.exists(d):
      rmtree(d, ignore_errors=True)
    os.mkdir(d)

  def generate_image_by_length(self, length):
      number = "".join(map(str,[random.choice(range(10)) for i in range(length)]))
      return number, self.build_image(number, rotate=True)

  def generate_image_by_position(self, digit, position):
    length = random.choice(range(position, 6))
    number = self._gen_number(digit, position, length)
    return number, self.build_image(number)

  def generate_sample_images_by_length(self, length, num_images=1):
    print "generating %d numbers with length %d"%(num_images, length)
    _generated_files = {}
    for i in range(num_images):
      number, image = self.generate_image_by_length(length)
      idx = _generated_files[number] = _generated_files.get(number, 0) + 1
      pos_dir = os.path.join(self.directory, "by_length", str(length))
      if not os.path.exists(pos_dir): os.makedirs(pos_dir)
      filename = os.path.join(pos_dir, "%s_%d.png"%(number, idx))
      image.save(filename)

  def generate_sample_images_by_digit_position(self, digit, position, num_images=1):
    pos_dir = os.path.join(self.directory, "by_digit", str(position))
    print "generating %d numbers with %d in position %d in %s"%(num_images, digit, position, pos_dir)
    _generated_files = {}
    for i in range(num_images):
      number, image = self.generate_image_by_position(digit, position)
      length = len(number)
      idx = _generated_files[number] = _generated_files.get(number, 0) + 1
      if not os.path.exists(pos_dir): os.makedirs(pos_dir)
      filename = os.path.join(pos_dir, "%d_%d_%s_%d.png"%(digit, length, number, idx))
      image.save(filename)

  def generate_length_data(self, n):
    self._reset_directory("by_length")
    for length in range(1, 6):
        self.generate_sample_images_by_length(length, num_images=n)

  def generate_number_data(self, n):
    self._reset_directory("by_digit")
    for digit in range(10):
      for position in range(1, 6):
        self.generate_sample_images_by_digit_position(digit, position, num_images=n)

def generate_svhn_number(number):
  '''
  Generates an image of the specified number by finding random instances
  of each digit in the SVHN dataset and stitching them together
  '''
  digits = [svhn.get_random_digit(int(digit)) for digit in str(number)]
  num_digits  = len(digits)
  max_height  = max(digit.height for digit in digits)
  #each digit can have a range of dimensions, so we need to rescale
  rescaled = []
  for digit in digits:
    if digit.height < max_height:
      #scale to somewhere between the max size and 3/4 of the max size
      new_height = random.randrange(max(digit.height, int(max_height * 0.75)), max_height)
      new_width  = int(digit.width * (float(new_height) / digit.height))
      digit = digit.resize((new_width, new_height), Image.NEAREST)
    rescaled.append(digit)  
  gap = random.choice((0, 5))
  total_width = sum(digit.width for digit in rescaled) + gap * (len(rescaled) - 1)
  combined = Image.new(digits[0].mode, (total_width, max_height), color=(128, 128, 128))
  canvas = Image.new(digits[0].mode, (int(combined.width * 1.33), int(combined.height * 1.33)), color=(128, 128, 128))
  h_offset = 0
  for idx, digit in enumerate(rescaled):
    if digit.height == max_height:
      v_offset = 0
    else:
      v_offset = random.choice((0, max_height - digit.height))
    combined.paste(digit, (h_offset, v_offset))
    h_offset += (digit.width + gap)
  canvas.paste(combined, ((canvas.width - combined.width) / 2, (canvas.height - combined.height) / 2))
  return canvas

def generate_random_svhn_composite(num_digits):
  '''
  Generate a random SVHN-based number with the given number of digits
  '''
  number = "".join([str(random.choice(range(10))) for i in range(num_digits)])
  return generate_svhn_number(number), number

if __name__ == "__main__":
  cmd = sys.argv[1]
  n = int(sys.argv[2])
  IG = MNISTImageGenerator()
  if cmd == "by_digit":
    IG.generate_number_data(n)
  elif cmd == "by_length":
    IG.generate_length_data(n)
  else :
    print "specify by_digit or by_length"
