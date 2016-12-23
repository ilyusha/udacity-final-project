from __future__ import division
from PIL        import Image, ImageDraw
from random     import randrange
import numpy    as np

"""
Module containing some functions for image manipulation
"""

def extract_data_from_image(image):
  '''return a numpy array containing normalized pixel values for the image'''
  data = np.fromiter(iter(image.getdata()), np.float32)
  data = (data - np.mean(data)) / 255
  return data

def _get_steps(length, batch_length, step_size):
  offset = 0
  yield offset, batch_length
  while offset + batch_length != length:
    if offset + batch_length + step_size >= length:
      offset = length - batch_length #move up to the edge instead of going over
    else:
      offset += step_size #move one step farther
    yield offset, offset + batch_length

def _filter_steps(iterator, skip_range):  
  if not skip_range:
    for start, stop in iterator: yield start, stop, False
  else:
    skip_range_min, skip_range_max = skip_range
    for start, stop in iterator:
      in_range = stop > skip_range_min and start < skip_range_max
      yield start, stop, in_range
  

def slide_window(image_dim, window_dim, step_x, step_y, skip_range=None):
  '''function that generates coordinates for a sliding window of the given size, 
     over the provided image dimensions, given x and y step values.
     skip_range is an optional parameter in the form of ((x-range), (y-range)) that, if provided,
     will not overlap with any of the windows'''
  x_img, y_img = image_dim
  x_window, y_window = window_dim
  if not skip_range:
    for y_offset, bottom_edge in _get_steps(y_img, y_window, step_y):
      for x_offset, right_edge, in _get_steps(x_img, x_window, step_x):
        yield x_offset, y_offset, right_edge, bottom_edge
  else:
    for y_offset, bottom_edge, in_range_y in _filter_steps(_get_steps(y_img, y_window, step_y), skip_range[1]):
      for x_offset, right_edge, in_range_x in _filter_steps(_get_steps(x_img, x_window, step_x), skip_range[0]):
        if in_range_x and in_range_y:
          continue
        else:
          yield x_offset, y_offset, right_edge, bottom_edge

  
def draw_bbox(image, bbox, color="red"):
  draw = ImageDraw.Draw(image)
  draw.rectangle(bbox, outline=color)


def bbox_random_shift(image, bbox, shift_percentage):
  left, upper, width, height = bbox
  bbox_width  = width - left
  bbox_height = height - upper 
  max_x_shift = max(int(bbox_width  * shift_percentage), 3)
  max_y_shift = max(int(bbox_height * shift_percentage), 3)
  x_shift = y_shift = 0
  while x_shift == 0:
    x_shift = randrange(-1 * max_x_shift, max_x_shift)
  while y_shift == 0:
    y_shift = randrange(-1 * max_y_shift, max_y_shift)
  new_bbox = left + x_shift, upper + y_shift, width + x_shift, height + y_shift
  return image.crop(new_bbox)

        
def generate_image_patches(image, window_dim, step_x, step_y, skip_range=None, draw=False):
  '''function that uses slide_window to iterate over window locations, and crops the image to each window'''
  for i, box in enumerate(slide_window(image.size, window_dim, step_x, step_y, skip_range=skip_range), 1):
    if draw:
      draw_bbox(image, box)
    yield box, image.crop(box)