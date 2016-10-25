from __future__ import division
from math       import sqrt
import tensorflow as tf
import numpy      as np
from flags        import FLAGS
import base


class ConvNet(base.Model):
  '''
  Configurable convolutional network.

  Parameters that control the layers in the network architecture are all lists of integers:

   - filter_sizes: [5,5,5] would build three 5x5 convolutions
   - filter_depths: [4,8,16] would make those convolutions have depths of 4, 8, and 16 respectively
   - fc_layer_sizes: [1000, 1000] would add two fully-connected layers
   - pooling_layers: [1,3] would add max pooling to convolution layers 1 and 3 
   - dropout_layers: [2,3] would add dropout to convolution layers 2 and 3
  
  Other network parameters:

   - conv_keep_prob: keep probability of convolutional layer dropout
   - fc_keep_prob: keep probability of fully-connected layer dropout
   - L2: whether to add L2 regularization the cost function
   - L2_beta: L2 beta parameter

  '''

  def __init__(self, input_size, num_labels, **kwargs):

    self.filter_sizes   = kwargs.get("filter_sizes",   [])
    self.filter_depths  = kwargs.get("filter_depths",  [])
    self.fc_layer_sizes = kwargs.get("fc_layer_sizes", [])
    self.pooling_layers = kwargs.get("pooling_layers", [])
    self.dropout_layers = kwargs.get("dropout_layers",[])
    self.conv_keep_prob = kwargs.get("conv_keep_prob", 1.0)
    self.fc_keep_prob   = kwargs.get("fc_keep_prob", 1.0)
    self.L2             = kwargs.get("L2", False)
    self.L2_beta        = kwargs.get("L2_beta", 0.0005)
    self.vars_to_save   = []
    self.l2_vars        = []
    self.input_img_dim  = int(sqrt(input_size))
    self.fc_dropout     = None
    self.conv_dropout   = None
    base.Model.__init__(self, input_size, num_labels, **kwargs)

  def _iterate_fc_layers(self, input_size):
    for layer_idx, output_size in enumerate(self.fc_layer_sizes, 1):
      yield (layer_idx, input_size, output_size)
      input_size = output_size

  def _iterate_conv_sizes(self):
    for idx, (filter_size, filter_depth) in enumerate(zip(self.filter_sizes, self.filter_depths), 1):
      yield idx, filter_size, filter_depth


  def build_nn(self, input_layer):
    conv_layers  = self.add_conv_layers(input_layer)
    fc_layers    = self.add_fc_layers(conv_layers)
    output_layer = self.add_output_layer(fc_layers)
    return output_layer

  def add_conv_layers(self, input_layer):
    reshaped = tf.reshape(input_layer, shape=[-1, self.input_img_dim, self.input_img_dim, 1])
    cur_layer = reshaped
    input_depth = 1

    #iterate over convolutional layers
    for idx, size, depth in self._iterate_conv_sizes():

      with tf.name_scope("conv_layer_%s"%idx):
        #define variables

        with tf.name_scope("weight"):
          weight = tf.Variable(tf.truncated_normal([size, size, input_depth, depth], stddev=0.1))
        with tf.name_scope("bias"):
          bias = tf.Variable(tf.zeros([depth]))

        #convolve and add bias
        with tf.name_scope("convolution"):
          convolution = self._conv(cur_layer, weight)

        #add activation
        with tf.name_scope("activated"):
          activated = tf.nn.relu(convolution + bias)

      #add dropout if necessary
      if self.conv_dropout is not None and idx in self.dropout_layers:
        with tf.name_scope("dropout"):
          activated = tf.nn.dropout(activated, self.conv_dropout)

      #add pooling if necessary
      if idx in self.pooling_layers:
        with tf.name_scope("max_pool"):
          activated = self._max_pool(activated)

      cur_layer = activated
      input_depth = depth
      self.vars_to_save.extend([weight, bias])

    return cur_layer

  def add_fc_layers(self, input_layer):
    #reshape output of last convolution layer to feed it into fully-connected layers
    _, h, w, d = input_layer.get_shape().as_list()
    reshaped_dim = h * w * d
    cur_layer = tf.reshape(input_layer, [-1, reshaped_dim])
    #iterate over fully-connected layers
    for idx, in_size, out_size in self._iterate_fc_layers(reshaped_dim):

      with tf.name_scope("fc_layer_%s"%idx):
        with tf.name_scope("weight"):
          fc_weight = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        with tf.name_scope("bias"):
          fc_bias = tf.Variable(tf.zeros([out_size]))
        with tf.name_scope("linear"):
          cur_layer = tf.matmul(cur_layer, fc_weight) + fc_bias
        with tf.name_scope("activated"):
          cur_layer = tf.nn.relu(cur_layer)
      if self.fc_dropout is not None:
        with tf.name_scope("dropout"):
          cur_layer = tf.nn.dropout(cur_layer, self.fc_dropout)
      self.vars_to_save.extend([fc_weight, fc_bias])
      self.l2_vars.extend([fc_weight, fc_bias])
    return cur_layer

  def add_output_layer(self, input_layer):

    with tf.name_scope("output_layer"):
      with tf.name_scope("weight"):
        out_weight = tf.Variable(tf.truncated_normal([self.fc_layer_sizes[-1], self.num_labels]))
      with tf.name_scope("bias"):
        out_bias = tf.Variable(tf.zeros([self.num_labels]))
      with tf.name_scope("linear"):
        output = tf.matmul(input_layer, out_weight) + out_bias

    self.vars_to_save.extend([out_weight, out_bias])
    self.l2_vars.extend([out_weight, out_bias])
    return output

  def _conv(self, tensor, filter_tensor):
    with tf.name_scope("convolution"):
      return tf.nn.conv2d(tensor, filter_tensor, strides=[1,1,1,1], padding="SAME")

  def _max_pool(self, tensor):
    with tf.name_scope("max_pool"):
      maxpool = tf.nn.max_pool(tensor, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    return maxpool

  def init_variables(self):

    with tf.name_scope("input"):
      self.X = tf.placeholder(tf.float32, [None, self.input_size])

    with tf.name_scope("output"):
      self.Y = tf.placeholder(tf.float32, [None, self.num_labels])

    if self.fc_keep_prob < 1.0:
      with tf.name_scope("fc_dropout"):
        self.fc_dropout = tf.placeholder(tf.float32)

    if self.conv_keep_prob < 1.0 and self.dropout_layers:
      with tf.name_scope("conv_dropout"):
        self.conv_dropout = tf.placeholder(tf.float32)

    self.predict_tensor = self.build_nn(self.X)

    with tf.name_scope("loss"):
      with tf.name_scope("cross_entropy"):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predict_tensor, self.Y))
      if self.L2:
        with tf.name_scope("L2"):
          l2_reg = tf.nn.l2_loss(self.l2_vars[0])
          for var in self.l2_vars[1:]:
            l2_reg += tf.nn.l2_loss(var)
        self.cost += self.L2_beta * l2_reg

  def get_training_tensor(self):
    return self.train_tensor

  def get_predict_tensor(self):
    return self.predict_tensor

  def get_cost_tensor(self):
    return self.cost

  def get_vars_to_save(self):
    return self.vars_to_save

  def get_train_feed_dict(self, batch_x, batch_y):
    d = {self.X: batch_x, self.Y: batch_y}
    if self.fc_dropout   is not None: d[self.fc_dropout]   = self.fc_keep_prob
    if self.conv_dropout is not None: d[self.conv_dropout] = self.conv_keep_prob
    return d

  def get_test_feed_dict(self, x_test, y_test):
    d = {self.X: x_test, self.Y: y_test}
    if self.fc_dropout   is not None: d[self.fc_dropout]   = 1.0
    if self.conv_dropout is not None: d[self.conv_dropout] = 1.0
    return d

  def get_classify_feed_dict(self, x_classify):
    d= {self.X: x_classify}
    if self.fc_dropout   is not None: d[self.fc_dropout]   = 1.0
    if self.conv_dropout is not None: d[self.conv_dropout] = 1.0
    return d

class MultiLogitDigitRecognizer(ConvNet):

  '''subclass of ConvNet that trains six models simultaneously: 
     - one for the number of digits in a number
     - one for each of the five possible digits.
  '''

  def __init__(self, **kwargs):
    self.num_logits = 6
    input_size = FLAGS.img_width * FLAGS.img_height
    num_labels = 11
    ConvNet.__init__(self, input_size, num_labels, **kwargs)

  def add_output_layer(self, input_layer):
    logits = []
    for logit in range(self.num_logits):
      with tf.name_scope("logit_%s"%(logit+1,)):
        with tf.name_scope("weight"):
          out_weight = tf.Variable(tf.truncated_normal([self.fc_layer_sizes[-1], self.num_labels]))
        with tf.name_scope("bias"):
          out_bias = tf.Variable(tf.zeros([self.num_labels]))
        with tf.name_scope("linear"):
          logit = tf.matmul(input_layer, out_weight) + out_bias
          logits.append(logit)
      self.vars_to_save.extend([out_weight, out_bias])
      self.l2_vars.extend([out_weight, out_bias])
    return logits

  def init_variables(self):

    with tf.name_scope("train_input"):
      self.X = tf.placeholder(tf.float32, [None, self.input_size])

    with tf.name_scope("train_output"):
      self.Y = tf.placeholder(tf.int64, [None, self.num_logits])

    if self.fc_keep_prob < 1.0:
      with tf.name_scope("fc_dropout"):
        self.fc_dropout = tf.placeholder(tf.float32)

    if self.conv_keep_prob < 1.0 and self.dropout_layers:
      with tf.name_scope("conv_dropout"):
        self.conv_dropout = tf.placeholder(tf.float32)

    self.logits = self.build_nn(self.X)
    costs = []
    with tf.name_scope("loss"):
      for idx,logit in enumerate(self.logits):
        x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit, self.Y[:,idx])
        costs.append(tf.reduce_mean(x_entropy))
      self.cost = tf.add_n(costs)
      if self.L2:
        with tf.name_scope("L2"):
          l2_reg = tf.nn.l2_loss(self.l2_vars[0])
          for var in self.l2_vars[1:]:
            l2_reg += tf.nn.l2_loss(var)
        self.cost += self.L2_beta * l2_reg

  def init_accuracy(self):
    with tf.name_scope("accuracies"):
      self.accuracies = tf.pack(self.logits)
      with tf.name_scope("correct_prediction"):
        argmax = tf.argmax(self.accuracies, 2)
        #the argmax of each row in the i'th logit needs to equal the i'th value in Y.
        #a correct prediction is one where everything matches.
        predict_matrix = tf.cast(tf.equal(tf.transpose(argmax), self.Y), tf.int64)
        #add the actual lengths of the numbers, so that we can calculate accuracy conditional on the number of digits
        length = tf.reshape(self.Y[:,0], (tf.shape(self.Y)[0], 1))
        concatenated = tf.concat(1, [length, predict_matrix])
        #processing of this matrix is done is interpret_accuracy
        return concatenated

  def merge_accuracies(self, accuracies, **kwargs):
    epoch = kwargs.get("epoch")
    #first element of each tuple is the accuracy data returned by interpret_accuracy
    accuracy_data  = [acc[0] for acc in accuracies] 
    #second element of each tuple is the batch size
    num_total      = sum(acc[1] for acc in accuracies) 
    all_correct    = sum(acc[0] for acc in accuracy_data)
    length_correct = sum(acc[1] for acc in accuracy_data)
    by_length      = [acc[2] for acc in accuracy_data]
    by_digit       = [acc[3] for acc in accuracy_data]
    n_correct_sums = [sum(l) for l in zip(*by_length)]
    nth_digit_sums = [sum(l) for l in zip(*by_digit)]

    accuracy = all_correct / num_total

    length_accuracy = length_correct / num_total
    print "length accuracy: {:.1%}".format(length_accuracy)
    self.train_writer.add_summary(self.sess.run(tf.scalar_summary("length-accuracy", length_accuracy)), epoch)

    for idx, n in enumerate(n_correct_sums, 1):
      percent_n_or_more_correct = n / num_total
      print "{} or more digit correct: {:.1%}".format(idx, percent_n_or_more_correct)
      self.train_writer.add_summary(self.sess.run(tf.scalar_summary("n-or-more-accuracy_%s"%idx, percent_n_or_more_correct)), epoch)

    for idx, n in enumerate(nth_digit_sums, 1):
      percent_nth_digit_correct = n / num_total
      print "digit {} accuracy: {:.1%}".format(idx, percent_nth_digit_correct)
      self.train_writer.add_summary(self.sess.run(tf.scalar_summary("nth-digit-accuracy_%s"%idx, percent_nth_digit_correct)), epoch)

    return accuracy

  def interpret_accuracy(self, accuracy_tensor, **kwargs):
    
    lengths = accuracy_tensor[:,0]
    correct_predictions = accuracy_tensor[:,1:].astype(bool)
    
    def _correct_prediction(row):
      '''
      Helper function that returns True if the first n+1
      values in a vector are true, starting at index 1,
      where n is the first element in the vector.
      [2, True, True, True, True]   -> True (every logit is correct)
      [2, True, True, True, False]  -> True (last logit is wrong, but that's OK)
      [2, True, True, False, False] -> False (got the length right but the second digit wrong)
      [2, False, True, True, True]  -> False (got the length wrong)
      '''

      actual_length = row[0]
      return np.all(row[1:actual_length+2])

    all_correct = sum(np.apply_along_axis(_correct_prediction, 1, accuracy_tensor))
    
    correct_length = sum(correct_predictions[:,0])

    correct_by_length = []
    correct_by_digit  = []
    for i in range(1, self.num_logits):
      correct_by_length.append(sum(np.sum(correct_predictions[:,1:], axis=1) >= i))
      correct_by_digit.append(sum(correct_predictions[:,i]))
    
    return all_correct, correct_length, correct_by_length, correct_by_digit

  def run_prediction(self, data):
    '''
    Overrides default method. Since we are evaluating six logits,
    we run the six predictions sequentially and then average their confidences.
    '''

    predictions = []
    confidences = []
    for logit in self.logits:
      prediction = self.sess.run(logit, feed_dict = self.get_classify_feed_dict(data))
      predictions.append(np.argmax(prediction))
      confidences.append(np.max(self._softmax(prediction)))
    confidence = np.mean(confidences)
    return predictions, confidence

  def process_label(self, classification):
    length = classification[0]
    digits = []
    for digit in classification[1:1+length]:
      if digit == 0:
        continue
      elif digit == 10:
        digits.append(0)
      else:
        digits.append(digit)
    return "".join(map(str, digits))


class NumberLocator(ConvNet):
  '''This subclass of ConvNet is a binary classifier that is trained to 
     identify whether a provided image contains a number'''

  def __init__(self, input_size, **kwargs):
    ConvNet.__init__(self, input_size, 2, **kwargs)

  def init_accuracy(self):
    with tf.name_scope("accuracy"):
      predicted = tf.argmax(self.get_predict_tensor(), 1)
      actual    = tf.argmax(self.Y, 1)
      combined  = tf.pack([predicted, actual]) #will analyze this outside of tf in interpret_accuracy
      return combined
      
  def interpret_accuracy(self, accuracy_tensor, **kwargs):
    #unpack columns packed in init_accuracy
    predicted = accuracy_tensor[0]
    actual    = accuracy_tensor[1]

    num_no_number  = np.sum(actual == 0)
    num_has_number = np.sum(actual == 1)

    no_number_correct  = np.sum((actual == 0) & (predicted == 0))
    has_number_correct = np.sum((actual == 1) & (predicted == 1))
    
    return no_number_correct, num_no_number, has_number_correct, num_has_number

  def merge_accuracies(self, accuracies, **kwargs):
    epoch = kwargs.get("epoch")
    no_number_correct   = sum(acc[0][0] for acc in accuracies)
    total_no_number     = sum(acc[0][1] for acc in accuracies)
    has_number_correct  = sum(acc[0][2] for acc in accuracies)
    total_has_number    = sum(acc[0][3] for acc in accuracies)
    num_total           = sum(acc[1] for acc in accuracies) 
    no_number_accuracy  = no_number_correct / total_no_number
    has_number_accuracy = has_number_correct / total_has_number
    print "no number: {:.1%}, has number: {:.1%}".format(total_no_number / num_total, total_has_number / num_total)
    print "no number accuracy: {:.1%}".format(no_number_accuracy)
    print "has number accuracy: {:.1%}".format(has_number_accuracy)
    self.train_writer.add_summary(self.sess.run(tf.scalar_summary("no-number",  no_number_accuracy)), epoch)
    self.train_writer.add_summary(self.sess.run(tf.scalar_summary("has-number", has_number_accuracy)), epoch)
    return (no_number_correct + has_number_correct) / num_total