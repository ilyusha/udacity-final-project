from __future__   import division
import numpy      as np
import tensorflow as tf
from functools    import partial
from PIL          import Image
from flags        import FLAGS
import random
import os

class Model(object):
  """
  Base class for tensorflow-powered classifiers.

  Subclasses override methods to implement different network architectures.
  """

  def __init__(self, input_size, num_labels, **kwargs):

    #the dimensionality of the input vector
    self.input_size            = input_size

    #the dimensionality of the output vector
    self.num_labels            = num_labels

    #the label to be used for the save file, as well as for the tensorboard run
    self.run_label             = kwargs.get("save_file", "model")

    #the name of the file the tensors are saved in
    self.save_file             = self.run_label + ".ckpt"

    #the number of traning epochs. however, several options below can
    #cause training to be terminated early.
    self.max_epochs            = kwargs.get("max_epochs", 50)

    #if positive, an epoch-over-epoch accuracy improvement of less tha
    #this value will result in stopping training
    self.improvement_threshold = kwargs.get("cutoff", 0)

    #how many items are passed to each training step.
    #this value is passed to the BatchIterator instance
    self.batch_size            = kwargs.get("batch_size", 100)

    #how many items are passed to each test step.
    #None means all test data will be evaluated at once.
    self.test_batch_size       = kwargs.get("test_batch_size", None)

    #if set, determines how many times the accuracy is allowed to decrease
    #before training is terminated.
    #a value of None removes this condition.
    self.acc_drops_allowed     = kwargs.get("acc_drops_allowed", None)
    self.name                  = kwargs.get("name", self.__class__.__name__)

    #the optimizer to be used for training
    self.optimizer             = kwargs.get("optimizer")

    self.save_dir              = FLAGS.classifier_dir
    tf.reset_default_graph()
    
    #set up the variables in the graph (implementation dependent)
    self.init_variables()

    #instantiate a Session
    self.sess   = tf.Session()

    self.saver  = None
    self.loaded = False

  def init_variables(self):
    '''construct the graph and set any necessary instance variables'''
    raise NotImplementedError

  def get_predict_tensor(self):
    '''returns the tensor that is used for actually classifying an input'''
    raise NotImplementedError

  def get_cost_tensor(self):
    '''return the tensor that represents the cost we are aiming to minimize'''
    raise NotImplementedError

  def get_vars_to_save(self):
    '''returns a list of tensorflow Variables to be saved for this model'''
    raise NotImplementedError

  def get_train_feed_dict(self, batch_x, batch_y):
    '''return the feed dict for the training operation'''
    return {self.X: batch_x, self.Y: batch_y}

  def get_test_feed_dict(self, x_test, y_test):
    '''return the feed dict used for the testing operation'''
    return {self.X: x_test, self.Y: y_test}

  def get_classify_feed_dict(self, x_classify):
    '''return the feed dict used by classification'''
    return {self.X: x_classify}

  def init_accuracy(self):
    '''in order to allow batch evaluation of test data, we don't actually calculate the percent accuracy here,
    only the number of correct predictions in a batch'''
    with tf.name_scope("accuracy"):
      correct_predictions = tf.equal(tf.argmax(self.get_predict_tensor(), 1), tf.argmax(self.Y, 1))
      return tf.reduce_sum(tf.cast(correct_predictions, tf.float32))

  def init_summaries(self):
    with tf.name_scope("summary"):
      self.total_cost = tf.Variable(0.0)

  def increment_epoch_cost(self, session, batch_cost):
    '''update the running epoch cost variable'''
    session.run(self.total_cost.assign(self.total_cost + tf.constant(batch_cost)))

  def reset_epoch_cost(self):
    '''reset the epoch cost to 0'''
    self.sess.run(self.total_cost.assign(0))

  def _train(self, **kwargs):
    print "training with %s"%self
    session = self.sess
    cost_tensor  = self.get_cost_tensor()

    with tf.name_scope("train"):
      train_tensor = self.optimizer.minimize(cost_tensor)

    print "initializing variables...",
    accuracy_tensor = self.init_accuracy()
    session.run(tf.initialize_all_variables())
    print "done"

    self.init_summaries()
    train_writer = tf.train.SummaryWriter('summaries/train/%s'%(self.run_label,), session.graph)

    self.train_writer = train_writer

    datasource = kwargs.get("data")
    
    prev_epoch_cost = None
    prev_accuracy   = None
    acc_decreased   = 0

    for epoch in range(1, self.max_epochs+1):
      print "\nstarting epoch %s"%epoch

      #initialize the batch iterator
      train_batch_iterator = datasource.get_train_data(self.batch_size)
      num_batches = train_batch_iterator.get_num_batches()

      #reset epoch cost
      self.reset_epoch_cost()

      #iterate over the training batches and run the optimizer
      for batch_number, (batch_x, batch_y) in enumerate(train_batch_iterator, 1):
        if num_batches:
          print "training batch %s/%s"%(batch_number, num_batches)
        else:
          print "training batch %s"%batch_number
        _, batch_cost = session.run([train_tensor, cost_tensor],
                                             feed_dict = self.get_train_feed_dict(batch_x, batch_y))
        
        #increment the current epoch's total cost
        self.increment_epoch_cost(session, batch_cost)

      #get the total epoch cost and send it to the summary writer
      epoch_cost = session.run(self.total_cost)
      print "\ncost at epoch %s: %d"%(epoch, epoch_cost)
      summary = session.run(tf.scalar_summary("cost_train", self.total_cost))
      train_writer.add_summary(summary, epoch)
      
      #train_accuracy_batch_iterator = datasource.get_train_data(self.batch_size * 5)
      test_accuracy_batch_iterator  = datasource.get_test_data(self.test_batch_size)

      #train_accuracy = self.calculate_accuracy(accuracy_tensor, session, train_accuracy_batch_iterator)
      #print "train accuracy at epoch {} is {:.1%}".format(epoch, train_accuracy)
      
      test_accuracy, test_cost = self.calculate_accuracy(accuracy_tensor, session, test_accuracy_batch_iterator, epoch=epoch)
      print "test accuracy at epoch {} is {:.1%}".format(epoch, test_accuracy)
      train_writer.add_summary(session.run(tf.scalar_summary("test_accuracy", test_accuracy)), epoch)
      train_writer.add_summary(session.run(tf.scalar_summary("cost_test", test_cost)), epoch)
      #determine whether or not to terminate training
      if epoch_cost == 0:
        print "total cost is 0 at epoch %s, ending training"%epoch
        break
      if prev_accuracy:
        improvement = test_accuracy - prev_accuracy
        if improvement < 0 and self.acc_drops_allowed is not None:
          acc_decreased += 1
          if acc_decreased < self.acc_drops_allowed:
            print "accuracy is worse than previous epoch! (%s/%s)"%(acc_decreased, self.acc_drops_allowed)
          else:
            print "ending training due to accuracy decreasing %s times in a row"%self.acc_drops_allowed
            break
        else:
          acc_decreased = 0
        if improvement > 0 and improvement < self.improvement_threshold:
          print "improvement of {:.1%} is sufficiently low; ending training".format(improvement)
          break
        else:
          print "improvement of {:.1%} from epoch {}".format(improvement, epoch)
      prev_epoch_cost = epoch_cost
      prev_accuracy = test_accuracy
    else:
      print "ending training due to reaching max %d epochs"%(self.max_epochs,)

    self._save()

  def _classify(self, **kwargs):
    if not self.loaded:
      self._load()
      self.loaded = True
    data = kwargs.get("data")
    return self.run_prediction(data)

  def run_prediction(self, data):
    '''
    Return both the argmax of the output activations, as well as the softmax value of that activation.
    Networks that evaluate multiple logits override this behavior.
    '''
    prediction = self.sess.run(self.get_predict_tensor(),
                               feed_dict = self.get_classify_feed_dict(data))
    return np.argmax(prediction), np.max(self._softmax(prediction))

  @staticmethod
  def _softmax(prediction):
    return np.exp(prediction) / np.sum(np.exp(prediction), axis=1)

  def interpret_accuracy(self, accuracy, **kwargs):
    '''
    Process the accuracy tensor in some way, if necessary. The base case does not
    need this behavior, but networks that evaluate multiple logits do.
    '''
    return accuracy

  def merge_accuracies(self, accuracies, **kwargs):
    '''merge accuracy data returned by batch accuracy calculations'''
    num_accurate, num_total = (sum(i) for i in zip(*accuracies))
    return float(num_accurate) / num_total

  def calculate_accuracy(self, accuracy_tensor, session, batch_iterator, **kwargs):
    '''
    Iterate over test data batches and handle any necessary post-processing.
    Return final accuracy percentage and the absolute cost value
    '''
    cost_tensor = self.get_cost_tensor()
    accuracies = []
    total_cost = 0
    for batch_x, batch_y in batch_iterator:
      correct_tensor, batch_cost = session.run([accuracy_tensor, cost_tensor], 
                                               feed_dict = self.get_test_feed_dict(batch_x, batch_y))
      accuracy_data  = self.interpret_accuracy(correct_tensor, **kwargs)
      accuracies.append((accuracy_data, batch_x.shape[0]))
      total_cost += batch_cost
    return self.merge_accuracies(accuracies, **kwargs), total_cost

  def _run_session(self, operation, **kwargs):
    if self.X is None or self.Y is None:
      raise Exception("input and output tensors X and Y must be set")
    if self.saver == None:
      self.saver = tf.train.Saver(self.get_vars_to_save())
    if operation == "train":
      self._train(**kwargs)
    elif operation == "classify":
      return self._classify(**kwargs)

  def _load(self):
    path = os.path.join(self.save_dir, self.save_file)
    print "loading model from", path
    self.saver.restore(self.sess, path)

  def _save(self):
    if not os.path.exists(self.save_dir): os.mkdir(self.save_dir)
    path = os.path.join(self.save_dir, self.save_file)
    print "saving model %s to %s"%(str(self), path)
    self.saver.save(self.sess, path)

  def train(self, datasource):
    self._run_session("train", data=datasource)

  def classify_image(self, image):
    '''
    Convenience method that normalizes and resizes an input image
    before passing its data to classify()
    '''
    image = image.convert("L").resize((FLAGS.img_width, FLAGS.img_height))
    data = np.fromiter(iter(image.getdata()), np.float32)
    data = (data - np.mean(data)) / 255
    data.resize((1, data.size))
    return self.classify(data)

  def classify(self, input):
    '''
    Classify an input vector. Return the label and classification confidence.
    '''
    classification, confidence = self._run_session("classify", data=input)
    return self.process_label(classification), confidence

  def process_label(self, classification):
    '''If necesary, convert the output label into something more meaningful'''
    return classification

  def __str__(self):
    return '[%s]'%(self.name,)
