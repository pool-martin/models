from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import signal 
import os
from multiprocessing import Process

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

from tensorflow.contrib.slim.python.slim.learning import train_step

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.005, 'The weight decay on the model weights.')


#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'porn2k', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v4', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 64, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

tf.app.flags.DEFINE_string(
    'gpu_to_use', '', 'gpus to use')

FLAGS = tf.app.flags.FLAGS

def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
    return None

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars)

def input(split, dataset):
    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          num_readers=FLAGS.num_readers,
          common_queue_capacity=20 * FLAGS.batch_size,
          common_queue_min=10 * FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    return image, label

def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    if not FLAGS.gpu_to_use:
        raise ValueError('You must supply the gpu to be used --gpu_to_use')

    if FLAGS.gpu_to_use:
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_to_use
    
    tf.logging.set_verbosity(tf.logging.INFO)
    graph = tf.Graph()
    with graph.as_default():

      # Create global_step
      global_step = slim.create_global_step()

      ######################
      # Select the dataset #
      ######################
      dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

      ######################
      # Select the network #
      ######################
      network_fn = nets_factory.get_network_fn(FLAGS.model_name, num_classes=(dataset.num_classes - FLAGS.labels_offset),
                                               weight_decay=FLAGS.weight_decay, is_training=True)
      #####################################
      # Select the preprocessing function #
      #####################################
      preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
      image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=True)

      ##############################################################
      # Create a dataset provider that loads data from the dataset #
      ##############################################################
      images, labels = input('train', dataset)
      images_validation, labels_validation = input('validation', dataset)
      images_test, labels_test = input('test', dataset)

      train_image_size = FLAGS.train_image_size or network_fn.default_image_size

      image = image_preprocessing_fn(images, train_image_size, train_image_size)
      image_validation = image_preprocessing_fn(images_validation, train_image_size, train_image_size)
      image_test = image_preprocessing_fn(images_test, train_image_size, train_image_size)

#     images, labels = tf.train.batch([image, label], batch_size=FLAGS.batch_size, capacity=1000 + 3 * FLAGS.batch_size, min_after_dequeue=1000)
      images, labels = tf.train.batch([image, labels], batch_size=FLAGS.batch_size, num_threads=FLAGS.num_preprocessing_threads, capacity=5 * FLAGS.batch_size)
      images_validation, labels_validation = tf.train.batch([image_validation, labels_validation], batch_size=FLAGS.batch_size, num_threads=FLAGS.num_preprocessing_threads, capacity=5 * FLAGS.batch_size)
      images_test, labels_test = tf.train.batch([image_test, labels_test], batch_size=FLAGS.batch_size, num_threads=FLAGS.num_preprocessing_threads, capacity=5 * FLAGS.batch_size)

      labels = slim.one_hot_encoding(labels, dataset.num_classes - FLAGS.labels_offset, on_value=1.0, off_value=0.0)
      labels_validation = slim.one_hot_encoding(labels_validation, dataset.num_classes - FLAGS.labels_offset, on_value=1.0, off_value=0.0)
      labels_test = slim.one_hot_encoding(labels_test, dataset.num_classes - FLAGS.labels_offset, on_value=1.0, off_value=0.0)

      with tf.variable_scope("model") as scope:
        predictions, endpoints = network_fn(images)
        scope.reuse_variables()
        predictions_validation, _ = network_fn(images_validation)
        predictions_test, _ = network_fn(images_test)
        
      slim.losses.softmax_cross_entropy(predictions, labels)
      optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
      train_op = slim.learning.create_train_op(slim.losses.get_total_loss(), optimizer)

      accuracy_validation = slim.metrics.accuracy(tf.to_int32(tf.argmax(predictions_validation, 1)), tf.to_int32(tf.argmax(labels_validation, 1)))
      accuracy_test = slim.metrics.accuracy(tf.to_int32(tf.argmax(predictions_test, 1)), tf.to_int32(tf.argmax(labels_test, 1)))
        
    def train_step_fn(session, *args, **kwargs):
      total_loss, should_stop = train_step(session, *args, **kwargs)

      if train_step_fn.step % FLAGS.validation_check == 0:
        accuracy = session.run(train_step_fn.accuracy_validation)
        print('Step %s - Loss: %.2f Accuracy: %.2f%%' % (str(train_step_fn.step).rjust(6, '0'), total_loss, accuracy * 100))

      if train_step_fn.step == (FLAGS.max_steps - 1):
        accuracy = session.run(accuracy_test)
        print('%s - Loss: %.2f Accuracy: %.2f%%' % ('FINAL TEST', total_loss, accuracy * 100))
        
      train_step_fn.step += 1
      return [total_loss, should_stop]

    train_step_fn.step = 0
    train_step_fn.accuracy_validation = accuracy_validation

    start = datetime.datetime.utcnow()
    print('Started on (UTC): ', start, sep='')
    if not FLAGS.experiment_file is None :
      experiment_file = open(FLAGS.experiment_file, 'w')
      print('Experiment metadata file:', file=experiment_file)
      print(FLAGS.experiment_file, file=experiment_file)
      print('========================', file=experiment_file)
      print('All command-line flags:', file=experiment_file)
      for flag_key in sorted(FLAGS.__flags.keys()) :
        print(flag_key, ' : ', FLAGS.__flags[flag_key], sep='', file=experiment_file)
      print('========================', file=experiment_file)
      print('Started on (UTC): ', start, sep='', file=experiment_file)
      experiment_file.flush()

    slim.learning.train(
      train_op,
      FLAGS.logs_dir,
      train_step_fn=train_step_fn,
      init_fn=_get_init_fn(),
      graph=graph,
      number_of_steps=FLAGS.max_steps
    )

    finish = datetime.datetime.utcnow()
    if not FLAGS.experiment_file is None :
      print('Finished on (UTC): ', finish, sep='', file=experiment_file)
      print('Elapsed: ', finish-start, sep='', file=experiment_file)
      experiment_file.flush()

if __name__ == '__main__':
  tf.app.run()
