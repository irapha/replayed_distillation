import datetime as dt
import json
import os
import sys
import tensorflow as tf
import numpy as np

from subprocess import check_output


def get_logger(f):
    if len(f.commit) == 0:
        print('No commit hash provided, using most recent on HEAD')
        f.commit = check_output(['git', 'rev-parse', 'HEAD'])

    if any(map(
        lambda x: x == 0,
        [len(f.run_name), len(f.dataset), len(f.model),
            len(f.procedure)])):
        print('No Run Name, Dataset, Model, or Procedure provided!')
        sys.exit(-1)

    log = {}
    log['run_file'] = 'train.py'
    log['start_time'] = dt.datetime.now().timestamp()
    log['end_time'] = None

    log['run_name'] = f.run_name
    log['commit'] = f.commit.decode('utf-8')
    log['dataset'] = f.dataset
    log['model'] = f.model
    log['rng_seed'] = f.rng_seed
    log['train_procedure'] = f.procedure

    log['epochs'] = f.epochs
    log['train_batch_size'] = f.train_batch_size
    log['test_batch_size'] = f.test_batch_size
    log['eval_interval'] = f.eval_interval
    log['checkpoint_interval'] = f.checkpoint_interval

    return log

def save_log(log, summary_folder, run_name, log_file):
    log['end_time'] = dt.datetime.now().timestamp()
    dirname = os.path.join(summary_folder, run_name)
    ensure_dir_exists(dirname)
    with open(os.path.join(dirname, log_file), 'w') as f:
        f.write(json.dumps(log))

#  def create_optional_params():
    #  keep_prob_input = tf.placeholder(tf.float32, name='keep_prob_input')
    #  keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    #  temp = tf.placeholder(tf.float32, name='temp')
    #  labels_temp = tf.placeholder(tf.float32, name='labels_temp')
    #  return keep_prob_input, keep_prob, temp, labels_temp

def create_train_ops(h, labels, scope='train_ops'):
    with tf.variable_scope('xent_' + scope):
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=h, name='sftmax_xent'))

    with tf.variable_scope('opt_' + scope):
        train_step = tf.train.AdamOptimizer().minimize(loss)

    return loss, train_step

def create_eval_ops(y, y_, scope='train_ops'):
    with tf.variable_scope('eval_' + scope):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def create_summary_ops(loss, accuracy):
    loss_summary_op = tf.summary.scalar('loss', loss)
    accuracy_summary_op = tf.summary.scalar('accuracy', accuracy)
    return tf.summary.merge([loss_summary_op, accuracy_summary_op])

def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def get_sess_config(use_gpu=True):
    if use_gpu:
        return None
    else:
        return tf.ConfigProto(device_count={'GPU': 0})

def get_uninitted_vars(sess):
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    return uninitialized_vars

def init_uninitted_vars(sess):
    sess.run(tf.variables_initializer(get_uninitted_vars(sess)))

def merge_summary_list(summary_list, do_print=False):
    summary_dict = {}

    for summary in summary_list:
        summary_proto = tf.Summary()
        summary_proto.ParseFromString(summary)

        for val in summary_proto.value:
            if val.tag not in summary_dict:
                summary_dict[val.tag] = []
            summary_dict[val.tag].append(val.simple_value)

    # get mean of each tag
    for k, v in summary_dict.items():
        summary_dict[k] = np.mean(v)

    if do_print:
        print(summary_dict)

    # create final Summary with mean of values
    final_summary = tf.Summary()
    final_summary.ParseFromString(summary_list[0])

    for i, val in enumerate(final_summary.value):
        final_summary.value[i].simple_value = summary_dict[val.tag]

    return final_summary
