# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of util functions for training and evaluating.
"""

import numpy
import tensorflow as tf
from sklearn import metrics
from tensorflow import logging
import metric as metrics

def quantize(features, min_quantized_value=-2.0, max_quantized_value=2.0):
    """
    Quantizes float32 `features` into string.
    """
    # assert len(features.shape) == 1  # 1-D array
    features =tf.clip_by_value(features, min_quantized_value, max_quantized_value)
    quantize_range = max_quantized_value - min_quantized_value
    features = tf.multiply((features - min_quantized_value),(255.0 / quantize_range))
    features = tf.round(features)

    #features = tf.reshape(features,[-1])
    #print(features)
    #features =tf.clip_by_value(features, -2.0, 2.0)
    #quantize_range = max_quantized_value - min_quantized_value
    #features = tf.multiply((features - min_quantized_value),(255.0 / quantize_range))
    #features = tf.map_fn(lambda f:tf.cast(tf.round(f),tf.int64), features, dtype=tf.int64)
    return features

def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
  """Dequantize the feature from the byte format to the float format.

  Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    A float vector which has the same shape as feat_vector.
  """
  assert max_quantized_value > min_quantized_value
  quantized_range = max_quantized_value - min_quantized_value
  scalar = quantized_range / 255.0
  bias = (quantized_range / 512.0) + min_quantized_value
  return tf.add(tf.multiply(feat_vector,scalar),bias)


def MakeSummary(name, value):
  """Creates a tf.Summary proto with the given name and value."""
  summary = tf.Summary()
  val = summary.value.add()
  val.tag = str(name)
  val.simple_value = float(value)
  return summary

# def CalMetric(logit, label):
#   """ Calulate matrix given preditions and labels"""
#   print("label",label)
#   print("logits",logit)
#   metric = {}
#   eps = 1e-6
#   num_classes = 20526
#   precision, recall, f1 = metrics.streaming_metric(label, logit, num_classes)
#   metric['precision'] = precision
#   metric['recall'] = recall
#   metric['f1'] = f1
#   # tf.summary.scalar('precision', precision)
#   # tf.summary.scalar('recall', recall)
#   # tf.summary.scalar('f1', f1)
#
#   return metric
def transMetric(logit,label):
    import numpy as np
    recalls = []
    precisions = []
    num = label.shape[0]
    for k in range(0, num):
        r1 = 0
        pred_label = [np.argmax(logit[k])]
        l = np.array(np.nonzero(label[k])).flatten()
        flag = [i for i in pred_label if i in l]
        if (len(flag) > 0):
            r1 += 1
        rc = r1 * 1.0 / (np.sum(label[k]) + 0.00001)
        p = r1 * 1.0 / len(pred_label)
        recalls.append(rc)
        precisions.append(p)
    recall = np.array(recalls).mean()
    precision = np.array(precisions).mean()
    f1_score = 2 * recall * precision / (recall + precision)

    return recall,precision,f1_score



def CalMetric(top_predictions_val, top_labels_val):
  """ Calulate matrix given preditions and labels"""
  metric = {}
  top_recall, top_precision, top_f1_score = transMetric(top_predictions_val,top_labels_val)

  metric['top_recall'] = top_recall
  metric['top_precision'] = top_precision
  metric['top_f1_score'] = top_f1_score
  #print("metric",metric)

  return metric
  

def AddSummary(summary_writer,
                    global_step_val,
                    epoch_info_dict,
                    summary_scope="Eval"):
  """Add the epoch summary to the Tensorboard.

  Args:
    summary_writer: Tensorflow summary_writer.
    global_step_val: a int value of the global step.
    epoch_info_dict: a dictionary of the evaluation metrics calculated for the
      whole epoch.
    summary_scope: Train or Eval.

  Returns:
    A string of this global_step summary
  """
  epoch_id = epoch_info_dict["epoch_id"]
  accuracy = epoch_info_dict["accuracy"]
  precision = epoch_info_dict["precision"]
  recall = epoch_info_dict["recall"]
  f1score = epoch_info_dict["f1score"]
  auc = epoch_info_dict["auc"]
  loss = epoch_info_dict["loss"]

  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_accuracy", accuracy),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_precision", precision),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_recall", recall),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_f1score", f1score),
          global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_loss", loss),
          global_step_val)
  summary_writer.flush()

  info = ("epoch/eval number {0} | accuracy: {1:.4f} | precision: {2:.4f} "
          "| recall: {3:.4f} | f1score: {4:.4f} | auc: {5:.4f} | loss: {6:.4f}").format(
          epoch_id, accuracy, precision, recall, f1score, auc, loss)
  return info


def GetListOfFeatureNamesAndSizes(feature_names, feature_sizes):
  """Extract the list of feature names and the dimensionality of each feature
     from string of comma separated values.

  Args:
    feature_names: string containing comma separated list of feature names
    feature_sizes: string containing comma separated list of feature sizes

  Returns:
    List of the feature names and list of the dimensionality of each feature.
    Elements in the first/second list are strings/integers.
  """
  list_of_feature_names = [
      feature_names.strip() for feature_names in feature_names.split(',')]
  list_of_feature_sizes = [
      int(feature_sizes) for feature_sizes in feature_sizes.split(',')]
  if len(list_of_feature_names) != len(list_of_feature_sizes):
    logging.error("length of the feature names (=" +
                  str(len(list_of_feature_names)) + ") != length of feature "
                  "sizes (=" + str(len(list_of_feature_sizes)) + ")")

  return list_of_feature_names, list_of_feature_sizes

