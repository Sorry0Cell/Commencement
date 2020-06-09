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

"""Provides readers configured for different datasets."""

import tensorflow as tf
import utils_old
import pickle
from tensorflow import logging
def resize_axis(tensor, axis, new_size, fill_value=0):
  """Truncates or pads a tensor to new_size on on a given axis.

  Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
  size increases, the padding will be performed at the end, using fill_value.

  Args:
    tensor: The tensor to be resized.
    axis: An integer representing the dimension to be sliced.
    new_size: An integer or 0d tensor representing the new value for
      tensor.shape[axis].
    fill_value: Value to use to fill any new entries in the tensor. Will be
      cast to the type of tensor.

  Returns:
    The resized tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unstack(tf.shape(tensor))

  pad_shape = shape[:]
  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

  shape[axis] = tf.minimum(shape[axis], new_size)
  shape = tf.stack(shape)

  resized = tf.concat([
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ], axis)

  # Update shape.
  new_shape = tensor.get_shape().as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized.set_shape(new_shape)
  return resized

class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()


class YT8MFrameFeatureReader(BaseReader):
  """Reads TFRecords of SequenceExamples.

  The TFRecords must contain SequenceExamples with the sparse in64 'labels'
  context feature and a fixed length byte-quantized feature vector, obtained
  from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  """

  def __init__(self,
               num_classes=4224):
    """Construct a YT8MFrameFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
    """

    self.num_classes = num_classes

  def get_video_matrix(self,
                       features,
                       feature_size,
                       max_frames,
                       max_quantized_value,
                       min_quantized_value):
    """Decodes features from an input string and quantizes it.

    Args:
      features: raw feature values
      feature_size: length of each frame feature vector
      max_frames: number of frames (rows) in the output feature_matrix
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      feature_matrix: matrix of all frame-features
      num_frames: number of frames in the sequence
    """
    decoded_features = tf.reshape(
        tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
    feature_matrix = utils_old.Dequantize(decoded_features,
                                          max_quantized_value,
                                          min_quantized_value)
    feature_matrix = resize_axis(feature_matrix, 0, max_frames)
    return feature_matrix, num_frames

  def do_pca(self, input,max_quantized_value=2.0,
                     min_quantized_value=-2.0):
    reduce_dim = 1024
    load_file = open("model_pca_tag_category_100w.pickle", "rb")
    mean_block3 = pickle.load(load_file)
    component_block3 = pickle.load(load_file)
    component_block3 = component_block3[:, 0:reduce_dim]
    singular_values_ = pickle.load(load_file)
    singular_block3 = tf.constant(singular_values_, dtype=tf.float32, name='pac_singular_block3')
    mean_block3 = tf.constant(mean_block3, dtype=tf.float32, name='pac_mean_block3')
    component_block3 = tf.constant(component_block3, dtype=tf.float32, name='pac_component_block3')
    res_fea_pca = tf.matmul(input - mean_block3, component_block3) / tf.sqrt(singular_block3[0:reduce_dim] + 1e-4)

    res_fea = utils_old.quantize(res_fea_pca, max_quantized_value=max_quantized_value, min_quantized_value=min_quantized_value)
    res_fea = utils_old.Dequantize(res_fea, max_quantized_value=max_quantized_value, min_quantized_value=min_quantized_value)
    # res_fea_pca = tf.reshape(res_fea_pca, [-1, frams, reduce_dim])
    # res_fea = tf.reshape(res_fea_pca, tf.shape(res_fea_pca))
    return res_fea

  def prepare_reader(self,
                     filename_queue, mode, 
                     max_quantized_value=2,
                     min_quantized_value=-2):
    """Creates a single reader thread for YouTube8M SequenceExamples.

    Args:
      filename_queue: A tensorflow queue of filename locations.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      A tuple of video indexes, video features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    return self.prepare_serialized_examples(serialized_example, mode, 
                                            max_quantized_value, min_quantized_value)

  def prepare_serialized_examples(self, serialized_example, mode, 
                                  max_quantized_value=2, min_quantized_value=-2):
    video_feature_size = 2048
    audio_feature_size = 128
    max_frames = 20
    fid = 'feed-feedid'
    # 2019-08-21 zhangdi_sx@qiyi.com
    video = "features-sv_content_label_1115_0612"
    if mode == 'train':
        labels = 'labels-video_audio_train_label_0819'
    elif mode == 'test':
        labels = 'labels-video_audio_train_label_0819'
    
    audio = "features-vggish_embedding"
    feats = tf.parse_single_example(serialized_example,
                                    features={video: tf.VarLenFeature(tf.float32),
                                              fid: tf.FixedLenFeature([1], tf.string),
                                              audio: tf.VarLenFeature(tf.float32),
                                              labels: tf.FixedLenFeature([1], tf.string)}
                                    )

    # 2019-08-21 zhangdi_sx@qiyi.com
    top_label = tf.string_split(feats[labels], ":").values

    top_label = tf.string_to_number(top_label, out_type=tf.int32)
    top_label = tf.one_hot(top_label, self.num_classes)
    batch_top_labels = tf.reduce_sum(top_label, axis=0)

    batch_video_ids = feats[fid]
    video_decoded_features = tf.sparse_to_dense(feats[video].indices,
                                          feats[video].dense_shape,
                                          feats[video].values,
                                          default_value=0.0)
    audio_decoded_features = tf.sparse_to_dense(feats[audio].indices,
                                          feats[audio].dense_shape,
                                          feats[audio].values,
                                          default_value=0.0)

    video_decoded_features = tf.reshape(video_decoded_features,
        shape=(-1, video_feature_size))
    video_feature_matrix = resize_axis(video_decoded_features, 0, max_frames)

    audio_frames = audio_decoded_features.get_shape().as_list()[0]
    audio_decoded_features = tf.reshape(audio_decoded_features,
        shape=(-1, audio_feature_size))
    audio_feature_matrix = resize_axis(audio_decoded_features, 0, max_frames)

    num_video_frames = tf.minimum(tf.shape(video_decoded_features)[0], max_frames)
    num_audio_frames = tf.minimum(tf.shape(audio_decoded_features)[0], max_frames)
    batch_video_matrix = tf.expand_dims(video_feature_matrix, 0)
    batch_audio_matrix = tf.expand_dims(audio_feature_matrix, 0)
    batch_top_labels = tf.expand_dims(batch_top_labels, 0)
    batch_video_frames = tf.expand_dims(num_video_frames, 0)
    batch_audio_frames = tf.expand_dims(num_video_frames, 0)

    return batch_video_ids, batch_video_matrix, batch_audio_matrix, batch_top_labels, batch_video_frames, batch_audio_frames

