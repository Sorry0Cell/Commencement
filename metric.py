import tensorflow as tf
import numpy as np

from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops


def metric_variable(shape, dtype, validate_shape=True, name=None):
    return variable_scope.variable(
        lambda: array_ops.zeros(shape, dtype),
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES],
        validate_shape=validate_shape,
        name=name,
    )


def streaming_metric(y_true, y_pred, num_classes):


    # Counts for the precision, recall, f1 score
    tp_var = metric_variable(
        shape=[], dtype=tf.int32, validate_shape=False, name="tp_var"
    )
    pred_pos_var = metric_variable(
        shape=[], dtype=tf.int32, validate_shape=False, name="pred_pos_var"
    )
    true_pos_var = metric_variable(
        shape=[], dtype=tf.int32, validate_shape=False, name="true_pos_var"
    )

    # Update ops.
    print("aaaaaaaaaaaaaaaaaaa")
    print("precision1111111111",(y_pred * y_true))
    up_tp = tf.assign_add(tp_var, tf.reduce_sum(tf.cast(
        tf.clip_by_value(y_pred*y_true, 0, 1) > 0.5, dtype=tf.int32)))
    print("up_tpppppppppppp",up_tp)
    up_pred_pos = tf.assign_add(pred_pos_var, tf.reduce_sum(tf.cast(
        tf.clip_by_value(y_pred, 0, 1) > 0.5, dtype=tf.int32)))

    up_true_pos = tf.assign_add(true_pos_var, tf.reduce_sum(tf.cast(
        tf.clip_by_value(y_true, 0, 1) > 0.5, dtype=tf.int32)))

    # Grouping values
    counts = (tp_var, pred_pos_var, true_pos_var)
    updates = tf.group(up_tp, up_pred_pos, up_true_pos)

    with tf.control_dependencies([updates]):
         precision, recall, f1 = streaming_f1(counts)
    print("precision2222", precision)
    return precision, recall, f1


def streaming_f1(counts):

    epsilon = 1e-7
    tp, pred_pos, true_pos = counts
    tp = tf.cast(tp, tf.float32)
    pred_pos = tf.cast(pred_pos, tf.float32)
    print("pred_posss",pred_pos)
    true_pos = tf.cast(true_pos, tf.float32)
    prec = tp / (pred_pos + epsilon)
    rec = tp / (true_pos + epsilon)
    f1 = 2 * prec * rec / (prec + rec)

    return prec, rec, f1


def tf_f1_score(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    #TP = tf.count_nonzero(y_pred * y_true, axis=None)
    TP = tf.reduce_sum(tf.cast(
        tf.clip_by_value(y_pred*y_true, 0, 1) > 0.5, dtype=tf.int64))
    FP = tf.count_nonzero(y_pred * (y_true - 1), axis=None)
    FN = tf.count_nonzero((y_pred - 1) * y_true, axis=None)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    f1score = tf.reduce_mean(f1)

    return precision, recall, f1score


def alter_data(_data):
    data = _data.copy()
    new_data = []
    for d in data:
        for i, l in enumerate(d):
            if np.random.rand() < 0.2:
                d[i] = d[i] / 10.
        new_data.append(d)
    return np.array(new_data)


def get_data():
    num_classes = 10
    classes = list(range(num_classes))
    examples = 10000
    max_labels = 5
    class_probabilities = np.array(
        list(6 * np.exp(-i * 5 / num_classes) + 1 for i in range(num_classes))
    )
    class_probabilities /= class_probabilities.sum()
    labels = [
        np.random.choice(
            classes,
            size=np.random.randint(1, max_labels),
            p=class_probabilities,
            replace=False,
        )
        for _ in range(examples)
    ]
    y_true = np.zeros((examples, num_classes)).astype(np.float32)
    for i, l in enumerate(labels):
        y_true[i][l] = 1
    y_pred = alter_data(y_true)
    return y_true, y_pred


if __name__ == "__main__":
    np.random.seed(0)
    y_true, y_pred = get_data()
    print(y_true[:2], y_pred[:2])
    num_classes = y_true.shape[-1]

    bs = 100
    t = tf.placeholder(tf.float32, [None, None], "y_true")
    p = tf.placeholder(tf.float32, [None, None], "y_pred")
    tf_f1 = tf_f1_score(t, p)
    streamed_f1 = streaming_metric(t, p, num_classes)

    with tf.Session() as sess:
        tf.local_variables_initializer().run()

        pre, rec, f1s = sess.run(tf_f1, feed_dict={t: y_true, p: y_pred})
        print("{:40}".format("\nTotal, overall f1 scores: "), pre, rec, f1s)

        for i in range(len(y_true) // bs):
            y_t = y_true[i * bs : (i + 1) * bs].astype(np.int32)
            y_p = y_pred[i * bs : (i + 1) * bs].astype(np.int32)
            rlt = sess.run(streamed_f1, feed_dict={t: y_t, p: y_p})

        precision, recall, f1 = rlt
        print("{:40}".format("\nStreamed, batch-wise f1 scores:"), precision, recall, f1)

