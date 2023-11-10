import tensorflow as tf

__all__ = ['dice_loss']


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    n_classes = y_pred.shape[-1]
    assert y_true.shape[-1] == y_pred.shape[-1]
    y_true = tf.reshape(y_true, [-1, n_classes])
    y_pred = tf.reshape(y_pred, [-1, n_classes])
    inter = tf.reduce_sum(y_true * y_pred, axis=0)
    union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0)
    return (2.*inter + smooth) / (union+smooth)


def dice_loss(y_true, y_pred, smooth=1e-6):
    return 1 - dice_coefficient(y_true, y_pred, smooth=smooth)
