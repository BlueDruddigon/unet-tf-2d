import tensorflow as tf

from .dice import dice_loss

__all__ = ['dice_ce_loss']


def dice_ce_loss(alpha: float = 0.5):
    cross_entropy = tf.keras.losses.CategoricalCrossentropy()
    
    def loss_fn(y_true, y_pred):
        if y_true.shape[-1] != y_pred.shape[-1]:
            y_true = tf.cast(y_true, dtype=tf.uint8)
            y_true = tf.squeeze(tf.one_hot(y_true, y_pred.shape[-1]), axis=3)
        
        dice = dice_loss(y_true, y_pred)
        ce = cross_entropy(y_true, y_pred)
        return alpha*dice + (1-alpha) * ce
    
    return loss_fn
