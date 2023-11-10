import tensorflow as tf
import tensorflow_datasets as tfds

from .augment import augment_fn, resize_fn

__all__ = ['make_augmented_ds']


def make_augmented_ds(img_size: int, batch_size: int):
    train_ds, valid_ds, test_ds = tfds.load(
      'mnist',
      split=['train', 'valid', 'test'],
      as_supervised=True,
    )
    
    augmented_train_ds = (
      train_ds.shuffle(batch_size * 2).map(lambda x, y: resize_fn(x, y, img_size),
                                           num_parallel_calls=tf.data.AUTOTUNE).map(
                                             lambda x, y: augment_fn(x, y, training=True),
                                             num_parallel_calls=tf.data.AUTOTUNE
                                           ).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    batch_valid_ds = (
      valid_ds.map(lambda x, y: resize_fn(x, y, img_size),
                   num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    batch_test_ds = (
      test_ds.map(lambda x, y: resize_fn(x, y, img_size),
                  num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    return augmented_train_ds, batch_valid_ds, batch_test_ds
