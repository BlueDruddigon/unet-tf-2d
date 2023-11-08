import tensorflow as tf
import tensorflow_datasets as tfds
from keras import Sequential
from keras.layers import RandomFlip, RandomRotation

ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False


def resize_fn(image, img_size):
    image = tf.image.resize_with_crop_or_pad(image, img_size, img_size)
    return image


augment_fn = Sequential([
  RandomFlip(),
  RandomRotation(factor=(-0.2, 0.2)),
])


def make_augmented_ds(img_size, batch_size):
    (train_ds, valid_ds, test_ds), _ = tfds.load(
      'synapse',
      split=['train', 'valid', 'test'],
      with_info=True,
      as_supervised=True,
    )
    
    augmented_train_ds = (
      train_ds.with_options(ignore_order).shuffle(batch_size * 2)
      .map(lambda x, y: (resize_fn(x, img_size), resize_fn(y, img_size)), num_parallel_calls=tf.data.AUTOTUNE)
      .map(lambda x, y: (augment_fn(x, training=True), augment_fn(y, training=True)), num_parallel_calls=tf.data.AUTOTUNE)
      .batch(batch_size)
      .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    augmented_valid_ds = (
      valid_ds.shuffle(batch_size * 2)
      .map(lambda x, y: (resize_fn(x, img_size), resize_fn(y, img_size)), num_parallel_calls=tf.data.AUTOTUNE)
      .batch(batch_size)
      .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return augmented_train_ds, augmented_valid_ds, test_ds
