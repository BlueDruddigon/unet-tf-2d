import random
from typing import List, Optional, Tuple, Union

import tensorflow as tf


class CustomRandomFlip(tf.keras.layers.Layer):
    modes = ['horizontal_and_vertical', 'horizontal', 'vertical']
    
    def __init__(self, mode: str = 'horizontal_and_vertical', seed: Optional[int] = None) -> None:
        super(CustomRandomFlip, self).__init__()
        
        assert mode in self.modes, 'This mode is not supported'
        
        self.image_augment = tf.keras.layers.RandomFlip(mode=mode, seed=seed)
        self.label_augment = tf.keras.layers.RandomFlip(mode=mode, seed=seed)
    
    def call(self, image: tf.Tensor, label: tf.Tensor, training: bool = True):
        if not training:
            return image, label
        
        image = self.image_augment(image)
        label = self.label_augment(label)
        
        return image, label


class CustomRandomRotation(tf.keras.layers.Layer):
    fill_modes = ['constant', 'reflect', 'wrap', 'nearest']
    interpolations = ['nearest', 'bilinear']
    
    def __init__(
      self,
      factor: Union[int, List[int], Tuple[int, int]],
      fill_mode: str = 'reflect',
      interpolation: str = 'bilinear',
      seed: Optional[int] = None
    ) -> None:
        super(CustomRandomRotation, self).__init__()
        
        assert fill_mode in self.fill_modes
        assert interpolation in self.interpolations
        
        self.image_augment = tf.keras.layers.RandomRotation(
          factor=factor, fill_mode=fill_mode, interpolation=interpolation, seed=seed
        )
        self.lable_augment = tf.keras.layers.RandomRotation(
          factor=factor, fill_mode=fill_mode, interpolation=interpolation, seed=seed
        )
    
    def call(self, image: tf.Tensor, label: tf.Tensor, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        if not training:
            return image, label
        
        image = self.image_augment(image)
        label = self.label_augment(label)
        
        return image, label


def resize_fn(image, label, img_size):
    image = tf.image.resize_with_crop_or_pad(image, img_size, img_size)
    label = tf.image.resize_with_crop_or_pad(label, img_size, img_size)
    return image, label


def augment_fn(image, label, training: Optional[bool] = None):
    seed = random.randint(0, 1000)
    L = tf.keras.Sequential([
      CustomRandomFlip(seed=seed),
      CustomRandomRotation(factor=(-0.2, 0.2), seed=seed),
    ])
    
    for layer in L.layers:
        image, label = layer(image, label, training=training)
    
    return image, label
