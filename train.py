import os

import tensorflow as tf

from datasets import make_augmented_ds
from losses import dice_ce_loss
from models.unet import get_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# hyper params
IMG_SIZE = 512
N_CLASSES = 14
N_CHANNELS = 1
EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 16

# dataset
train_ds, valid_ds, test_ds = make_augmented_ds(IMG_SIZE, BATCH_SIZE)

save_freq = len(train_ds) * 5
saved_path = 'weights.{epoch:03d}-{val_loss:.4f}.h5'

# model definition
model = get_model(IMG_SIZE, N_CLASSES, N_CHANNELS)

# LR scheduler
lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(LEARNING_RATE, 1000)

# custom loss fn
custom_loss_fn = dice_ce_loss(0.3)

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler),
  loss=custom_loss_fn,
  metrics=['accuracy'],
)

# custom list of callbacks
my_cbs = [
  tf.keras.callbacks.EarlyStopping(patience=10),
  tf.keras.callbacks.ModelCheckpoint(filepath=saved_path, save_freq=save_freq),
]

history = model.fit(
  train_ds,
  epochs=EPOCHS,
  validation_data=valid_ds,
  callbacks=my_cbs,
)
