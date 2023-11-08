import os

import tensorflow as tf

from datasets.augment import make_augmented_ds
from losses.dice_ce import dice_ce_loss
from models.unet import get_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TPU_NAME'] = 'my-tpu'

# TPU initialization
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # default distribution strategy in Tensorflow. Works on CPU and single GPU.
    tpu_strategy = tf.distribute.get_strategy()

# hyper params
IMG_SIZE = 512
N_CLASSES = 14
N_CHANNELS = 1
EPOCHS = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 2 * tpu_strategy.num_replicas_in_sync

# dataset
train_ds, valid_ds, test_ds = make_augmented_ds(IMG_SIZE, BATCH_SIZE)

save_freq = len(train_ds) * 5
saved_path = 'weights.{epoch:03d}-{val_loss:.4f}.h5'

# model compute
with tpu_strategy.scope():
    model = get_model(IMG_SIZE, N_CLASSES, N_CHANNELS)
    
    # LR scheduler
    lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(LEARNING_RATE, 1000)
    
    # custom loss fn
    custom_loss_fn = dice_ce_loss(0.4)
    
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler),
      loss=custom_loss_fn,
      metrics=['accuracy'],
    )
    
    if tpu:
        save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
        model_checkpointing = tf.keras.callbacks.ModelCheckpoint(
          filepath=saved_path, save_freq=save_freq, options=save_locally
        )
    else:
        model_checkpointing = tf.keras.callbacks.ModelCheckpoint(filepath=saved_path, save_freq=save_freq)
    
    # custom list of callbacks
    my_cbs = [tf.keras.callbacks.EarlyStopping(patience=10), model_checkpointing]

history = model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds, callbacks=my_cbs)
