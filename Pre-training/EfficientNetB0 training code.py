import tensorflow as tf
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import cv2
import os
from sklearn.metrics import confusion_matrix, roc_curve### metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (GlobalAveragePooling2D, Activation, MaxPooling2D, Add, Conv2D, MaxPool2D, Dense,
                                     Flatten, InputLayer, BatchNormalization, Input, Embedding, Permute,
                                     Dropout, RandomFlip, RandomRotation, LayerNormalization, MultiHeadAttention,
                                     RandomContrast, Rescaling, Resizing, Reshape)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers  import L2, L1
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import callbacks
import json
from tensorflow.keras.callbacks import CSVLogger

class HistorySaver(callbacks.Callback):
    def __init__(self, filepath="training_progress.json"):
        super(HistorySaver, self).__init__()
        self.filepath = filepath
        self.history_data = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history_data.setdefault(k, []).append(float(v))
        
        self._save_history()

    def _save_history(self):
        # Викликати json тут, а не зберігати як атрибут
        with open(self.filepath, "w") as f:
            json.dump(self.history_data, f)

train_directory = "/kaggle/input/deepfake-and-real-images/Dataset/Train"
val_directory = "/kaggle/input/deepfake-and-real-images/Dataset/Validation"

CONFIGURATION = {
    "BATCH_SIZE": 32,
    "IM_SIZE":224,
    "LEARNING_RATE": 1e-4,
    "N_EPOCHS": 100,
    "DROPOUT_RATE": 0.05,
    "REGULARIZATION_RATE": 0.001,
    "N_FILTERS": 6,
    "KERNEL_SIZE": 3,
    "N_STRIDES": 1,
    "POOL_SIZE": 2,
    "N_DENSE_1": 1024,
    "N_DENSE_2": 128,
    "NUM_CLASSES": 2,
    "PATCH_SIZE": 32,
    "PROJ_DIM": 768,
    "CLASS_NAMES": ["fake","real"],
    "PATIENCE": 3
}

path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
root_dir = os.path.join(path, "real_vs_fake", "real-vs-fake")

train_directory = os.path.join(root_dir, "train")
val_directory   = os.path.join(root_dir, "valid")
test_directory  = os.path.join(root_dir, "test")

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label

train_dataset = (
    tf.keras.utils.image_dataset_from_directory(
        train_directory,
        labels='inferred',
        label_mode='categorical',
        class_names=CONFIGURATION["CLASS_NAMES"],
        color_mode='rgb',
        batch_size=CONFIGURATION["BATCH_SIZE"],
        image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
        shuffle=True
    )
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.keras.utils.image_dataset_from_directory(
        val_directory,
        labels='inferred',
        label_mode='categorical',
        class_names=CONFIGURATION["CLASS_NAMES"],
        color_mode='rgb',
        batch_size=1,
        image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
        shuffle=True,
        seed=99,
    )
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

training_dataset = (
    train_dataset
    .prefetch(tf.data.AUTOTUNE)
)


validation_dataset = (
    val_dataset
    .prefetch(tf.data.AUTOTUNE)
)

input_shape = (224, 224, 3)
# Load the EfficientNetB0 model, excluding the top layers (include_top=False)
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
# Freeze the base model
base_model.trainable = False

# Create a Sequential model and add layers
model = models.Sequential()

# Add the EfficientNet base model
model.add(base_model)

# Add a global average pooling layer
model.add(layers.GlobalAveragePooling2D())

# Add a dropout layer for regularization
model.add(layers.Dropout(0.3))

# Add a dense layer with softmax activation for classification
model.add(layers.Dense(CONFIGURATION["NUM_CLASSES"], activation = "sigmoid"))

loss_function = BinaryCrossentropy()

earlystop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=CONFIGURATION["PATIENCE"],
    min_delta=0.01,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    "efficientnetb0_best.keras",
    monitor="val_accuracy",
    save_best_only=True
)
csv_logger = CSVLogger("efficientnetb0_history.csv", append=True)
model.compile(
  optimizer = Adam(learning_rate = CONFIGURATION["LEARNING_RATE"]),
  loss = loss_function,metrics=['accuracy']
)

history =model.fit(
  training_dataset,
  validation_data = validation_dataset,
  batch_size= CONFIGURATION["BATCH_SIZE"],
  epochs = CONFIGURATION["N_EPOCHS"],
  callbacks=[earlystop, checkpoint, csv_logger],
  verbose = 1
)

history_dir = './'
if not os.path.exists(history_dir):
    os.makedirs(history_dir)

with open(os.path.join(history_dir, 'efficientnetb0_history.json'), 'w') as f:
    json.dump(history.history, f)
    
model.save('./efficientnetb0.keras')

