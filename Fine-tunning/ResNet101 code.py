import os
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import kagglehub

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
FINE_EPOCHS = 50
FINE_LEARNING_RATE = 1e-5
PATIENCE = 3
NUM_LAYERS_TO_UNFREEZE = 20 


path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
root_dir = "%s/real_vs_fake/real-vs-fake" % path
train_csv = "%s/train.csv" % path
valid_csv = "%s/valid.csv" % path
test_csv  = "%s/test.csv" % path

df_train = pd.read_csv(train_csv)
df_valid = pd.read_csv(valid_csv)

# Generating datasets with flipped images

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True 
)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=root_dir,
    x_col="path",
    y_col="label",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=True
)
valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=df_valid,
    directory=root_dir,
    x_col="path",
    y_col="label",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=False
)

model = tf.keras.models.load_model('./exp1/resnet101_final_model.h5')
with open('./exp1/resnet101_history.json', 'r') as f:
    history_pre = json.load(f)


for layer in model.layers[-3-NUM_LAYERS_TO_UNFREEZE:-3]:
    layer.trainable = True

model.compile(
    optimizer=optimizers.Adam(learning_rate=FINE_LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Adding callbacks

early_stop_fine = callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
checkpoint_fine = callbacks.ModelCheckpoint('resnet101_finetuned_model.keras', monitor='val_loss', save_best_only=True)

# Adding history

history_fine = model.fit(
    train_generator,
    epochs=FINE_EPOCHS,
    validation_data=valid_generator,
    callbacks=[early_stop_fine, checkpoint_fine]
)

model.save('./exp2/resnet101_finetuned_model.keras')

with open('./resnet101_finetune_history.json', 'w') as f:
    json.dump(history_fine.history, f)



