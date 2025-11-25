# train.py
# Main training script for CECS 456 - Final Project -Animals10 

import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# 0. Reproducibility
# This helps keep results more consistent
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# 1. Paths and parameters
DATA_DIR = os.path.join("data", "animals10")

# smaller images for much faster training
IMG_SIZE = (64, 64)         
BATCH_SIZE = 32

# fewer epochs so training finishes quickly
EPOCHS = 10                

VAL_SPLIT = 0.15


# 2. Data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    validation_split=VAL_SPLIT,
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

class_names = list(train_gen.class_indices.keys())
num_classes = len(class_names)
print("Classes:", class_names)


# 3. Build model (custom CNN)
def create_model() -> tf.keras.Model:
    model = models.Sequential(
        [
            # first conv block
            layers.Conv2D(16, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
            layers.MaxPooling2D(),

            # second conv block
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(),

            # classifier part
            layers.Flatten()
            ,
            layers.Dense(64, activation="relu"),   # fewer units for speed 
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = create_model()
model.summary()


# 4. Train model
# limit steps per epoch means each epoch does not need to see every single batch
steps_per_epoch = min(len(train_gen), 50)
validation_steps = min(len(val_gen), 20)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)

os.makedirs("results", exist_ok=True)

# save training history
history_path = os.path.join("results", "history.json")
with open(history_path, "w", encoding="utf-8") as f:
    json.dump(history.history, f)
print("Training history saved to", history_path)


# 5. Save model
model_path = os.path.join("results", "animals10_cnn.h5")
model.save(model_path)
print("Model saved to", model_path)


# 6. Evaluation on validation subset
val_gen.reset()
preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

print("Classification Report for validation subset")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix for validation subset")
plt.colorbar()
plt.xticks(np.arange(num_classes), class_names, rotation=45)
plt.yticks(np.arange(num_classes), class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
cm_path = os.path.join("results", "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print("Confusion matrix saved to", cm_path)


# 7. Plot accuracy and loss curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="validation accuracy")
plt.title("Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
curves_path = os.path.join("results", "training_curves.png")
plt.savefig(curves_path)
plt.close()
print("Training curves saved to", curves_path)
