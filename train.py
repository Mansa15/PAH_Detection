"""
Pulmonary Artery Hypertension (PAH) Detection - Training Script
Uses CNN (EfficientNetB0) for binary classification of chest X-rays / CT scans
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import json
import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 50
NUM_CLASSES = 2          # 0 = Normal, 1 = PAH
RANDOM_SEED = 42
DATA_DIR    = "data/images"      # Folder with subfolders: normal/ and pah/
MODEL_DIR   = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ─── STEP 1: DATA PREPARATION ─────────────────────────────────────────────────

def build_generators(data_dir: str):
    """Build train / validation / test ImageDataGenerators."""

    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )

    test_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        seed=RANDOM_SEED,
    )

    val_gen = train_aug.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        seed=RANDOM_SEED,
        shuffle=False,
    )

    return train_gen, val_gen


# ─── STEP 2: CLASS WEIGHTS ────────────────────────────────────────────────────

def get_class_weights(train_gen):
    """Compute class weights to handle imbalanced datasets."""
    labels = train_gen.classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\n[Class Weights] {class_weight_dict}\n")
    return class_weight_dict


# ─── STEP 3: MODEL DEFINITION ─────────────────────────────────────────────────

def build_model():
    """
    Transfer learning with EfficientNetB0 as backbone.
    Top layers are fine-tuned for PAH binary classification.
    """
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    # Freeze all base layers initially
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),   # Binary output
    ], name="PAH_Detector")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()
    return model


# ─── STEP 4: TRAINING ─────────────────────────────────────────────────────────

def train(model, train_gen, val_gen, class_weights):
    """Train the model with callbacks."""

    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.h5"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]

    print("\n[Phase 1] Training top layers only...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # Fine-tune: unfreeze last 30 layers of base
    print("\n[Phase 2] Fine-tuning last 30 base layers...")
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    return history1, history2


# ─── EVALUATION & PLOTS ───────────────────────────────────────────────────────

def plot_history(h1, h2):
    """Plot training curves."""
    acc  = h1.history["accuracy"]  + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    auc  = h1.history["auc"]  + h2.history["auc"]
    vauc = h1.history["val_auc"] + h2.history["val_auc"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(acc, label="Train Acc")
    axes[0].plot(vacc, label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(auc, label="Train AUC")
    axes[1].plot(vauc, label="Val AUC")
    axes[1].set_title("AUC")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
    print(f"\nTraining curves saved to {MODEL_DIR}/training_curves.png")


def evaluate(model, val_gen):
    """Print full classification report."""
    val_gen.reset()
    preds = (model.predict(val_gen) > 0.5).astype(int).flatten()
    labels = val_gen.classes
    print("\n── Classification Report ──")
    print(classification_report(labels, preds, target_names=["Normal", "PAH"]))

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:\n", cm)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PAH Detection - Training ===\n")

    train_gen, val_gen = build_generators(DATA_DIR)
    class_weights = get_class_weights(train_gen)

    # Save class index map
    with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
        json.dump(train_gen.class_indices, f)

    model = build_model()
    h1, h2 = train(model, train_gen, val_gen, class_weights)

    plot_history(h1, h2)
    evaluate(model, val_gen)

    # Save final model
    model.save(os.path.join(MODEL_DIR, "pah_model_final.h5"))
    print("\nModel saved to models/pah_model_final.h5")
