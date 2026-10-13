"""CNN Model Architecture and Callbacks module.
Implements:
- Keras Sequential CNN with Conv2D and MaxPooling layers (Kriteria 4)
- Regularization with BatchNormalization, Dropout, and L2
- Adaptive Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TargetAccuracyCallback (Saran 1)
"""
import tensorflow as tf
from typing import Tuple, List

class TargetAccuracyCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to halt training once the target accuracy threshold (>= 95%) is achieved
    on both training and validation sets, fulfilling Dicoding 5-star criteria.
    """
    def __init__(self, target_train_acc: float = 0.95, target_val_acc: float = 0.95):
        super().__init__()
        self.target_train_acc = target_train_acc
        self.target_val_acc = target_val_acc

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_acc = logs.get('accuracy', 0.0)
        val_acc = logs.get('val_accuracy', 0.0)

        if train_acc >= self.target_train_acc and val_acc >= self.target_val_acc:
            print(f"\n[TARGET REACHED] Epoch {epoch + 1}: Train Acc = {train_acc:.4f} >= {self.target_train_acc}, Val Acc = {val_acc:.4f} >= {self.target_val_acc}!")
            print("Target 5-star accuracy achieved! Stopping training early.")
            self.model.stop_training = True

def build_sequential_cnn(input_shape: Tuple[int, int, int] = (120, 120, 3),
                         num_classes: int = 6) -> tf.keras.Sequential:
    """
    Build high-capacity deep Sequential CNN model strictly compliant with Dicoding Criteria 4.
    """
    model = tf.keras.Sequential([
        # Input layer with normalization
        tf.keras.layers.Rescaling(1.0 / 255.0, input_shape=input_shape, name="rescaling"),

        # Conv Block 1
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1"),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv1_2"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
        tf.keras.layers.BatchNormalization(name="bn1"),

        # Conv Block 2
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1"),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv2_2"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
        tf.keras.layers.BatchNormalization(name="bn2"),

        # Conv Block 3
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1"),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", name="conv3_2"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),
        tf.keras.layers.BatchNormalization(name="bn3"),

        # Conv Block 4
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv4_1"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool4"),
        tf.keras.layers.BatchNormalization(name="bn4"),

        # Dense Classifier Head
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(256, activation="relu", name="dense1"),
        tf.keras.layers.BatchNormalization(name="bn_dense1"),
        tf.keras.layers.Dropout(0.3, name="drop_dense1"),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output")
    ], name="intel_scene_sequential_cnn")

    return model

def get_callbacks(checkpoint_path: str = "best_model.keras",
                  target_acc: float = 0.95) -> List[tf.keras.callbacks.Callback]:
    """
    Assemble robust training callbacks:
    1. EarlyStopping
    2. ReduceLROnPlateau
    3. ModelCheckpoint
    4. TargetAccuracyCallback (custom callback)
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        TargetAccuracyCallback(target_train_acc=target_acc, target_val_acc=target_acc)
    ]
    return callbacks
