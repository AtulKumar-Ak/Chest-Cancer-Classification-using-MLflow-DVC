import os 
import tensorflow as tf 
import urllib.request as request
from zipfile import ZipFile
from pathlib import Path
from src.cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        # CRITICAL FIX: Unfreeze last few layers for better learning
        if freeze_all:
            # Freeze all base model layers
            model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            # Freeze all layers first
            model.trainable = True
            # Then freeze only the first layers, keeping last layers trainable
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False
        else:
            # Train all layers
            model.trainable = True

        # Add custom classification head
        flatten_in = tf.keras.layers.Flatten()(model.output)
        
        # Add dropout for regularization
        dropout1 = tf.keras.layers.Dropout(0.5)(flatten_in)
        
        # Add a dense layer before final classification
        dense1 = tf.keras.layers.Dense(
            units=256,
            activation="relu"
        )(dropout1)
        
        dropout2 = tf.keras.layers.Dropout(0.3)(dense1)
        
        # Final classification layer
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(dropout2)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Use Adam optimizer instead of SGD
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        
        # Print trainable vs non-trainable parameters
        trainable_params = sum([tf.size(w).numpy() for w in full_model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in full_model.non_trainable_weights])
        print(f"\n{'='*60}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print(f"{'='*60}\n")
        
        return full_model
    
    def update_base_model(self):
        # CHANGED: Unfreeze last 4 layers instead of freezing all
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,  # Changed from True
            freeze_till=4,     # Unfreeze last 4 layers
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
