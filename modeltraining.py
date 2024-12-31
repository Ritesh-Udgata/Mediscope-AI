#This is the final and optimized version of the code. The original code was modified for efficiency, clarity, debugging and comments for better understanding

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, Input,
                                   Concatenate, Add, LeakyReLU, PReLU)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                      ModelCheckpoint, TensorBoard)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.initializers import HeNormal, HeUniform
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

base_dir = "disease_prediction_model"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(f"{base_dir}/logs", exist_ok=True)
os.makedirs(f"{base_dir}/models", exist_ok=True)
os.makedirs(f"{base_dir}/plots", exist_ok=True)

class DiseasePredictor:
    def __init__(self, data_path=None, df=None, min_samples_per_class=2):
        self.data_path = data_path
        self.df = df
        self.min_samples_per_class = min_samples_per_class
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None

    def validate_dataframe(self, df):
        """Validate and clean the input dataframe"""
        initial_rows = len(df)
        null_counts = df.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]

        if not columns_with_nulls.empty:
            print("\nFound null values in the following columns:")
            print(columns_with_nulls)

            df = df.dropna()
            print(f"Removed {initial_rows - len(df)} rows with null values")

        inf_counts = np.isinf(df.select_dtypes(include=np.number)).sum()
        columns_with_infs = inf_counts[inf_counts > 0]

        if not columns_with_infs.empty:
            print("\nFound infinite values in the following columns:")
            print(columns_with_infs)

            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            print(f"Removed {initial_rows - len(df)} rows with infinite values")

        if 'diseases' not in df.columns:
            raise ValueError("DataFrame must contain a 'diseases' column")

        non_numeric_cols = df.drop('diseases', axis=1).select_dtypes(exclude=np.number).columns
        if not non_numeric_cols.empty:
            raise ValueError(f"Found non-numeric feature columns: {non_numeric_cols.tolist()}")

        return df

    def filter_rare_classes(self, df):
        """Remove or combine classes with too few samples"""
        class_counts = df['diseases'].value_counts()
        rare_classes = class_counts[class_counts < self.min_samples_per_class].index

        if len(rare_classes) > 0:
            print(f"\nWarning: Found {len(rare_classes)} classes with fewer than {self.min_samples_per_class} samples")
            print("Rare classes:", rare_classes.tolist())

            df_filtered = df[~df['diseases'].isin(rare_classes)]

            print(f"Removed {len(df) - len(df_filtered)} samples with rare classes")
            return df_filtered

        return df

    def load_data(self):
      """Load and preprocess the data"""
      if self.data_path:
          self.df = pd.read_csv(self.data_path)

      if self.df is None:
          raise ValueError("No data provided. Please provide either data_path or df")

      print("\nInitial data shape:", self.df.shape)

      self.df = self.validate_dataframe(self.df)
      print("Shape after validation:", self.df.shape)
      self.df = self.filter_rare_classes(self.df)
      print("Shape after filtering rare classes:", self.df.shape)
      self.df = self.df.dropna(subset=['diseases'])
      print("Shape after dropping rows with missing 'diseases':", self.df.shape)

      X = self.df.drop('diseases', axis=1).values
      y = self.df['diseases'].values

      if len(X) != len(y):
          raise ValueError(f"Inconsistent number of samples: X has {len(X)} samples but y has {len(y)} samples")

      X_scaled = self.scaler.fit_transform(X)

    # Encode labels
      y_encoded = self.label_encoder.fit_transform(y)
      y_one_hot = to_categorical(y_encoded)

    # Save preprocessors
      with open(f'{base_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(self.scaler, f)
      with open(f'{base_dir}/label_encoder.pkl', 'wb') as f:
        pickle.dump(self.label_encoder, f)

      class_weights = compute_class_weight(
          class_weight='balanced',
          classes=np.unique(y_encoded),
          y=y_encoded
      )
      self.class_weight_dict = dict(enumerate(class_weights))

      print("\nFinal data summary:")
      print(f"Number of features: {X_scaled.shape[1]}")
      print(f"Number of classes: {len(np.unique(y))}")
      print(f"Number of samples: {len(X_scaled)}")
      print("\nClass distribution:")
      for class_name, count in self.df['diseases'].value_counts().items():
          print(f"{class_name}: {count} samples")

      return X_scaled, y_one_hot, y_encoded
    def create_residual_block(self, input_layer, units, dropout_rate=0.3):
        """Create a residual block with skip connection"""
        x = Dense(
            units,
            kernel_initializer=HeNormal(),
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            kernel_constraint=MaxNorm(3)
        )(input_layer)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dropout(dropout_rate)(x)

        # Skip connection if input shape matches
        if input_layer.shape[-1] == units:
            x = Add()([x, input_layer])

        return x

    def build_model(self, input_shape, num_classes):
        """Build the neural network model"""
        inputs = Input(shape=input_shape)

        # First layer
        x = Dense(
            512,
            kernel_initializer=HeNormal(),
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            kernel_constraint=MaxNorm(3)
        )(inputs)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dropout(0.3)(x)

        # Residual blocks
        x = self.create_residual_block(x, 512)
        x = self.create_residual_block(x, 512)
        x = self.create_residual_block(x, 256)
        
        outputs = Dense(num_classes, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=outputs)

        return self.model

    def get_callbacks(self):
        """Setup training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                f'{base_dir}/models/best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=f"{base_dir}/logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                histogram_freq=1
            )
        ]
        return callbacks
    

    def plot_training_history(self, histories, save_path):
        """Plot training metrics"""
        metrics = ['loss', 'accuracy', 'auc']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for metric, ax in zip(metrics, axes):
            for fold, history in enumerate(histories):
                ax.plot(history[metric], label=f'Fold {fold+1} Train')
                ax.plot(history[f'val_{metric}'], label=f'Fold {fold+1} Val')
            ax.set_title(f'Model {metric}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def prune_model(self, model, X_train, y_train, X_test, y_test):
        """Prune the model to reduce size"""
        end_step = np.ceil(len(X_train) / 64).astype(np.int32) * 10
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.30,
                final_sparsity=0.70,
                begin_step=0,
                end_step=end_step
            )
        }

        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        pruned_model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=f"{base_dir}/logs/pruning")
        ]

        pruned_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=64,
            callbacks=callbacks
        )

        final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        final_model.save(f'{base_dir}/models/final_pruned_model.keras')

        return final_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        """Train the model"""
        model = self.build_model((X_train.shape[1],), y_train.shape[1])
        optimizer = Adam(learning_rate=1e-4, amsgrad=True)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            class_weight=self.class_weight_dict,
            verbose=1
        )

        model.save(f'{base_dir}/models/best_model.keras')
        return model, history  

def main():
    predictor = DiseasePredictor(df=df_filtered)

    X, y, y_encoded = predictor.load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    best_model, history = predictor.train_model(X_train, y_train, X_val, y_val)
    test_scores = best_model.evaluate(X_val, y_val)
    print("\nBest Model Evaluation:")
    print(f"Test Loss: {test_scores[0]:.4f}")
    print(f"Test Accuracy: {test_scores[1]:.4f}")
    print(f"Test AUC: {test_scores[2]:.4f}")

    final_pruned_model = predictor.prune_model(
        best_model,
        X_train, y_train,
        X_val, y_val
    )
    pruned_scores = final_pruned_model.evaluate(X_val, y_val)
    print("\nPruned Model Evaluation:")
    print(f"Test Loss: {pruned_scores[0]:.4f}")
    print(f"Test Accuracy: {pruned_scores[1]:.4f}")

if __name__ == "__main__":
    main()
