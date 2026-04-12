from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from config import Config
from dataset import build_dataset, CLASS_NAMES
from losses import BinaryFocalLoss, OHEMBinaryLoss


def _build_feature_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """Create a headless version of a trained model that outputs feature vectors.
    
    Removes the final Dense(1, sigmoid) output layer and returns the model
    that outputs the penultimate layer's features (before the classification head).
    """
    # Find the last Dense layer and use the layer before it as output
    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        if isinstance(layer, tf.keras.layers.Dense) and layer.output_shape[-1] == 1:
            # Found the classification head — use the previous layer's output
            feature_output = model.layers[i - 1].output
            return tf.keras.Model(inputs=model.input, outputs=feature_output)
    
    # Fallback: use the second-to-last layer
    return tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)


def extract_features(
    model_paths: list[str],
    df: pd.DataFrame,
    config: Config,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract deep features from multiple models and concatenate them.
    
    For each model, removes the classification head and passes all images
    through the backbone + intermediate layers to get feature vectors.
    The features from all models are concatenated horizontally.
    
    Args:
        model_paths: Paths to trained Keras models.
        df: DataFrame with 'image_path' column.
        config: Training config.
        batch_size: Batch size for feature extraction.
    
    Returns:
        features: np.ndarray of shape (n_samples, total_feature_dim)
    """
    paths = df['image_path'].astype(str).to_numpy()
    image_size = config.image_size_phase3
    dummy_labels = np.zeros(len(paths), dtype=np.float32)
    
    all_features = []
    custom_objects = {
        'BinaryFocalLoss': BinaryFocalLoss,
        'OHEMBinaryLoss': OHEMBinaryLoss,
    }
    
    for model_path in model_paths:
        print(f'  Extracting features from: {model_path}')
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        extractor = _build_feature_extractor(model)
        
        ds = build_dataset(
            paths, dummy_labels, image_size, batch_size,
            center_crop_ratio=config.center_crop_ratio,
        )
        
        features = extractor.predict(ds, verbose=0)
        all_features.append(features)
        
        # Cleanup
        del model, extractor
        tf.keras.backend.clear_session()
    
    # Concatenate features from all models: (n_samples, dim1 + dim2 + ...)
    concatenated = np.concatenate(all_features, axis=1)
    print(f'  Total feature dimension: {concatenated.shape[1]}')
    return concatenated


def train_xgboost_stacker(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    config: Config,
    output_path: Path,
) -> object:
    """Train an XGBoost classifier on extracted deep features.
    
    Uses the concatenated feature vectors from multiple CNN/Transformer
    backbones to learn non-linear decision boundaries that simple
    averaging or thresholding cannot capture.
    
    Args:
        train_features: Training features from extract_features().
        train_labels: Integer class labels (0=spiral, 1=elliptical, 2=irregular).
        val_features: Validation features.
        val_labels: Validation integer class labels.
        config: Training config with XGBoost hyperparameters.
        output_path: Path to save the trained model.
    
    Returns:
        Trained XGBoost classifier.
    """
    try:
        import xgboost as xgb
    except ImportError:
        print('WARNING: xgboost not available. Installing...')
        import subprocess
        subprocess.check_call(['pip', 'install', 'xgboost', '-q'])
        import xgboost as xgb
    
    # Compute sample weights for class imbalance
    class_counts = np.bincount(train_labels, minlength=3)
    total = len(train_labels)
    class_weights = total / (3.0 * class_counts + 1e-8)
    sample_weights = np.array([class_weights[label] for label in train_labels])
    
    print(f'\n[XGBoost Stacker] Training on {train_features.shape[0]} samples, '
          f'{train_features.shape[1]} features...')
    print(f'  Class distribution: {dict(zip(CLASS_NAMES, class_counts))}')
    
    clf = xgb.XGBClassifier(
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        tree_method='hist',      # Fast histogram-based method
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,           # L1 regularization
        reg_lambda=1.0,          # L2 regularization
        random_state=config.seed,
        verbosity=0,
    )
    
    clf.fit(
        train_features, train_labels,
        sample_weight=sample_weights,
        eval_set=[(val_features, val_labels)],
        verbose=False,
    )
    
    # Evaluate on validation set
    val_preds = clf.predict(val_features)
    val_acc = float(np.mean(val_preds == val_labels))
    print(f'  XGBoost val accuracy: {val_acc:.4f}')
    
    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f'  Model saved to: {output_path}')
    
    return clf


def predict_with_stacker(
    clf,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run XGBoost stacker prediction.
    
    Args:
        clf: Trained XGBoost classifier.
        features: Feature vectors from extract_features().
    
    Returns:
        predictions: Integer class predictions.
        probabilities: Class probability matrix (n_samples, 3).
    """
    probabilities = clf.predict_proba(features)
    predictions = clf.predict(features)
    return predictions.astype(np.int32), probabilities


def load_stacker(model_path: Path):
    """Load a saved XGBoost stacker from disk."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)
