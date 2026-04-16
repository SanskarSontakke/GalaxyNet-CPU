import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

# Import local components
from config import Config
from dataset import build_dataset
from losses import RMSELoss, rmse_metric

def calculate_per_target_rmse(y_true, y_pred, target_cols):
    """Calculates RMSE for each individual target column."""
    rmses = {}
    for i, col in enumerate(target_cols):
        mse = np.mean(np.square(y_true[:, i] - y_pred[:, i]))
        rmses[col] = float(np.sqrt(mse))
    return rmses

def main():
    print("🌌 Starting local model verification...")
    config = Config()
    
    # Override paths for local testing
    SOLUTIONS_CSV = Path("galaxy_raw/training_solutions_rev1.csv")
    IMAGE_DIR = Path("galaxy_raw/training_images/images_training_rev1")
    MODEL_PATH = Path("unified_best.keras")
    
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    # 1. Load Data Subset
    print(f"Reading {SOLUTIONS_CSV}...")
    df = pd.read_csv(SOLUTIONS_CSV)
    target_cols = [c for c in df.columns if c.startswith('Class')]
    
    # Filter for images that exist
    df['image_path'] = df['GalaxyID'].apply(lambda x: str(IMAGE_DIR / f"{x}.jpg"))
    df['exists'] = df['image_path'].apply(lambda x: os.path.exists(x))
    df = df[df['exists']].sample(100, random_state=config.seed)
    
    print(f"Loaded 100 images for verification.")
    
    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    custom_objects = {
        'RMSELoss': RMSELoss,
        'rmse_metric': rmse_metric,
    }
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    
    # 3. Prepare dataset
    test_paths = df['image_path'].values
    y_true = df[target_cols].values.astype(np.float32)
    
    # Standard Pred
    print("\nRunning standard prediction (No TTA)...")
    ds = build_dataset(
        test_paths, np.zeros_like(y_true), config.image_size_phase3, batch_size=16,
        center_crop_ratio=config.center_crop_ratio, augment=False, shuffle=False
    )
    t0 = time.time()
    y_pred_std = model.predict(ds, verbose=1)
    std_time = time.time() - t0
    
    # 4-pass TTA (Lite version for CPU)
    print("\nRunning Lite TTA (4 passes)...")
    tta_preds = [y_pred_std]
    for i in range(3):
        ds_aug = build_dataset(
            test_paths, np.zeros_like(y_true), config.image_size_phase3, batch_size=16,
            center_crop_ratio=config.center_crop_ratio, augment=True, shuffle=False
        )
        preds = model.predict(ds_aug, verbose=1)
        tta_preds.append(preds)
        print(f"  TTA Pass {i+2}/4 completed.")
    
    y_pred_tta = np.mean(tta_preds, axis=0)
    
    # 4. Calculate Metrics
    global_rmse_std = np.sqrt(np.mean(np.square(y_true - y_pred_std)))
    global_rmse_tta = np.sqrt(np.mean(np.square(y_true - y_pred_tta)))
    
    per_target_rmse = calculate_per_target_rmse(y_true, y_pred_tta, target_cols)
    
    # 5. Save Results
    results = {
        "global": {
            "rmse_standard": float(global_rmse_std),
            "rmse_tta_lite": float(global_rmse_tta),
            "inference_time_sec": float(std_time)
        },
        "per_target": per_target_rmse
    }
    
    with open("test_results_local.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n✅ Verification complete!")
    print(f"Standard RMSE: {global_rmse_std:.5f}")
    print(f"TTA-4 RMSE:     {global_rmse_tta:.5f}")
    print("Results saved to test_results_local.json")

if __name__ == "__main__":
    main()
