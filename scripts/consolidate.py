import re
from pathlib import Path

# Order matters for dependencies
MODULES = [
    'config.py',
    'utils.py',
    'losses.py',
    'model.py',
    'dataset.py',
    'evaluate.py',
    'train.py'
]

OUTPUT_FILE = 'train_kaggle_bundle.py'

def consolidate():
    output_lines = [
        '# AUTO-GENERATED KAGGLE BUNDLE',
        '# This file merges modular components for Kaggle compatibility.',
        'from __future__ import annotations',
        'import os',
        'import sys',
        'import time',
        'import argparse',
        'import json',
        'import zipfile',
        'import shutil',
        'from pathlib import Path',
        '',
        'import numpy as np',
        'import pandas as pd',
        'import tensorflow as tf',
        'import matplotlib.pyplot as plt',
        'import seaborn as sns',
        'from sklearn.metrics import confusion_matrix, classification_report',
        '',
    ]

    # Pattern to match: from [module] import ... or import [module]
    module_names = [m.replace('.py', '') for m in MODULES]
    pattern = re.compile(rf'^(from\s+({"|".join(module_names)})\s+import|import\s+({"|".join(module_names)}))')

    seen_imports = set()

    for filename in MODULES:
        file_path = Path(filename)
        if not file_path.exists():
            print(f"Warning: {filename} not found.")
            continue
        
        output_lines.append(f'\n# {"="*20}\n# START OF {filename}\n# {"="*20}\n')
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Strip future/system imports we already put at top
                if 'from __future__ import annotations' in line:
                    continue
                if line.startswith('import ') or (line.startswith('from ') and ' import ' in line):
                    # Check if it's one of OUR modules
                    if pattern.match(line):
                        continue
                    # Skip common imports we already added
                    strip_line = line.strip()
                    if strip_line in seen_imports:
                        # continue # Actually, keeping duplicates inside classes/functions is safer if they exist
                        pass
                    # If it's a new system import, we can keep it or move it to top. 
                    # For simplicity, we'll keep it as long as it's not a local module import.
                
                output_lines.append(line.rstrip())

    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Successfully created {OUTPUT_FILE}")

if __name__ == '__main__':
    consolidate()
