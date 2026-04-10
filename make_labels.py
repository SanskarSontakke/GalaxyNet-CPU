from __future__ import annotations

import argparse
from pathlib import Path

from config import Config
from dataset import generate_labels_df, validate_class_counts, CLASS_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate cleaned 3-class labels CSV from Galaxy Zoo solutions CSV (V25 thresholds).'
    )
    parser.add_argument('--solutions-csv', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--output-csv', type=Path, default=Path('labels.csv'))
    args = parser.parse_args()

    config = Config()
    print('=== GalaxyNet V25 Label Generation ===')
    print(f'  Elliptical threshold: Class1.1 >= {config.elliptical_threshold}')
    print(f'  Spiral threshold:     Class1.2 >= {config.spiral_disk_threshold} AND Class4.1 >= {config.spiral_arms_threshold}')
    print(f'  Irregular threshold:  Class6.1 >= {config.irregular_threshold} AND margin >= {config.irregular_margin}')

    labels_df = generate_labels_df(args.solutions_csv, args.image_dir, config)

    # Validate class counts
    validate_class_counts(labels_df, config)

    # Check minimums
    counts = labels_df['label'].value_counts()
    all_ok = True
    min_counts = {
        'irregular': config.min_irregular_count,
        'spiral': config.min_spiral_count,
        'elliptical': config.min_elliptical_count,
    }
    for cls_name, min_count in min_counts.items():
        actual = counts.get(cls_name, 0)
        if actual < min_count:
            print(f'\n⚠️  {cls_name} count ({actual}) below minimum ({min_count}).')
            print(f'    Consider relaxing threshold by 0.02 and re-running.')
            all_ok = False

    if all_ok:
        print('\n✓ All class counts within acceptable range.')

    labels_df.to_csv(args.output_csv, index=False)

    print(f'\nSaved labels to: {args.output_csv}')
    print('Class distribution:')
    print(labels_df['label'].value_counts().to_string())
    print(f'Total usable labeled images: {len(labels_df)}')


if __name__ == '__main__':
    main()
