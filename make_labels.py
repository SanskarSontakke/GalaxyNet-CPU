from __future__ import annotations

import argparse
from pathlib import Path

from config import Config
from dataset import generate_labels_df


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate cleaned 3-class labels CSV from Galaxy Zoo solutions CSV.')
    parser.add_argument('--solutions-csv', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--output-csv', type=Path, default=Path('labels.csv'))
    args = parser.parse_args()

    config = Config()
    labels_df = generate_labels_df(args.solutions_csv, args.image_dir, config)
    labels_df.to_csv(args.output_csv, index=False)

    print(f'Saved labels to: {args.output_csv}')
    print('Class distribution:')
    print(labels_df['label'].value_counts())
    print(f'Total usable labeled images: {len(labels_df)}')


if __name__ == '__main__':
    main()
