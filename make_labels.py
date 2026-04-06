import pandas as pd

def determine_class(row):
    # If the crowd strongly voted that the galaxy has odd/irregular features
    if row['Class6.1'] > 0.5:
        return 'irregular'
    # If the crowd strongly voted that the galaxy is smooth and rounded
    elif row['Class1.1'] > 0.5:
        return 'elliptical'
    # If the crowd strongly voted it has a disk AND has spiral arms
    elif row['Class1.2'] > 0.5 and row['Class4.1'] > 0.5:
        return 'spiral'
    else:
        # If the galaxy is ambiguous or a star/artifact, we skip it
        return 'unknown'

def main():
    print("Loading Kaggle solutions...")
    # Update this path if your extracted Kaggle csv is in a different folder
    csv_path = 'galaxy_raw/training_solutions_rev1.csv' 
    df = pd.read_csv(csv_path)

    print("Applying classification logic...")
    df['label'] = df.apply(determine_class, axis=1)

    # Filter out ambiguous images to ensure a high-quality training set
    df_filtered = df[df['label'] != 'unknown'].copy()

    # The Kaggle images are named "<GalaxyID>.jpg", so we format the filename column
    df_filtered['filename'] = df_filtered['GalaxyID'].astype(str) + '.jpg'

    # Save exactly the two columns required by organize_dataset.py
    output_path = 'labels.csv'
    df_filtered[['filename', 'label']].to_csv(output_path, index=False)
    
    print(f"Done! Created {output_path} with {len(df_filtered)} confident labels.")
    print(df_filtered['label'].value_counts())

if __name__ == "__main__":
    main()
