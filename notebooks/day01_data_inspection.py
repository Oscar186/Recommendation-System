import pandas as pd
from pathlib import Path

Data_DIR = Path('data/raw')

files = ["movies.csv","ratings.csv","tags.csv","links.csv"]

for file in files:
    print("\n" + "=" * 60)
    print(f"DATASET: {file}")
    print("=" * 60)

    csv_path = Data_DIR/file

    df = pd.read_csv(csv_path)

    print("\nShape:", df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nStatistical summary:")
    print(df.describe(include="all"))

