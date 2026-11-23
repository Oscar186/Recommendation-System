import pandas as pd
from pathlib import Path

class DataIngestion:

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def load_csv(self, filename):
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f'Dataset not found: {file_path}'
            )

        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError(
                f'Dataset is empty: {filename}'
            )

        print(
            f'Successfully loaded {filename}: '
            f'{df.shape[0]} rows, {df.shape[1]} columns'
        )

        return df

    def load_all(self):
        return {
            'movies': self.load_csv('movies.csv'),
            'ratings': self.load_csv('ratings.csv'),
            'tags': self.load_csv('tags.csv'),
            'links': self.load_csv("links.csv")
        }