import pandas as pd
from src.data.validation import DataValidator

def test_valid_rating():

    validator = DataValidator()

    ratings = pd.DataFrame({
        'userId': [1],
        'movieId': [1],
        'rating': [4.5],
        'timestamp': [964982703]
    })

    validator.validate_ratings(ratings)

    assert len(validator.issues) == 0

def test_invalid_rating():
    validator = DataValidator()

    ratings = pd.DataFrame({
        'userId': [1],
        'movieId': [1],
        'rating': [4.5],
        'timestamp': [964982703]
    })

    validator.validate_ratings(ratings)

    assert any(
        issue['rule'] == 'Invalid rating range'
        for issue in validator.issues
    )