
from pathlib import Path

import pandas as pd
import pytest

from src.data.validation import DataValidator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "src" / "configs" / "data_contract.yaml"


@pytest.fixture
def validator():
    return DataValidator(CONTRACT_PATH)


@pytest.fixture
def valid_datasets():
    return {
        "movies": pd.DataFrame({
            "movieId": [1, 2],
            "title": ["Toy Story", "Jumanji"],
            "genres": ["Adventure|Animation", "Adventure|Children"],
        }),
        "ratings": pd.DataFrame({
            "userId": [1, 1],
            "movieId": [1, 2],
            "rating": [4.5, 3.0],
            "timestamp": [964982703, 964981247],
        }),
        "tags": pd.DataFrame({
            "userId": [1],
            "movieId": [1],
            "tag": ["fun"],
            "timestamp": [964982703],
        }),
        "links": pd.DataFrame({
            "movieId": [1, 2],
            "imdbId": [114709, 113497],
            "tmdbId": [862.0, 8844.0],
        }),
    }


def get_errors(issues):
    return [issue for issue in issues if issue["severity"] == "ERROR"]


def test_valid_datasets(validator, valid_datasets):
    issues = validator.run_all(valid_datasets)
    assert get_errors(issues) == []


def test_invalid_rating_range(validator, valid_datasets):
    valid_datasets["ratings"].loc[0, "rating"] = 5.5
    issues = validator.run_all(valid_datasets)
    assert get_errors(issues)


def test_invalid_rating_increment(validator, valid_datasets):
    valid_datasets["ratings"].loc[0, "rating"] = 4.3
    issues = validator.run_all(valid_datasets)
    assert get_errors(issues)


def test_duplicate_movie_id(validator, valid_datasets):
    valid_datasets["movies"] = pd.concat(
        [valid_datasets["movies"], valid_datasets["movies"].iloc[[0]]],
        ignore_index=True,
    )
    issues = validator.run_all(valid_datasets)
    assert get_errors(issues)


def test_duplicate_user_movie_pair(validator, valid_datasets):
    valid_datasets["ratings"] = pd.concat(
        [valid_datasets["ratings"], valid_datasets["ratings"].iloc[[0]]],
        ignore_index=True,
    )
    issues = validator.run_all(valid_datasets)
    assert get_errors(issues)


def test_missing_required_value(validator, valid_datasets):
    valid_datasets["movies"].loc[0, "title"] = None
    issues = validator.run_all(valid_datasets)
    assert get_errors(issues)


def test_unknown_movie_reference(validator, valid_datasets):
    valid_datasets["ratings"].loc[0, "movieId"] = 999
    issues = validator.run_all(valid_datasets)
    assert get_errors(issues)


def test_invalid_user_id(validator, valid_datasets):
    valid_datasets["ratings"].loc[0, "userId"] = 0
    issues = validator.run_all(valid_datasets)
    assert get_errors(issues)


def test_invalid_timestamp(validator, valid_datasets):
    valid_datasets["ratings"].loc[0, "timestamp"] = -1
    issues = validator.run_all(valid_datasets)
    assert get_errors(issues)


def test_missing_required_column_does_not_crash(validator, valid_datasets):
    valid_datasets["movies"] = valid_datasets["movies"].drop(columns=["title"])
    issues = validator.run_all(valid_datasets)
    assert get_errors(issues)


def test_wrong_column_type(validator, valid_datasets):
    valid_datasets["movies"]["movieId"] = (
        valid_datasets["movies"]["movieId"].astype(str)
    )
    issues = validator.run_all(valid_datasets)
    assert get_errors(issues)


def test_nullable_tmdb_id_allowed(validator, valid_datasets):
    valid_datasets["links"].loc[0, "tmdbId"] = None
    issues = validator.run_all(valid_datasets)
    assert get_errors(issues) == []
