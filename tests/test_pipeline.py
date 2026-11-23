
import pandas as pd

from src.data.pipeline import run_pipeline


def test_ingestion_validation_report(tmp_path):

    data_dir = tmp_path / "raw"
    data_dir.mkdir()

    # Create sample datasets
    pd.DataFrame({
        "movieId": [1, 2],
        "title": ["Toy Story", "Jumanji"],
        "genres": ["Animation|Comedy", "Adventure"]
    }).to_csv(data_dir / "movies.csv", index=False)

    pd.DataFrame({
        "userId": [1, 1],
        "movieId": [1, 2],
        "rating": [4.5, 3.0],
        "timestamp": [964982703, 964981247]
    }).to_csv(data_dir / "ratings.csv", index=False)

    pd.DataFrame({
        "userId": [1],
        "movieId": [1],
        "tag": ["funny"],
        "timestamp": [964982703]
    }).to_csv(data_dir / "tags.csv", index=False)

    pd.DataFrame({
        "movieId": [1, 2],
        "imdbId": [114709, 113497],
        "tmdbId": [862.0, 8844.0]
    }).to_csv(data_dir / "links.csv", index=False)

    # Run the actual pipeline
    project_root = tmp_path.parents[0]

    contract_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src" / "configs" / "data_contract.yaml"
    )

    report_path = tmp_path / "reports" / "validation.csv"

    result = run_pipeline(
        data_dir=data_dir,
        contract_path=contract_path,
        report_path=report_path
    )

    # Verify all stages
    assert result["status"] == "PASSED"
    assert report_path.exists()
    assert isinstance(result["issues"], list)
