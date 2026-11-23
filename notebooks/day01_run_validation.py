
from pathlib import Path

from src.data.pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"

CONTRACT_PATH = (
    PROJECT_ROOT / "src" / "configs" / "data_contract.yaml"
)

REPORT_PATH = (
    PROJECT_ROOT / "artifacts" / "reports"
    / "day02_validation_report.csv"
)


result = run_pipeline(
    data_dir=DATA_DIR,
    contract_path=CONTRACT_PATH,
    report_path=REPORT_PATH
)

if result["status"] == "FAILED":
    raise SystemExit(1)
