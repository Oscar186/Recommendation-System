
from pathlib import Path

import pandas as pd

from src.data.ingestion import DataIngestion
from src.data.validation import DataValidator


def run_pipeline(data_dir, contract_path, report_path):

    # 1. Initialize validator and load contract
    validator = DataValidator(contract_path)

    # 2. Ingest datasets using contract configuration
    ingestion = DataIngestion(data_dir)

    datasets = ingestion.load_all(
        validator.contract["datasets"]
    )

    # 3. Run validation
    issues = validator.run_all(datasets)

    # 4. Generate report
    report = pd.DataFrame(
        issues,
        columns=["dataset", "rule", "count", "severity"]
    )

    report_path = Path(report_path)
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report.to_csv(report_path, index=False)

    # 5. Determine pipeline status
    errors = [
        issue for issue in issues
        if issue["severity"] == "ERROR"
    ]

    status = "FAILED" if errors else "PASSED"

    print("\n========== VALIDATION REPORT ==========")

    if report.empty:
        print("No validation issues found.")
    else:
        print(report.to_string(index=False))

    print(f"\nPIPELINE STATUS: {status}")
    print(f"Report saved to: {report_path}")

    return {
        "status": status,
        "issues": issues,
        "report_path": report_path
    }
