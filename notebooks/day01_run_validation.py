from src.data.ingestion import DataIngestion
from src.data.validation import DataValidator

from pathlib import Path
import pandas as pd

Base_Dir = Path(__file__).resolve().parents[1]
Data_Dir = Base_Dir / 'data' / 'raw'
Report_Dir = Base_Dir / 'artifacts' / 'reports'

Report_Dir.mkdir(parents = True, exist_ok = True)

# Step01: Ingestion
ingestion = DataIngestion(Data_Dir)
datasets = ingestion.load_all()

# Step02: Validation
validation = DataValidator()
issues = validation.run_all(datasets)

# Step03: Display report
report = pd.DataFrame(issues)

print("\n========== VALIDATION REPORT ==========")

if report.empty:
    print("No validation issues found.")
else:
    print(report.to_string(index=False))

# Step 4: Save report
report_path = Report_Dir / "day02_validation_report.csv"
report.to_csv(report_path, index=False)

# Step 5: Pipeline decision
errors = [
    issue for issue in issues
    if issue["severity"] == "ERROR"
]

if errors:
    print("\nVALIDATION FAILED")
    raise SystemExit(1)

print("\nVALIDATION PASSED")
print(f"Report saved to: {report_path}")
