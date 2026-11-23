
from pathlib import Path

import pandas as pd
import yaml

from pandas.api.types import (
    is_integer_dtype,
    is_float_dtype,
    is_string_dtype,
    is_object_dtype,
    is_numeric_dtype,
)


class DataValidator:

    def __init__(self, contract_path):
        self.contract_path = Path(contract_path)
        self.contract = self._load_contract()
        self.issues = []

    def _load_contract(self):
        if not self.contract_path.exists():
            raise FileNotFoundError(
                f"Contract not found: {self.contract_path}"
            )

        with open(self.contract_path, "r", encoding="utf-8") as file:
            contract = yaml.safe_load(file)

        if not isinstance(contract, dict) or "datasets" not in contract:
            raise ValueError("Invalid data contract structure")

        return contract

    def _add_issue(self, dataset, rule, count, severity):
        if count > 0:
            self.issues.append({
                "dataset": dataset,
                "rule": rule,
                "count": int(count),
                "severity": severity
            })

    def _check_dtype(self, series, expected_type):

        if expected_type == "integer":
            return is_integer_dtype(series.dtype)

        if expected_type == "float":
            return is_float_dtype(series.dtype)

        if expected_type == "string":
            return (
                is_string_dtype(series.dtype)
                or is_object_dtype(series.dtype)
            )

        raise ValueError(
            f"Unsupported data type in contract: {expected_type}"
        )

    def _validate_schema(self, name, df, config):
        expected_columns = config["required_columns"]

        missing = set(expected_columns) - set(df.columns)
        unexpected = set(df.columns) - set(expected_columns)

        self._add_issue(
            name, "Missing columns", len(missing), "ERROR"
        )

        self._add_issue(
            name, "Unexpected columns", len(unexpected), "WARNING"
        )

        valid_columns = set(df.columns) - missing

        for column, expected_type in expected_columns.items():

            if column not in df.columns:
                continue

            if not self._check_dtype(df[column], expected_type):
                self._add_issue(
                    name,
                    f"Incorrect data type: {column}, expected {expected_type}",
                    1,
                    "ERROR"
                )

                valid_columns.discard(column)

        return valid_columns

    def _validate_nulls(self, name, df, config, valid_columns):
        nullable = set(config.get("nullable_columns", []))

        for column in config["required_columns"]:

            if column not in valid_columns:
                continue

            if column in nullable:
                continue

            missing = df[column].isna().sum()

            self._add_issue(
                name,
                f"Missing values: {column}",
                missing,
                "ERROR"
            )

    def _validate_primary_key(self, name, df, config, valid_columns):
        keys = config.get("primary_key", [])

        if not keys or not set(keys).issubset(valid_columns):
            return

        duplicate_count = df.duplicated(subset=keys).sum()

        self._add_issue(
            name,
            f"Duplicate primary key: {keys}",
            duplicate_count,
            "ERROR"
        )

    def _validate_positive_columns(
        self, name, df, config, valid_columns
    ):
        for column in config.get("positive_columns", []):

            if column not in valid_columns:
                continue

            if not is_numeric_dtype(df[column].dtype):
                continue

            invalid = (df[column].dropna() <= 0).sum()

            self._add_issue(
                name,
                f"Non-positive values: {column}",
                invalid,
                "ERROR"
            )

    def _validate_non_empty_columns(
        self, name, df, config, valid_columns
    ):
        for column in config.get("non_empty_columns", []):

            if column not in valid_columns:
                continue

            if not (
                is_string_dtype(df[column].dtype)
                or is_object_dtype(df[column].dtype)
            ):
                continue

            invalid = (
                df[column]
                .dropna()
                .astype("string")
                .str.strip()
                .eq("")
                .sum()
            )

            self._add_issue(
                name,
                f"Empty values: {column}",
                invalid,
                "ERROR"
            )

    def _validate_rating(self, name, df, config, valid_columns):
        rating_config = config.get("rating")

        if not rating_config:
            return

        column = rating_config["column"]

        if column not in valid_columns:
            return

        if not is_numeric_dtype(df[column].dtype):
            return

        values = df[column].dropna()

        invalid_range = (
            ~values.between(
                rating_config["min"],
                rating_config["max"]
            )
        ).sum()

        increment = rating_config["increment"]

        invalid_increment = (
            (values / increment - (values / increment).round())
            .abs() > 1e-8
        ).sum()

        self._add_issue(
            name, "Invalid rating range",
            invalid_range, "ERROR"
        )

        self._add_issue(
            name, "Invalid rating increments",
            invalid_increment, "ERROR"
        )

    def _validate_timestamps(
        self, name, df, config, valid_columns
    ):
        for column in config.get("timestamp_columns", []):

            if column not in valid_columns:
                continue

            if not is_integer_dtype(df[column].dtype):
                continue

            parsed = pd.to_datetime(
                df[column],
                unit="s",
                errors="coerce",
                utc=True
            )

            invalid = (parsed.isna()| (df[column] < 0)).sum()

            self._add_issue(
                name,
                f"Invalid timestamps: {column}",
                invalid,
                "ERROR"
            )

    def _validate_relationships(self, datasets, valid_columns):
        relationships = self.contract.get("relationships", [])

        for relationship in relationships:

            child = relationship["child_dataset"]
            child_column = relationship["child_column"]
            parent = relationship["parent_dataset"]
            parent_column = relationship["parent_column"]

            if child not in datasets or parent not in datasets:
                continue

            if (
                child_column not in valid_columns.get(child, set())
                or parent_column not in valid_columns.get(parent, set())
            ):
                continue

            child_values = datasets[child][child_column].dropna()
            parent_values = datasets[parent][parent_column].dropna()

            invalid = (~child_values.isin(parent_values)).sum()

            self._add_issue(
                child,
                f"Unknown reference: {child_column} -> {parent}.{parent_column}",
                invalid,
                "ERROR"
            )

    def _validate_metadata(self, datasets, valid_columns):
        metadata_rules = self.contract.get("metadata_rules", {})

        for rule_name, rule in metadata_rules.items():

            dataset = rule["dataset"]
            column = rule["column"]

            if dataset not in datasets:
                continue

            if column not in valid_columns.get(dataset, set()):
                continue

            df = datasets[dataset]

            if rule_name == "duplicate_movie_titles":
                count = df[column].duplicated().sum()

            elif rule_name == "missing_tmdb_ids":
                count = df[column].isna().sum()

            else:
                continue

            self._add_issue(
                dataset,
                rule_name,
                count,
                rule["severity"]
            )

    def run_all(self, datasets):
        self.issues = []

        valid_columns = {}

        for name, config in self.contract["datasets"].items():

            if name not in datasets:
                self._add_issue(
                    name, "Missing dataset", 1, "ERROR"
                )
                continue

            df = datasets[name]

            valid_columns[name] = self._validate_schema(
                name, df, config
            )

            self._validate_nulls(
                name, df, config, valid_columns[name]
            )

            self._validate_primary_key(
                name, df, config, valid_columns[name]
            )

            self._validate_positive_columns(
                name, df, config, valid_columns[name]
            )

            self._validate_non_empty_columns(
                name, df, config, valid_columns[name]
            )

            self._validate_rating(
                name, df, config, valid_columns[name]
            )

            self._validate_timestamps(
                name, df, config, valid_columns[name]
            )

        self._validate_relationships(datasets, valid_columns)
        self._validate_metadata(datasets, valid_columns)

        return self.issues

    def has_errors(self):
        return any(
            issue["severity"] == "ERROR"
            for issue in self.issues
        )
