#!/usr/bin/env python3
"""
Steam Sentiment Analysis - Data Validation Module
==================================================

Lightweight Great Expectations-style validation for Steam review data quality.
Validates required columns, null-rate thresholds, and duplicate reviews.

Usage:
    python validate_data.py [--data-path PATH] [--null-threshold FLOAT]

Example:
    python validate_data.py --data-path ./data/reviews.csv --null-threshold 0.05

Artifacts Generated:
- reports/data-quality/schema-check-report.md
- reports/data-quality/null-rate-summary.csv
- reports/data-quality/dataset-validation-notes.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")


# Required columns for Steam review sentiment analysis
REQUIRED_COLUMNS = ["review", "language", "voted_up", "author.playtime_forever"]

# Default null-rate thresholds (configurable)
DEFAULT_NULL_THRESHOLDS = {
    "review": 0.01,  # Reviews should rarely be null
    "language": 0.005,  # Language should almost always be present
    "voted_up": 0.001,  # Sentiment label must be present
    "author.playtime_forever": 0.01,  # Playtime important for analysis
}


class DataValidator:
    """
    Data quality validator for Steam review datasets.
    Implements Great Expectations-style expectations without full GE scaffolding.
    """

    def __init__(
        self,
        output_dir: str = "reports/data-quality",
        null_thresholds: Optional[Dict[str, float]] = None,
        required_columns: Optional[List[str]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.null_thresholds = null_thresholds or DEFAULT_NULL_THRESHOLDS.copy()
        self.required_columns = required_columns or REQUIRED_COLUMNS.copy()
        self.validation_results = {}
        self.check_results = []

    def validate_required_columns(self, df: pd.DataFrame) -> Dict:
        """Check that all required columns exist in the DataFrame."""
        existing_columns = set(df.columns)
        required_set = set(self.required_columns)
        missing = required_set - existing_columns

        result = {
            "check_name": "required_columns",
            "passed": len(missing) == 0,
            "message": f"Missing required columns: {sorted(missing)}"
            if missing
            else "All required columns present",
            "details": {
                "missing_columns": sorted(list(missing)),
                "existing_required": sorted(list(existing_columns & required_set)),
                "required_columns": self.required_columns,
                "total_columns": len(df.columns),
            },
        }
        self.check_results.append(result)
        self.validation_results["required_columns"] = result
        return result

    def validate_null_rates(self, df: pd.DataFrame) -> Dict:
        """Check null rates against configurable thresholds."""
        null_rates = {}
        violations = {}

        for col in self.required_columns:
            if col in df.columns:
                null_rate = df[col].isnull().sum() / len(df)
                null_rates[col] = null_rate

                threshold = self.null_thresholds.get(col, 0.05)
                if null_rate > threshold:
                    violations[col] = {
                        "actual_rate": null_rate,
                        "threshold": threshold,
                        "excess": null_rate - threshold,
                    }

        if violations:
            violation_msgs = [
                f"{col}: {v['actual_rate']:.2%} (threshold: {v['threshold']:.2%})"
                for col, v in violations.items()
            ]
            result = {
                "check_name": "null_rates",
                "passed": False,
                "message": f"Null rate violations: {'; '.join(violation_msgs)}",
                "details": {
                    "violations": violations,
                    "null_rates": null_rates,
                    "thresholds": self.null_thresholds,
                },
            }
        else:
            result = {
                "check_name": "null_rates",
                "passed": True,
                "message": "All null rates within thresholds",
                "details": {
                    "null_rates": null_rates,
                    "thresholds": self.null_thresholds,
                },
            }

        self.check_results.append(result)
        self.validation_results["null_rates"] = result
        return result

    def validate_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate reviews in the dataset."""
        if "review" not in df.columns:
            result = {
                "check_name": "duplicate_reviews",
                "passed": False,
                "message": "Cannot check duplicates: 'review' column missing",
                "details": {"error": "review column not found"},
            }
            self.check_results.append(result)
            self.validation_results["duplicates"] = result
            return result

        duplicate_mask = df["review"].duplicated(keep="first")
        duplicate_count = duplicate_mask.sum()
        duplicate_rate = duplicate_count / len(df)

        duplicate_reviews = df[duplicate_mask]["review"].dropna().head(5).tolist()

        result = {
            "check_name": "duplicate_reviews",
            "passed": True,
            "message": f"Found {duplicate_count:,} duplicate reviews ({duplicate_rate:.2%})",
            "details": {
                "duplicate_count": int(duplicate_count),
                "duplicate_rate": duplicate_rate,
                "total_reviews": len(df),
                "sample_duplicates": duplicate_reviews[:5] if duplicate_reviews else [],
            },
        }

        self.check_results.append(result)
        self.validation_results["duplicates"] = result
        return result

    def validate_schema(
        self, df: pd.DataFrame, expected_columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Validate DataFrame schema against expected structure.

        Args:
            df (pd.DataFrame): DataFrame to validate
            expected_columns (Optional[List[str]]): Expected column names

        Returns:
            Dict: Schema validation results
        """
        schema_info = {
            "columns": [],
            "total_columns": len(df.columns),
            "total_rows": len(df),
            "validation_passed": True,
            "issues": [],
        }

        # Analyze each column
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null_count": df[col].notna().sum(),
                "null_count": df[col].isna().sum(),
                "unique_count": df[col].nunique(),
                "sample_values": df[col].dropna().head(3).tolist()
                if df[col].notna().any()
                else [],
            }

            # Determine validation status
            if df[col].isna().all():
                col_info["status"] = "CRITICAL"
                col_info["issue"] = "Column is entirely null"
                schema_info["validation_passed"] = False
            elif df[col].isna().sum() > len(df) * 0.5:
                col_info["status"] = "WARNING"
                col_info["issue"] = (
                    f"High null rate: {df[col].isna().sum() / len(df) * 100:.1f}%"
                )
            else:
                col_info["status"] = "OK"
                col_info["issue"] = None

            schema_info["columns"].append(col_info)

        # Check expected columns if provided
        if expected_columns:
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                schema_info["issues"].append(
                    f"Missing expected columns: {missing_cols}"
                )
                schema_info["validation_passed"] = False

        self.validation_results["schema"] = schema_info
        return schema_info

    def calculate_null_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate null rate percentages for all columns.

        Args:
            df (pd.DataFrame): DataFrame to analyze

        Returns:
            pd.DataFrame: Null rate summary
        """
        null_summary = pd.DataFrame(
            {
                "column_name": df.columns,
                "total_count": len(df),
                "null_count": df.isna().sum().values,
                "non_null_count": df.notna().sum().values,
                "null_rate_percent": (df.isna().sum() / len(df) * 100).round(2).values,
                "unique_values": df.nunique().values,
            }
        )

        self.validation_results["null_rates"] = null_summary
        return null_summary

    def check_duplicates(
        self, df: pd.DataFrame, subset: Optional[List[str]] = None
    ) -> Dict:
        """
        Check for duplicate records in the dataset.

        Args:
            df (pd.DataFrame): DataFrame to check
            subset (Optional[List[str]]): Columns to consider for duplicates

        Returns:
            Dict: Duplicate analysis results
        """
        duplicate_info = {
            "total_rows": len(df),
            "duplicate_rows": df.duplicated(subset=subset).sum(),
            "unique_rows": (~df.duplicated(subset=subset)).sum(),
            "duplicate_rate_percent": round(
                df.duplicated(subset=subset).sum() / len(df) * 100, 2
            ),
        }

        self.validation_results["duplicates"] = duplicate_info
        return duplicate_info

    def generate_schema_report(self) -> str:
        """
        Generate markdown schema validation report.

        Returns:
            str: Formatted markdown report
        """
        if "schema" not in self.validation_results:
            return "# Schema Check Report\n\nNo schema validation performed."

        schema = self.validation_results["schema"]
        report_lines = [
            "# Schema Check Report",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Dataset Overview\n",
            f"- **Total Columns:** {schema['total_columns']}",
            f"- **Total Rows:** {schema['total_rows']}",
            f"- **Validation Status:** {'✓ PASSED' if schema['validation_passed'] else '✗ FAILED'}",
            f"\n## Column Details\n",
            "| Column Name | Data Type | Non-Null Count | Null Count | Unique Values | Status |",
            "|-------------|-----------|----------------|------------|---------------|--------|",
        ]

        for col in schema["columns"]:
            status_icon = (
                "✓"
                if col["status"] == "OK"
                else ("⚠" if col["status"] == "WARNING" else "✗")
            )
            report_lines.append(
                f"| {col['name']} | {col['dtype']} | {col['non_null_count']} | "
                f"{col['null_count']} | {col['unique_count']} | {status_icon} {col['status']} |"
            )

        if schema["issues"]:
            report_lines.append("\n## Issues Found\n")
            for issue in schema["issues"]:
                report_lines.append(f"- {issue}")

        return "\n".join(report_lines)

    def generate_null_rate_csv(self) -> str:
        """
        Generate CSV null rate summary.

        Returns:
            str: CSV formatted null rate summary
        """
        if "null_rates" not in self.validation_results:
            return "column_name,total_count,null_count,non_null_count,null_rate_percent,unique_values"

        return self.validation_results["null_rates"].to_csv(index=False)

    def generate_validation_notes(self) -> str:
        """
        Generate markdown validation notes with findings and recommendations.

        Returns:
            str: Formatted markdown validation notes
        """
        report_lines = [
            "# Dataset Validation Notes",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Summary\n",
        ]

        # Schema findings
        if "schema" in self.validation_results:
            schema = self.validation_results["schema"]
            report_lines.append("### Schema Validation\n")
            report_lines.append(
                f"- **Status:** {'PASSED' if schema['validation_passed'] else 'FAILED'}"
            )
            report_lines.append(f"- **Columns Analyzed:** {schema['total_columns']}")

            critical_cols = [c for c in schema["columns"] if c["status"] == "CRITICAL"]
            warning_cols = [c for c in schema["columns"] if c["status"] == "WARNING"]

            if critical_cols:
                report_lines.append(
                    f"- **Critical Issues:** {len(critical_cols)} columns with critical problems"
                )
            if warning_cols:
                report_lines.append(
                    f"- **Warnings:** {len(warning_cols)} columns with high null rates"
                )

        # Null rate findings
        if "null_rates" in self.validation_results:
            null_df = self.validation_results["null_rates"]
            report_lines.append("\n### Null Rate Analysis\n")

            high_null_cols = null_df[null_df["null_rate_percent"] > 50]
            if len(high_null_cols) > 0:
                report_lines.append(
                    f"- **High Null Rate Columns:** {len(high_null_cols)} columns with >50% nulls"
                )
                for _, row in high_null_cols.iterrows():
                    report_lines.append(
                        f"  - {row['column_name']}: {row['null_rate_percent']}% null"
                    )
            else:
                report_lines.append("- **No columns with critical null rates (>50%)**")

        # Duplicate findings
        if "duplicates" in self.validation_results:
            dup = self.validation_results["duplicates"]
            report_lines.append("\n### Duplicate Analysis\n")
            report_lines.append(f"- **Total Rows:** {dup['total_rows']}")
            report_lines.append(
                f"- **Duplicate Rows:** {dup['duplicate_rows']} ({dup['duplicate_rate_percent']}%)"
            )
            report_lines.append(f"- **Unique Rows:** {dup['unique_rows']}")

        # Recommendations
        report_lines.append("\n## Recommendations\n")

        if "schema" in self.validation_results:
            critical_cols = [
                c
                for c in self.validation_results["schema"]["columns"]
                if c["status"] == "CRITICAL"
            ]
            if critical_cols:
                report_lines.append("### Critical Actions Required")
                for col in critical_cols:
                    report_lines.append(
                        f"- **{col['name']}**: {col['issue']} - Consider removing or investigating this column"
                    )

        report_lines.append("\n### General Recommendations")
        report_lines.append(
            "1. Review columns with high null rates for potential data quality issues"
        )
        report_lines.append(
            "2. Investigate duplicate records to determine if they are valid or need removal"
        )
        report_lines.append(
            "3. Validate data types match expected schema for downstream processing"
        )
        report_lines.append(
            "4. Document any data quality decisions made during analysis"
        )

        return "\n".join(report_lines)

    def save_artifacts(self) -> Dict[str, Path]:
        """
        Save all validation artifacts to output directory.

        Returns:
            Dict[str, Path]: Dictionary of artifact names and file paths
        """
        artifacts = {}

        # Save schema report
        schema_report_path = self.output_dir / "schema-check-report.md"
        schema_report_path.write_text(self.generate_schema_report())
        artifacts["schema_report"] = schema_report_path

        # Save null rate summary
        null_rate_path = self.output_dir / "null-rate-summary.csv"
        null_rate_path.write_text(self.generate_null_rate_csv())
        artifacts["null_rate_summary"] = null_rate_path

        # Save validation notes
        validation_notes_path = self.output_dir / "dataset-validation-notes.md"
        validation_notes_path.write_text(self.generate_validation_notes())
        artifacts["validation_notes"] = validation_notes_path

        return artifacts

    def validate_and_report(
        self,
        df: pd.DataFrame,
        expected_columns: Optional[List[str]] = None,
        duplicate_subset: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """
        Run full validation pipeline and generate all artifacts.

        Args:
            df (pd.DataFrame): DataFrame to validate
            expected_columns (Optional[List[str]]): Expected column names
            duplicate_subset (Optional[List[str]]): Columns for duplicate detection

        Returns:
            Dict[str, Path]: Dictionary of generated artifact paths
        """
        print("Running data validation pipeline...")

        # Run validations
        print("  - Validating schema...")
        self.validate_schema(df, expected_columns)

        print("  - Calculating null rates...")
        self.calculate_null_rates(df)

        print("  - Checking for duplicates...")
        self.check_duplicates(df, duplicate_subset)

        # Generate and save artifacts
        print("  - Generating artifacts...")
        artifacts = self.save_artifacts()

        print(f"\n✓ Validation complete. {len(artifacts)} artifacts generated:")
        for name, path in artifacts.items():
            print(f"  - {name}: {path}")

        return artifacts

    def run_all_checks(self, df: pd.DataFrame) -> List[Dict]:
        """Run all three validation checks in sequence."""
        self.check_results = []
        self.validate_required_columns(df)
        self.validate_null_rates(df)
        self.validate_duplicates(df)
        return self.check_results

    def get_summary(self) -> Dict:
        """Get a summary of all validation results."""
        passed = sum(1 for r in self.check_results if r.get("passed", False))
        failed = len(self.check_results) - passed
        return {
            "total_checks": len(self.check_results),
            "passed": passed,
            "failed": failed,
            "all_passed": failed == 0,
        }

    def validate_and_report(
        self,
        df: pd.DataFrame,
        expected_columns: Optional[List[str]] = None,
        duplicate_subset: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """Run full validation pipeline and generate all artifacts."""
        print("Running data validation pipeline...")

        print(" - Validating required columns...")
        self.validate_required_columns(df)

        print(" - Checking null rates against thresholds...")
        self.validate_null_rates(df)

        print(" - Detecting duplicate reviews...")
        self.validate_duplicates(df)

        print(" - Generating artifacts...")
        self.validate_schema(df, expected_columns)
        self.calculate_null_rates(df)
        artifacts = self.save_artifacts()

        print(f"\n✓ Validation complete. {len(artifacts)} artifacts generated:")
        for name, path in artifacts.items():
            print(f" - {name}: {path}")

        return artifacts


def print_check_results(results: List[Dict], summary: Dict) -> None:
    """Print validation results to console with clear formatting."""
    print("\n" + "=" * 60)
    print("STEAM REVIEW DATA VALIDATION REPORT")
    print("=" * 60)

    for result in results:
        status = "✓ PASS" if result.get("passed", False) else "✗ FAIL"
        print(f"\n[{result.get('check_name', 'UNKNOWN').upper()}] {status}")
        print(f"  {result.get('message', 'No message')}")

        details = result.get("details", {})
        if details:
            if "null_rates" in details:
                print("  Null rates:")
                for col, rate in details["null_rates"].items():
                    threshold = details.get("thresholds", {}).get(col, "N/A")
                    threshold_str = (
                        f"{threshold:.2%}"
                        if isinstance(threshold, float)
                        else str(threshold)
                    )
                    print(f"    - {col}: {rate:.2%} (threshold: {threshold_str})")

            if "duplicate_count" in details:
                print(f"  Total reviews: {details['total_reviews']:,}")
                print(f"  Duplicates: {details['duplicate_count']:,}")
                if details.get("sample_duplicates"):
                    print("  Sample duplicates:")
                    for i, dup in enumerate(details["sample_duplicates"][:3], 1):
                        preview = str(dup)[:50] + "..." if len(str(dup)) > 50 else dup
                        print(f"    {i}. {preview}")

    print("\n" + "-" * 60)
    print(f"SUMMARY: {summary['passed']}/{summary['total_checks']} checks passed")

    if summary["all_passed"]:
        print("STATUS: All validations passed ✓")
    else:
        print(f"STATUS: {summary['failed']} validation(s) failed ✗")

    print("=" * 60 + "\n")


def create_sample_data(output_path: str, n_rows: int = 100) -> pd.DataFrame:
    """Create a sample dataset for testing the validator."""
    np.random.seed(42)

    languages = ["english", "schinese", "russian", "spanish", "french", "german"]
    reviews = [
        "Great game, loved the story!",
        "Too short for the price.",
        "Amazing soundtrack and visuals.",
        "Would recommend to friends.",
        "Not worth the money.",
        "Fantastic gameplay mechanics.",
        "Buggy but fun.",
        "Perfect for casual gaming.",
        "Disappointed with the ending.",
        "One of the best games I've played.",
    ]

    data = {
        "review": np.random.choice(reviews, n_rows),
        "language": np.random.choice(languages, n_rows),
        "voted_up": np.random.choice([True, False], n_rows, p=[0.85, 0.15]),
        "author.playtime_forever": np.random.exponential(50, n_rows).astype(int),
        "timestamp_created": np.random.randint(1600000000, 1700000000, n_rows),
        "author.steamid": np.random.randint(10000000, 99999999, n_rows),
    }

    null_indices = np.random.choice(n_rows, size=int(n_rows * 0.02), replace=False)
    data["review"] = np.where(
        np.isin(np.arange(n_rows), null_indices), None, data["review"]
    )

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Sample data created: {output_path} ({n_rows} rows)")
    return df


def main():
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(
        description="Validate Steam review data quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_data.py --data-path reviews.csv
  python validate_data.py --data-path reviews.csv --null-threshold 0.10
  python validate_data.py --create-sample
        """,
    )

    parser.add_argument(
        "--data-path", type=str, default=None, help="Path to the CSV file to validate"
    )

    parser.add_argument(
        "--null-threshold",
        type=float,
        default=None,
        help="Override null threshold for all columns (0.0-1.0)",
    )

    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample dataset for testing",
    )

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=100,
        help="Number of rows for sample data (default: 100)",
    )

    args = parser.parse_args()

    if args.create_sample:
        sample_path = Path(__file__).parent / "sample_data.csv"
        create_sample_data(str(sample_path), args.sample_rows)
        return 0

    if not args.data_path:
        print("\n" + "=" * 60)
        print("╔══════════════════════════════════════════════════════╗")
        print("║              🎮 DEMO MODE ACTIVE 🎮                  ║")
        print("╚══════════════════════════════════════════════════════╝")
        print("=" * 60)
        print("\nThis is a demonstration of the data validation framework.")
        print("Sample data with intentional issues will be validated to")
        print("show how the framework detects and reports data quality problems.")
        print(
            "\nTo validate actual data, use: python validate_data.py --data-path <file>"
        )
        print("=" * 60 + "\n")

        sample_data = {
            "review": [
                "Great game!",
                "Not recommended",
                "Amazing experience",
                None,
                "Good",
            ],
            "language": ["english", "english", "schinese", "english", "russian"],
            "voted_up": [True, False, True, True, True],
            "author.playtime_forever": [120.5, 45.2, 200.0, 88.3, 150.0],
        }
        df = pd.DataFrame(sample_data)

        validator = DataValidator(output_dir="reports/data-quality")
        results = validator.run_all_checks(df)
        summary = validator.get_summary()

        print_check_results(results, summary)

        validator.validate_schema(df, REQUIRED_COLUMNS)
        validator.calculate_null_rates(df)
        validator.check_duplicates(df)
        validator.save_artifacts()

        # Demo mode: always exit cleanly (it's a demo, not a real failure)
        print("\n" + "-" * 60)
        print("📋 DEMO COMPLETE")
        print("-" * 60)
        print("This was a demonstration with sample data containing intentional")
        print("issues to showcase validation capabilities. In production use,")
        print("validation failures would result in exit code 1.")
        print("-" * 60 + "\n")
        return 0

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return 1

    print(f"Loading data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    thresholds = None
    if args.null_threshold is not None:
        if not 0 <= args.null_threshold <= 1:
            print("Error: --null-threshold must be between 0.0 and 1.0")
            return 1
        thresholds = {col: args.null_threshold for col in REQUIRED_COLUMNS}

    validator = DataValidator(null_thresholds=thresholds)
    results = validator.run_all_checks(df)
    summary = validator.get_summary()

    print_check_results(results, summary)

    validator.validate_schema(df, REQUIRED_COLUMNS)
    validator.calculate_null_rates(df)
    validator.check_duplicates(df)
    validator.save_artifacts()

    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
