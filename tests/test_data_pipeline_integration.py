"""
test_data_pipeline_integration.py
Integration tests for data_pipeline.py — reads from disk.
Requires data pipeline to have run first:
    python backend/data_pipeline.py
Run: pytest tests/test_data_pipeline_integration.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

# FIXTURES

@pytest.fixture(scope="module")
def transactions_df():
    """Load transactions.parquet once for all tests in this module."""
    path = PROCESSED_DIR / "transactions.parquet"
    if not path.exists():
        pytest.skip("transactions.parquet not found — run data_pipeline.py first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def entities_df():
    """Load entities.parquet once for all tests in this module."""
    path = PROCESSED_DIR / "entities.parquet"
    if not path.exists():
        pytest.skip("entities.parquet not found — run data_pipeline.py first")
    return pd.read_parquet(path)

# transactions.parquet tests

class TestTransactionsParquet:

    def test_file_exists(self):
        assert (PROCESSED_DIR / "transactions.parquet").exists()

    def test_file_is_not_empty(self, transactions_df):
        assert len(transactions_df) > 0

    def test_correct_row_count(self, transactions_df):
        """IEEE-CIS has 590,540 transactions."""
        assert len(transactions_df) == 590540

    def test_has_required_columns(self, transactions_df):
        required = ["TransactionID", "TransactionAmt", "isFraud"]
        for col in required:
            assert col in transactions_df.columns, f"Missing column: {col}"

    def test_no_nan_in_numeric_columns(self, transactions_df):
        numeric_cols = transactions_df.select_dtypes(
            include=[np.number]
        ).columns
        total_nan = transactions_df[numeric_cols].isnull().sum().sum()
        assert total_nan == 0, f"Found {total_nan} NaN values in numeric columns"

    def test_isfaud_is_binary(self, transactions_df):
        """isFraud should only contain 0 and 1."""
        unique_values = set(transactions_df["isFraud"].unique())
        assert unique_values.issubset({0, 1})

    def test_transaction_amount_is_positive(self, transactions_df):
        """All transaction amounts should be positive."""
        assert (transactions_df["TransactionAmt"] > 0).all()

    def test_file_size_is_reasonable(self):
        """File should be between 50MB and 200MB."""
        size_mb = (PROCESSED_DIR / "transactions.parquet").stat().st_size / 1024 / 1024
        assert 50 < size_mb < 200, f"Unexpected file size: {size_mb:.1f}MB"

# entities.parquet tests

class TestEntitiesParquet:

    def test_file_exists(self):
        assert (PROCESSED_DIR / "entities.parquet").exists()

    def test_file_is_not_empty(self, entities_df):
        assert len(entities_df) > 0

    def test_has_required_columns(self, entities_df):
        required = [
            "entity_id",
            "transaction_count",
            "total_amount",
            "avg_amount",
            "max_amount",
            "counterparty_count",
            "transaction_types",
            "structuring_flag",
            "velocity_flag",
            "fraud_label",
            "risk_tier",
            "text_description",
        ]
        for col in required:
            assert col in entities_df.columns, f"Missing column: {col}"

    def test_risk_tier_values_are_valid(self, entities_df):
        valid_tiers = {"low", "medium", "high", "critical"}
        actual_tiers = set(entities_df["risk_tier"].unique())
        assert actual_tiers.issubset(valid_tiers), \
            f"Invalid risk tiers found: {actual_tiers - valid_tiers}"

    def test_fraud_count_matches_expected(self, entities_df):
        """PaySim has exactly 8,213 fraud entities."""
        assert entities_df["fraud_label"].sum() == 8213

    def test_fraud_entities_are_critical(self, entities_df):
        """All fraud entities must be risk_tier critical."""
        fraud_entities = entities_df[entities_df["fraud_label"] == 1]
        assert (fraud_entities["risk_tier"] == "critical").all()

    def test_text_description_not_empty(self, entities_df):
        """Every entity must have a non-empty text description."""
        min_length = entities_df["text_description"].str.len().min()
        assert min_length > 50, f"Found very short description: {min_length} chars"

    def test_text_description_contains_entity_id(self, entities_df):
        """Text description must reference the entity ID."""
        sample = entities_df.head(100)
        for _, row in sample.iterrows():
            assert row["entity_id"] in row["text_description"]

    def test_transaction_count_is_positive(self, entities_df):
        assert (entities_df["transaction_count"] > 0).all()

    def test_total_amount_is_positive(self, entities_df):
        assert (entities_df["total_amount"] >= 0).all()

    def test_entity_ids_are_unique(self, entities_df):
        """Each entity_id should appear exactly once."""
        assert entities_df["entity_id"].nunique() == len(entities_df)

    def test_file_size_is_reasonable(self):
        """File should be between 100MB and 300MB."""
        size_mb = (PROCESSED_DIR / "entities.parquet").stat().st_size / 1024 / 1024
        assert 100 < size_mb < 300, f"Unexpected file size: {size_mb:.1f}MB"