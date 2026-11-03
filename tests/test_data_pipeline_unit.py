"""
test_data_pipeline_unit.py
Unit tests for data_pipeline.py — no external dependencies.
Run: pytest tests/test_data_pipeline_unit.py -v
"""

import pytest
import pandas as pd
import numpy as np
from backend.data_pipeline import (
    detect_structuring,
    detect_velocity,
    compute_risk_tier,
    build_entity_description,)


# detect_structuring

class TestDetectStructuring:

    def test_returns_true_for_clear_structuring(self):
        """5 amounts all just below 10k — classic structuring pattern."""
        amounts = pd.Series([9800, 9750, 9900, 9850, 9700])
        assert detect_structuring(amounts) is True

    def test_returns_false_for_legitimate_amounts(self):
        """Mixed amounts with no suspicious pattern."""
        amounts = pd.Series([150, 500, 2000, 8000, 50000])
        assert detect_structuring(amounts) is False

    def test_returns_false_when_too_few_suspicious(self):
        """Only 2 amounts in suspicious range — need at least 3."""
        amounts = pd.Series([9800, 9750, 100, 200, 300, 400, 500])
        assert detect_structuring(amounts) is False

    def test_returns_false_when_not_majority(self):
        """3 suspicious amounts but only 30% of total — need more than 50%."""
        amounts = pd.Series([9800, 9750, 9900, 100, 200, 300, 400, 500, 600, 700])
        assert detect_structuring(amounts) is False

    def test_returns_false_for_amounts_above_10k(self):
        """Amounts above 10k are not structuring."""
        amounts = pd.Series([10001, 10500, 11000, 12000, 15000])
        assert detect_structuring(amounts) is False

    def test_returns_false_for_amounts_below_8k(self):
        """Amounts below 8k are not structuring."""
        amounts = pd.Series([100, 200, 300, 400, 500])
        assert detect_structuring(amounts) is False

# detect_velocity

class TestDetectVelocity:

    def test_returns_true_for_rapid_transactions(self):
        """Transactions at hours 1,1,2 — two gaps of <=2 hours."""
        steps = pd.Series([1, 1, 2, 50, 200])
        assert detect_velocity(steps) is True

    def test_returns_false_for_spread_out_transactions(self):
        """Transactions every 100 hours — clearly not rapid."""
        steps = pd.Series([1, 100, 200, 300, 400])
        assert detect_velocity(steps) is False

    def test_returns_false_when_fewer_than_3_transactions(self):
        """Need at least 3 transactions to detect velocity pattern."""
        steps = pd.Series([1, 2])
        assert detect_velocity(steps) is False

    def test_returns_false_when_only_one_rapid_gap(self):
        """One rapid gap is not enough — need at least 2."""
        steps = pd.Series([1, 2, 100, 200, 300])
        assert detect_velocity(steps) is False

    def test_returns_true_for_same_hour_transactions(self):
        """Multiple transactions in same hour — gap of 0."""
        steps = pd.Series([5, 5, 5, 100, 200])
        assert detect_velocity(steps) is True


# compute_risk_tier

class TestComputeRiskTier:

    def test_critical_when_fraud_label_is_1(self):
        assert compute_risk_tier(False, False, 1, 5) == "critical"

    def test_high_when_both_flags_true(self):
        assert compute_risk_tier(True, True, 0, 5) == "high"

    def test_medium_when_structuring_only(self):
        assert compute_risk_tier(True, False, 0, 5) == "medium"

    def test_medium_when_velocity_only(self):
        assert compute_risk_tier(False, True, 0, 5) == "medium"

    def test_medium_when_high_tx_count(self):
        assert compute_risk_tier(False, False, 0, 25) == "medium"

    def test_low_when_no_flags(self):
        assert compute_risk_tier(False, False, 0, 5) == "low"

    def test_critical_overrides_all_flags(self):
        """fraud_label=1 always returns critical regardless of other flags."""
        assert compute_risk_tier(True, True, 1, 100) == "critical"

    def test_tx_count_boundary_at_20(self):
        """tx_count of exactly 20 is not medium — must be above 20."""
        assert compute_risk_tier(False, False, 0, 20) == "low"

    def test_tx_count_boundary_at_21(self):
        """tx_count of 21 crosses threshold into medium."""
        assert compute_risk_tier(False, False, 0, 21) == "medium"


# build_entity_description

class TestBuildEntityDescription:

    def _make_row(self, structuring=False, velocity=False,
                  risk_tier="medium", tx_count=10):
        """Helper — creates a sample entity row for testing."""
        return pd.Series({
            "entity_id":          "C123456",
            "transaction_count":  tx_count,
            "total_amount":       50000.00,
            "avg_amount":         5000.00,
            "counterparty_count": 3,
            "transaction_types":  "TRANSFER",
            "risk_tier":          risk_tier,
            "structuring_flag":   structuring,
            "velocity_flag":      velocity,
        })

    def test_returns_string(self):
        row = self._make_row()
        assert isinstance(build_entity_description(row), str)

    def test_contains_entity_id(self):
        row = self._make_row()
        assert "C123456" in build_entity_description(row)

    def test_contains_transaction_count(self):
        row = self._make_row(tx_count=10)
        assert "10" in build_entity_description(row)

    def test_contains_structuring_flag_when_true(self):
        row = self._make_row(structuring=True)
        assert "structuring pattern detected" in build_entity_description(row)

    def test_contains_velocity_flag_when_true(self):
        row = self._make_row(velocity=True)
        assert "rapid transaction velocity detected" in build_entity_description(row)

    def test_no_suspicious_patterns_when_no_flags(self):
        row = self._make_row()
        assert "no suspicious patterns detected" in build_entity_description(row)

    def test_both_flags_in_description(self):
        """Both flags present — both should appear in description."""
        row = self._make_row(structuring=True, velocity=True)
        desc = build_entity_description(row)
        assert "structuring pattern detected" in desc
        assert "rapid transaction velocity detected" in desc

    def test_description_is_not_empty(self):
        row = self._make_row()
        assert len(build_entity_description(row)) > 50