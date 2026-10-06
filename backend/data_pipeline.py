"""
data_pipeline.py
Phase 1 — processes raw IEEE-CIS and PaySim data into:
  - data/processed/transactions.parquet  (for XGBoost training)
  - data/processed/entities.parquet      (for Qdrant ingestion)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# IEEE-CIS PIPELINE

def load_ieee_cis() -> pd.DataFrame:
    """Merge transaction and identity files on TransactionID."""
    logger.info("Loading IEEE-CIS transaction file...")
    transactions = pd.read_csv(RAW_DIR / "train_transaction.csv")
    logger.info(f"Transactions shape: {transactions.shape}")

    logger.info("Loading IEEE-CIS identity file...")
    identity = pd.read_csv(RAW_DIR / "train_identity.csv")
    logger.info(f"Identity shape: {identity.shape}")

    logger.info("Merging on TransactionID...")
    df = transactions.merge(identity, on="TransactionID", how="left")
    logger.info(f"Merged shape: {df.shape}")

    return df


def clean_ieee_cis(df: pd.DataFrame) -> pd.DataFrame:
    """Clean merged IEEE-CIS dataframe."""
    logger.info("Cleaning IEEE-CIS data...")

    # drop columns with more than 90% missing values
    threshold = 0.9
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist() #cols with > 90% missing val
    logger.info(f"Dropping {len(cols_to_drop)} columns with >90% missing values")
    df = df.drop(columns=cols_to_drop)

    # fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("isFraud") # get columns excluding isFraud

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())


    # fill categorical columns with "unknown"
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    df[categorical_cols] = df[categorical_cols].fillna("unknown")

    # encode categorical columns as category dtype
    df[categorical_cols] = df[categorical_cols].astype("category")

    logger.info(f"Clean shape: {df.shape}")
    return df


def save_transactions(df: pd.DataFrame) -> None:
    """Save cleaned transactions to parquet."""
    output_path = PROCESSED_DIR / "transactions.parquet"
    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved transactions.parquet — {size_mb:.1f}MB")


def run_ieee_cis_pipeline() -> pd.DataFrame:
    """Run full IEEE-CIS pipeline."""

    logger.info("Starting IEEE-CIS pipeline")

    df = load_ieee_cis()
    df = clean_ieee_cis(df)
    save_transactions(df)
    return df

# PAYSIM PIPELINE 

def load_paysim() -> pd.DataFrame:
    """Load PaySim dataset."""
    logger.info("Loading PaySim dataset...")
    df = pd.read_csv(RAW_DIR / "PS_20174392719_1491204439457_log.csv")
    logger.info(f"PaySim shape: {df.shape}")
    logger.info(f"Transaction types: {df['type'].value_counts().to_dict()}")
    return df


def filter_paysim(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only TRANSFER and CASH-OUT — fraud only occurs here."""
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    logger.info(f"After filtering to TRANSFER+CASH_OUT: {df.shape}")
    return df


def detect_structuring(amounts: pd.Series) -> bool:
    """True if amounts are consistently just below $10,000.
    3 trans between 8k and 10k and total of such tras are more than 50 % of total trna for that entity"""
    below_10k = amounts[(amounts > 8000) & (amounts < 10000)]
    return len(below_10k) >= 3 and (len(below_10k) / len(amounts)) > 0.5 


def detect_velocity(steps: pd.Series) -> bool:
    """True if multiple transactions happen in rapid succession."""
    if len(steps) < 3:
        return False
    time_diffs = steps.sort_values().diff().dropna()
    rapid = time_diffs[time_diffs <= 2]
    return len(rapid) >= 2


def compute_risk_tier(
    structuring: bool,
    velocity: bool,
    fraud_label: int,
    tx_count: int
) -> str:
    """Compute risk tier from flags."""
    if fraud_label == 1:
        return "critical"
    if structuring and velocity:
        return "high"
    if structuring or velocity or tx_count > 20:
        return "medium"
    return "low"


def build_entity_description(row: pd.Series) -> str:
    """Build text description for embedding."""
    flags = []
    if row["structuring_flag"]:
        flags.append("structuring pattern detected")
    if row["velocity_flag"]:
        flags.append("rapid transaction velocity detected")

    flag_text = " and ".join(flags) if flags else "no suspicious patterns detected"

    return (
        f"Account {row['entity_id']} conducted {row['transaction_count']} "
        f"{row['transaction_types']} transactions totalling "
        f"${row['total_amount']:,.2f} across {row['counterparty_count']} "
        f"counterparties. Average transaction amount: "
        f"${row['avg_amount']:,.2f}. Risk tier: {row['risk_tier']}. "
        f"Flags: {flag_text}."
    )


def build_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Group PaySim by account and build entity profiles using vectorised ops."""
    logger.info("Building entity profiles from PaySim...")

    grp = df.groupby("nameOrig")

    # vectorised aggregations — all in one pass
    entities_df = pd.DataFrame({
        "entity_id":          grp["amount"].count().index,
        "transaction_count":  grp["amount"].count().values,
        "total_amount":       grp["amount"].sum().values.round(2),
        "avg_amount":         grp["amount"].mean().values.round(2),
        "max_amount":         grp["amount"].max().values.round(2),
        "counterparty_count": grp["nameDest"].nunique().values,
        "fraud_label":        grp["isFraud"].max().values.astype(int),
    })

    # transaction types per account
    entities_df["transaction_types"] = grp["type"].apply(lambda x: "+".join(sorted(x.unique().tolist()))).values

    # reuse existing functions
    entities_df["structuring_flag"] = grp["amount"].apply(detect_structuring).values

    entities_df["velocity_flag"] = grp["step"].apply(detect_velocity).values

    # risk tier using existing function
    entities_df["risk_tier"] = entities_df.apply(lambda r: compute_risk_tier(
            r["structuring_flag"],
            r["velocity_flag"],
            r["fraud_label"],
            r["transaction_count"]), axis=1)

    entities_df["jurisdiction"] = "UNKNOWN"

    # text description for embedding
    entities_df["text_description"] = entities_df.apply(build_entity_description, axis=1)

    logger.info(f"Built {len(entities_df)} entity profiles")
    logger.info(f"Risk tier distribution:\n{entities_df['risk_tier'].value_counts()}")
    logger.info(f"Fraud entities: {entities_df['fraud_label'].sum()}")

    return entities_df



def save_entities(df: pd.DataFrame) -> None:
    """Save entity profiles to parquet."""
    output_path = PROCESSED_DIR / "entities.parquet"
    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved entities.parquet — {size_mb:.1f}MB")


def run_paysim_pipeline() -> pd.DataFrame:
    """Run full PaySim pipeline."""

    logger.info("Starting PaySim pipeline")

    df = load_paysim()
    df = filter_paysim(df)
    entities_df = build_entities(df)
    save_entities(entities_df)
    return entities_df


# MAIN 

if __name__ == "__main__":
    run_ieee_cis_pipeline()
    run_paysim_pipeline()

    logger.info("Phase 1 data pipeline complete")
