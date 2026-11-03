"""
qdrant_ingest.py
Phase 1: 
 loads them into Qdrant.
Run from project root: python -m backend.qdrant_ingest
"""

import uuid
import logging
import pandas as pd
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from backend.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENTITIES_PATH = Path("data/processed/entities.parquet")
MODEL_NAME    = "all-MiniLM-L6-v2"
VECTOR_SIZE   = 384
SAMPLE_SIZE   = 50_000
BATCH_SIZE    = 256
RANDOM_SEED   = 42

PAYLOAD_COLUMNS = [
    "entity_id", "transaction_count", "total_amount", "avg_amount",
    "max_amount", "counterparty_count", "transaction_types",
    "structuring_flag", "velocity_flag", "fraud_label",
    "risk_tier", "jurisdiction", "text_description",
]


def make_point_id(entity_id: str) -> str:
    """Qdrant IDs must be int or UUID, so derive a stable UUID from entity_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, entity_id))


def build_payload(row: pd.Series) -> dict:
    """Convert one entity row into a JSON-safe payload dict."""
    return {
        "entity_id":          str(row["entity_id"]),
        "transaction_count":  int(row["transaction_count"]),
        "total_amount":       float(row["total_amount"]),
        "avg_amount":         float(row["avg_amount"]),
        "max_amount":         float(row["max_amount"]),
        "counterparty_count": int(row["counterparty_count"]),
        "transaction_types":  str(row["transaction_types"]),
        "structuring_flag":   bool(row["structuring_flag"]),
        "velocity_flag":      bool(row["velocity_flag"]),
        "fraud_label":        int(row["fraud_label"]),
        "risk_tier":          str(row["risk_tier"]),
        "jurisdiction":       str(row["jurisdiction"]),
        "text_description":   str(row["text_description"]),
    }


def load_entity_sample() -> pd.DataFrame:
    """All fraud entities + random sample of non-fraud, SAMPLE_SIZE total."""
    df = pd.read_parquet(ENTITIES_PATH, columns=PAYLOAD_COLUMNS)
    fraud = df[df["fraud_label"] == 1]
    legit = df[df["fraud_label"] == 0].sample(n=SAMPLE_SIZE - len(fraud), random_state=RANDOM_SEED)

    sample = pd.concat([fraud, legit]).sample(frac=1, random_state=RANDOM_SEED)
    logger.info(f"Sample: {len(sample)} entities ({len(fraud)} fraud)")
    return sample.reset_index(drop=True)


def ensure_collection(client: QdrantClient) -> None:
    """Create collection (cosine, 384 dims) and payload indexes if missing."""
    name = settings.qdrant_collection
    if client.collection_exists(name):
        logger.info(f"Collection '{name}' already exists")
        return

    client.create_collection(collection_name=name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE))
    
    # payload indexes make pre-filtering fast
    client.create_payload_index(name, "risk_tier", PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "fraud_label", PayloadSchemaType.INTEGER)
    logger.info(f"Created collection '{name}'")


def ingest() -> None:
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    ensure_collection(client)

    df = load_entity_sample()

    logger.info(f"Loading embedding model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Ingesting"):
        batch = df.iloc[start:start + BATCH_SIZE]

        vectors = model.encode(
            batch["text_description"].tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )

        points = [
            PointStruct(
                id=make_point_id(row["entity_id"]),
                vector=vec.tolist(),
                payload=build_payload(row),
            )
            for (_, row), vec in zip(batch.iterrows(), vectors)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points)

    count = client.count(settings.qdrant_collection).count
    logger.info(f"Done. Qdrant now holds {count} points")


if __name__ == "__main__":
    ingest()