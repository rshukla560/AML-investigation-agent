from pydantic_settings import BaseSettings
from pydantic import validator
from typing import Optional


class Settings(BaseSettings):

    # Anthropic — required, no default
    #anthropic_api_key: str

    # Qdrant — defaults work for Docker Compose
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "aml_entities"

    # Agent thresholds — tunable without redeploying
    confidence_threshold: float = 0.75
    disagreement_threshold: float = 0.3
    min_evidence_count: int = 2
    min_citation_count: int = 2
    max_entity_retries: int = 3

    # App
    debug: bool = False
    environment: str = "development"

    @validator("confidence_threshold")
    def confidence_must_be_probability(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence_threshold must be between 0 and 1, got {v}")
        return v

    @validator("disagreement_threshold")
    def disagreement_must_be_probability(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"disagreement_threshold must be between 0 and 1, got {v}")
        return v

    '''@validator("anthropic_api_key")
    def api_key_not_placeholder(cls, v):
        if v == "your_anthropic_key_here":
            raise ValueError("ANTHROPIC_API_KEY is still set to placeholder value")
        return v'''

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Single instance — imported everywhere in the project
settings = Settings()