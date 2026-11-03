# Autonomous AML Investigation Agent

Multi-agent system that investigates suspicious-activity alerts and drafts
Suspicious Activity Reports (SARs). Every agent's output is validated before it
is written to shared state, so unverified claims never reach the final report.

**Status: in active development. Phase 1 of 5 is nearly complete.**

## Design

- LangGraph supervisor routing six agents: entity research, regulatory check,
  risk scoring, SAR generation, escalation, observability
- Validation layers: guardrails, structured output (Pydantic), confidence
  calibration, and a critic pass on the SAR
- Qdrant for entity memory, XGBoost + LLM for risk scoring, Postgres audit trail
- Low-confidence or disagreeing cases escalate to a human analyst

## Progress

| Phase | Scope | Status |
|---|---|---|
| 1 | Poetry, Docker Compose (6 services), config, data pipeline, tests | Done, except Qdrant ingestion (script written, not yet run) |
| 2 | AlertState, supervisor, entity + regulatory agents, Kafka | Planned |
| 3 | XGBoost, risk agent, SAR agent, critic | Planned |
| 4 | Escalation, dashboard, cost panel | Planned |
| 5 | CI/CD, AWS deploy, evaluation | Planned |

## What is built so far

- Data pipeline: IEEE-CIS (590k transactions) and PaySim (2.7M accounts)
  converted to Parquet, with entity profiles (structuring and velocity flags,
  risk tier) ready for vector search
- 48 tests (28 unit, 20 integration) for the pipeline
- Typed configuration with Pydantic Settings and validated thresholds
- Docker Compose setup for FastAPI, Streamlit, Qdrant, Postgres, Kafka, ZooKeeper
- FastAPI health endpoint and Streamlit placeholder

## Run the tests

    poetry install --with dev
    pytest tests/test_data_pipeline_unit.py -v

The integration tests need the processed Parquet files, which come from
`python backend/data_pipeline.py`. Raw datasets are not included. Download
IEEE-CIS and PaySim from Kaggle into `data/raw/`.

## Other work

AI Underwriting Copilot (live demo): https://huggingface.co/spaces/rshukla560/underwriting-copilot