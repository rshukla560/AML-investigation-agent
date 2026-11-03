"""
dashboard.py
Phase 1 — Streamlit placeholder dashboard.
Full dashboard built in Phase 4.
"""

import streamlit as st
from datetime import datetime

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Investigation Agent",
    page_icon="🔍",
    layout="wide"
)

# ── header ─────────────────────────────────────────────────────────────────
st.title("🔍 AML Investigation Agent")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.divider()

# ── system status ──────────────────────────────────────────────────────────
st.subheader("System Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Phase", "1 — Infrastructure", delta="Complete")

with col2:
    st.metric("Agents Built", "0 / 6", delta="Phase 2 next")

with col3:
    st.metric("Entities in Qdrant", "2,768,630", delta="Loaded")

with col4:
    st.metric("Status", "Building", delta=None)

st.divider()

# ── roadmap ────────────────────────────────────────────────────────────────
st.subheader("Build Roadmap")

phases = {
    "Phase 1 — Infrastructure":     "✅ Complete — Docker, config, data pipeline, Qdrant ingestion",
    "Phase 2 — Agents 1 & 2":       "🔄 Next — Entity Agent, Regulatory Agent, AlertState, Supervisor",
    "Phase 3 — Risk & SAR":         "⏳ Pending — XGBoost, Risk Agent, SAR Agent, Critic pattern",
    "Phase 4 — Dashboard & Escalation": "⏳ Pending — Full dashboard, escalation queue, cost panel",
    "Phase 5 — Deploy & Observe":   "⏳ Pending — AWS, CI/CD, RAGAS eval, HuggingFace demo",
}

for phase, description in phases.items():
    st.markdown(f"**{phase}**")
    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{description}")
    st.markdown("")

st.divider()

# ── placeholder sections ───────────────────────────────────────────────────
st.subheader("Investigation Dashboard")
st.info("🔄 Coming in Phase 4 — Alert list, SAR viewer, validation status per agent")

st.subheader("Cost Panel")
st.info("🔄 Coming in Phase 4 — Cost per investigation vs manual analyst ($0.08 vs $150)")

st.subheader("Escalation Queue")
st.info("🔄 Coming in Phase 4 — Human analyst queue, override tracking")