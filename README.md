# 🎯 SearchSense AI | PS-01 Search Term Triage Engine

> **Transparent, Explainable PPC Search Term Triage & Budget Waste Protection**  
> An enterprise-grade Streamlit application designed for PPC strategists and digital advertisers to evaluate search term queries, prevent ad spend bleed, classify search intent, handle statistical uncertainty, and scale high-performing keywords.

---

## 🌟 Business Context & Core Philosophy

SearchSense AI implements the **PS-01 Search Term Triage Engine**, engineered for high-stakes PPC accounts (specifically configured with domain ontologies for **Commercial Business Lending**, including Business Loans, SBA Loans, Line of Credit, Equipment Financing, Merchant Cash Advance, and Working Capital).

### Key Architectural Tenets:
1. **Conservative Triage Rules**: Every single search term receives exactly one of three decisions:
   - **`PROMOTE`**: Scalable high-performers with proven statistical significance (conversions &ge; 2, clicks &ge; 15, ROAS &ge; 2.0x or CPA &le; $60) and confirmed commercial intent. **Never automatically promoted merely for 1 lucky conversion.**
   - **`BLOCK`**: Explicit waste candidates (irrelevant tokens like jobs, grants, templates, personal loans, or spend &ge; $40 with 0 conversions).
   - **`IGNORE`**: Normal exploratory traffic that does not warrant exact-match promotion or negative keyword blocking. Continues running in existing match types.
2. **Review Required Flag & Uncertainty Prioritization**: Low-evidence queries with early conversions or borderline metrics are routed to the **Human Review Queue** (`review_required = True`), prioritized by potential financial impact.
3. **Near-Duplicate Query Grouping**: High-speed clustering of queries sharing semantic token signatures, word order variations, or common PPC typos (e.g. `financingg`, `advvance`, `lineo f credit`).
4. **Scoped Negative Keyword Proposals**: Suggestions proposed with distinct scopes (`Account-level` for universal negative tokens, `Campaign-level` or `Ad-group-level` for specific query bleed) and match types (`[Exact]` vs. `"Phrase"`).
5. **Human-in-the-Loop Auditability**: Strategists can override decisions, add reviewer rationale notes, and export an audited dataset with complete historical provenance.
6. **Safety & Zero Black-Box Claims**: Built on deterministic, explainable rule heuristics. It **never automatically modifies live Google Ads accounts**, labels historical waste as an **estimate of past inefficiency**, and clearly designates synthetic demo data.

---

## 📑 Application Modules

The application is structured around 6 functional pages:

1. 📊 **Executive Dashboard**:
   - Live KPI cards calculated directly from the active dataset (Total Spend, Revenue, Conversions, CTR, CPC, CVR, CPA, ROAS, and Estimated Past Waste).
   - Interactive Plotly visualizations: Decision Donut Breakdown, Query Intent vs. Spend, Match Type Performance, and Campaign Performance.
2. 🔍 **Search Term Explorer**:
   - Multi-dimensional filters (Keyword search, Campaign, Match Type, Decision, Intent Category, Review Required status, and Spend sliders).
   - Sortable, paginated data grid with Row Inspector showing query-level economics and one-line reasons.
3. 🧠 **Decision Intelligence**:
   - Triage Rule Architecture documentation and transparent criteria.
   - **Uncertainty Prioritization Queue**: Sorts uncertain terms by financial risk and sample size.
   - **Near-Duplicate Query Grouping**: Identifies query clusters with shared token signatures and aggregated spend.
4. 💸 **Budget Waste Detector**:
   - Negative keyword candidate inspection.
   - Account-level phrase match negative lists and Campaign-level exact match lists.
   - Historical spend at risk calculation.
5. ✍️ **Human Review Center**:
   - Queue for terms flagged with `review_required = True`.
   - Interactive override form to update final decision (`PROMOTE`, `BLOCK`, `IGNORE`) and document reviewer notes.
   - Real-time audit log of reviewer modifications.
6. 📥 **Export Center**:
   - Downloadable complete audited CSV containing **every original row** (all 25,001 rows) alongside system decisions, reasons, confidence, `review_required`, `final_decision`, and reviewer notes.
   - Platform export presets: Google Ads Negative Keywords, Exact Match Promotion, and Agency Review Queue.

---

## 📂 Project Architecture

```text
SearchSense-AI/
│
├── app.py                       # Main Streamlit web application & 6-page navigation
├── requirements.txt             # Core Python dependencies (Streamlit, Pandas, Plotly, etc.)
├── README.md                    # Project documentation & execution guide
│
├── data/
│   ├── search_terms_report.csv  # Production dataset (25,001 commercial lending search terms)
│   └── sample_search_terms.csv  # Synthetic demo dataset (clearly marked synthetic)
│
├── src/
│   ├── __init__.py              # Package initializer
│   ├── preprocessing.py         # Robust cleaning, schema validation, and PPC metric derivations
│   ├── intent_analysis.py       # Compiled regex intent classification & near-duplicate grouping
│   └── decision_engine.py       # PS-01 conservative triage engine & uncertainty scoring
│
└── tests/
    ├── __init__.py              # Tests package initializer
    └── test_triage_engine.py    # Automated test suite (5 verification suites)
```

---

## 💻 Exact Commands to Run in VS Code

### 1. Open VS Code Terminal
Open VS Code, press ``Ctrl + ` `` (or `Terminal > New Terminal`), and navigate to the project directory:
```powershell
cd "c:\Users\D E L L\OneDrive\Desktop\SearchSense-AI"
```

### 2. Set Up Virtual Environment *(Recommended)*
- **PowerShell (Windows):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **Command Prompt (CMD):**
  ```cmd
  python -m venv venv
  .\venv\Scripts\activate.bat
  ```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run Automated Test Suite
```powershell
python -m unittest discover tests
```
*Expected output: `Ran 5 tests ... OK`*

### 5. Launch the Streamlit Application
```powershell
streamlit run app.py
```
*The app will launch at `http://localhost:8501`. By default, it loads the 25,001-row real commercial lending report.*

---

## 🧪 Tested Features Summary

| Feature Tested | Test File & Function | Verification Result |
| :--- | :--- | :--- |
| **Small-Sample Uncertainty** | `tests/test_triage_engine.py::test_01_small_sample_uncertainty` | Terms with 1 click & 1 conversion are **NOT** promoted; assigned `IGNORE` with `review_required=True`. |
| **Unsupported Intent Detection** | `tests/test_triage_engine.py::test_02_unsupported_product_intent` | Jobs, grants, templates, personal loans, and crack downloads categorized as negative intent and assigned `BLOCK`. |
| **Complete 25,001-Row Coverage** | `tests/test_triage_engine.py::test_03_complete_decisions_for_all_rows` | All 25,001 rows evaluated; zero null decisions; 100% in {`PROMOTE`, `BLOCK`, `IGNORE`}; non-null reasons and confidences. |
| **Human Decision Overrides** | `tests/test_triage_engine.py::test_04_human_overrides` | Reviewer overrides successfully update `final_decision` and `reviewer_notes` while preserving original system suggestions. |
| **Full CSV Export Integrity** | `tests/test_triage_engine.py::test_05_csv_export_completeness` | Preserves all original columns (`row_id`, `search_term`, `campaign`, `ad_group`, `match_type`, metrics) and appends triage audit columns. |
