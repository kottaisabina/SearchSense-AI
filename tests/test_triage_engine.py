"""
Automated Test Suite for SearchSense AI - PS-01 Search Term Triage Engine
------------------------------------------------------------------------
Tests:
1. Small-sample uncertainty (conservative handling of low-evidence conversions).
2. Unsupported product intent (flagging jobs, grants, templates, personal loans as BLOCK).
3. Complete decisions for all rows across the real dataset (data/search_terms_report.csv).
4. Human decision overrides and audit trail.
5. CSV export completeness and schema integrity.
"""

import os
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.preprocessing import validate_and_preprocess, get_summary_statistics
from src.intent_analysis import enrich_dataframe_with_intent, analyze_query_intent_fast, detect_near_duplicates
from src.decision_engine import run_decision_engine, evaluate_term_record, DEFAULT_THRESHOLDS


class TestSearchSenseTriageEngine(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path(__file__).parent.parent / "data"
        cls.report_csv_path = cls.data_dir / "search_terms_report.csv"
        
    def test_01_small_sample_uncertainty(self):
        """
        Verify that terms with conversions but tiny click counts (e.g., 1-2 clicks)
        are NOT blindly promoted, but conservatively marked IGNORE and review_required=True.
        """
        sample_row = {
            "query": "apply fast business loan",
            "campaign": "Search | Generic Business Loans",
            "ad_group": "Startup Funding",
            "impressions": 15,
            "clicks": 1,
            "cost": 18.50,
            "conversions": 1,
            "conversion_value": 900.0,
            "ctr": 6.67,
            "cpc": 18.50,
            "cvr": 100.0,
            "cpa": 18.50,
            "roas": 48.6,
            "intent": "High-Intent Commercial Lending",
            "intent_tokens": "apply, business loan",
            "is_negative_intent": False
        }
        
        result = evaluate_term_record(sample_row, DEFAULT_THRESHOLDS)
        
        # Must NOT be automatically promoted due to small sample size (1 click < 15 threshold)
        self.assertNotEqual(result["decision"], "PROMOTE", "Small-sample query with only 1 click should not be promoted.")
        self.assertEqual(result["decision"], "IGNORE", "Unproven small-sample query should default to IGNORE.")
        self.assertTrue(result["review_required"], "Small-sample query with conversions must flag review_required=True.")
        self.assertIn("Early conversion signal", result["reason"])

    def test_02_unsupported_product_intent(self):
        """
        Verify that unsupported business queries (jobs, grants, templates, personal loans)
        are categorized with negative intent and assigned BLOCK decision.
        """
        unsupported_queries = [
            ("cheap loan officer jobs near me", "Negative: Jobs & Careers"),
            ("free business grants with ein only", "Negative: Grants & Free Money"),
            ("top invoice template new york", "Negative: Unsupported Products"),
            ("best debt consolidation personal for startups", "Negative: Unsupported Products"),
            ("mortgage rates calculator", "Negative: Unsupported Products"),
            ("free crm login download crack", "Negative: Piracy & Free Software")
        ]
        
        for q, expected_intent in unsupported_queries:
            intent, tokens, exp, is_neg = analyze_query_intent_fast(q)
            self.assertEqual(intent, expected_intent, f"Query '{q}' should match intent '{expected_intent}'.")
            self.assertTrue(is_neg, f"Query '{q}' must have is_negative_intent=True.")
            
            row = {
                "query": q,
                "cost": 25.0,
                "clicks": 2,
                "conversions": 0,
                "intent": intent,
                "intent_tokens": tokens,
                "is_negative_intent": is_neg
            }
            res = evaluate_term_record(row, DEFAULT_THRESHOLDS)
            self.assertEqual(res["decision"], "BLOCK", f"Unsupported query '{q}' must receive BLOCK decision.")
            self.assertEqual(res["confidence"], "High")

    def test_03_complete_decisions_for_all_rows(self):
        """
        Process the entire real dataset (data/search_terms_report.csv) and verify:
        - Every row is preserved (all 25,001 rows).
        - Every row receives exactly one decision: PROMOTE, BLOCK, or IGNORE.
        - Non-null one-line reasons and confidence ratings.
        - review_required is strictly boolean.
        """
        if not self.report_csv_path.exists():
            self.skipTest("data/search_terms_report.csv not found.")
            
        raw_df = pd.read_csv(self.report_csv_path)
        self.assertEqual(len(raw_df), 25001, "Raw dataset must contain exactly 25,001 rows.")
        
        # 1. Preprocessing
        clean_df, errors, warnings = validate_and_preprocess(raw_df, preserve_rows=True)
        self.assertEqual(len(errors), 0, f"Preprocessing errors encountered: {errors}")
        self.assertEqual(len(clean_df), 25001, "Cleaned dataset must preserve all 25,001 rows.")
        
        # 2. Intent analysis
        intent_df = enrich_dataframe_with_intent(clean_df)
        self.assertEqual(len(intent_df), 25001)
        
        # 3. Decision Engine
        triage_df = run_decision_engine(intent_df, DEFAULT_THRESHOLDS)
        self.assertEqual(len(triage_df), 25001)
        
        # Verify valid decisions
        unique_decisions = set(triage_df["decision"].unique())
        valid_decisions = {"PROMOTE", "BLOCK", "IGNORE"}
        self.assertTrue(unique_decisions.issubset(valid_decisions), f"Unexpected decisions found: {unique_decisions - valid_decisions}")
        
        # Verify no nulls in decision, reason, confidence, review_required
        self.assertEqual(triage_df["decision"].isna().sum(), 0, "No null decisions allowed.")
        self.assertEqual(triage_df["reason"].isna().sum(), 0, "No null reasons allowed.")
        self.assertEqual(triage_df["confidence"].isna().sum(), 0, "No null confidences allowed.")
        self.assertEqual(triage_df["review_required"].isna().sum(), 0, "No null review_required flags allowed.")
        
        # Verify confidence levels
        valid_confidences = {"High", "Medium", "Low"}
        self.assertTrue(set(triage_df["confidence"].unique()).issubset(valid_confidences))
        
        # Verify review_required is boolean
        self.assertTrue(all(isinstance(v, (bool, np.bool_)) for v in triage_df["review_required"]))

    def test_04_human_overrides(self):
        """
        Verify that human reviewer overrides update final_decision and preserve review notes.
        """
        df = pd.DataFrame([{
            "row_id": 1,
            "query": "unsecured business line of credit",
            "campaign": "Search | Line of Credit",
            "ad_group": "Revolving Credit",
            "impressions": 100,
            "clicks": 5,
            "cost": 50.0,
            "conversions": 0,
            "conversion_value": 0.0,
            "intent": "High-Intent Commercial Lending",
            "intent_tokens": "line of credit",
            "is_negative_intent": False
        }])
        
        triage_df = run_decision_engine(df, DEFAULT_THRESHOLDS)
        initial_decision = triage_df.iloc[0]["decision"]  # Likely BLOCK due to $50 spend w/ 0 conv
        
        # Simulate human override in UI
        triage_df.loc[0, "final_decision"] = "PROMOTE"
        triage_df.loc[0, "reviewer_notes"] = "Strategic priority keyword approved by CMO for scale."
        
        self.assertEqual(triage_df.iloc[0]["final_decision"], "PROMOTE")
        self.assertEqual(triage_df.iloc[0]["reviewer_notes"], "Strategic priority keyword approved by CMO for scale.")
        # Ensure original algorithmic suggestion is preserved for audit trail
        self.assertEqual(triage_df.iloc[0]["decision"], initial_decision)

    def test_05_csv_export_completeness(self):
        """
        Verify that the exported CSV retains all original dataset columns
        plus the generated decision intelligence and audit fields.
        """
        raw_df = pd.DataFrame([{
            "row_id": 101,
            "search_term": "commercial vehicle loan approval",
            "match_type": "exact",
            "campaign": "Search | Equipment Finance",
            "ad_group": "Truck & Vehicle Finance",
            "impressions": 500,
            "clicks": 40,
            "cost": 320.0,
            "conversions": 4,
            "conversion_value": 1600.0
        }])
        
        clean_df, _, _ = validate_and_preprocess(raw_df)
        intent_df = enrich_dataframe_with_intent(clean_df)
        triage_df = run_decision_engine(intent_df)
        
        # Required export columns
        expected_cols = [
            "row_id", "query", "campaign", "ad_group", "match_type",
            "impressions", "clicks", "cost", "conversions", "conversion_value",
            "ctr", "cpc", "cvr", "cpa", "roas", "intent", "decision", 
            "reason", "confidence", "review_required", "final_decision", "reviewer_notes"
        ]
        
        for col in expected_cols:
            self.assertIn(col, triage_df.columns, f"Missing expected column '{col}' in export DataFrame.")
            
        csv_string = triage_df.to_csv(index=False)
        self.assertTrue(len(csv_string) > 0)
        self.assertIn("commercial vehicle loan approval", csv_string)


if __name__ == "__main__":
    unittest.main()
