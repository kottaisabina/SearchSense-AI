"""
SearchSense AI - PS-01 Search Term Triage Engine
------------------------------------------------
This module implements the PS-01 Search Term Triage Engine.
Evaluates PPC search terms to produce exactly one decision: PROMOTE, BLOCK, or IGNORE,
along with a human-readable one-line reason, confidence level, review_required flag,
and negative keyword match-type and scoping proposals.

Notice: This module utilizes explainable, transparent heuristic intelligence.
It does NOT claim to be an opaque or unvalidated deep learning model.
"""

from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np


# Default conservative PPC heuristic thresholds for commercial business lending
DEFAULT_THRESHOLDS = {
    "min_spend_waste_threshold": 40.00,  # Spend threshold with 0 conversions to trigger BLOCK
    "target_cpa": 60.00,                 # Benchmark cost per acquisition
    "min_roas_promote": 2.00,            # Minimum Return on Ad Spend to qualify for PROMOTE
    "min_conversions_promote": 2,        # Minimum conversion volume for PROMOTE confidence
    "min_clicks_significance": 15,       # Minimum clicks required for statistical confidence
    "high_cpa_multiplier": 2.5           # Multiplier above target CPA considered severe inefficiency
}


def evaluate_term_record(
    row: Dict[str, Any], 
    thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """
    Evaluates a single search term against conservative PS-01 triage rules.
    
    Returns:
    --------
    Dict[str, Any]:
        - 'decision': 'PROMOTE', 'BLOCK', or 'IGNORE'
        - 'reason': Concise one-line explanation
        - 'confidence': 'High', 'Medium', or 'Low'
        - 'review_required': bool (True if uncertain / low-evidence)
        - 'estimated_historical_waste': float
        - 'scope_recommendation': 'Account-level', 'Campaign-level', or 'Ad-group-level'
        - 'match_type_proposal': '[Exact]', '\"Phrase\"', or 'N/A'
        - 'uncertainty_impact_score': float (for review queue prioritization)
    """
    cost = float(row.get("cost", 0.0))
    clicks = int(row.get("clicks", 0))
    conversions = int(row.get("conversions", 0))
    conversion_value = float(row.get("conversion_value", 0.0))
    roas = float(row.get("roas", 0.0))
    cpa = float(row.get("cpa", 0.0))
    impressions = int(row.get("impressions", 0))
    intent = str(row.get("intent", "General Exploratory"))
    intent_tokens = str(row.get("intent_tokens", "None"))
    is_negative_intent = bool(row.get("is_negative_intent", False))
    match_type = str(row.get("match_type", "broad")).lower()

    t_waste = thresholds.get("min_spend_waste_threshold", 40.0)
    t_cpa = thresholds.get("target_cpa", 60.0)
    t_roas = thresholds.get("min_roas_promote", 2.0)
    t_conv = int(thresholds.get("min_conversions_promote", 2))
    t_clicks = int(thresholds.get("min_clicks_significance", 15))
    t_cpa_mult = thresholds.get("high_cpa_multiplier", 2.5)

    decision = "IGNORE"
    confidence = "Medium"
    review_required = False
    estimated_waste = 0.0
    scope_rec = "Campaign-level"
    match_proposal = "N/A"
    reason = "Normal exploration term within acceptable performance boundaries."

    # -------------------------------------------------------------
    # RULE SET 1: Negative Intent or Wasteful Bleed -> BLOCK
    # -------------------------------------------------------------
    if is_negative_intent:
        decision = "BLOCK"
        confidence = "High"
        estimated_waste = cost
        match_proposal = '"Phrase"'
        scope_rec = "Account-level"
        review_required = False  # Clear-cut negative intent does not require manual review unless huge spend
        
        if "Jobs & Careers" in intent:
            reason = f"Job seeker/career intent detected [{intent_tokens}]. Block as negative phrase to prevent budget bleed."
        elif "Grants & Free Money" in intent:
            reason = f"Free grant/government aid intent detected [{intent_tokens}]. Non-commercial loan query."
        elif "Unsupported Products" in intent:
            reason = f"Unsupported product/consumer query [{intent_tokens}]. Not a commercial financing product."
        else:
            reason = f"Negative intent pattern detected [{intent_tokens}]. Recommend adding as negative keyword."
            
        # If huge spend, flag for human verification
        if cost > 150.0:
            review_required = True
            reason += f" (High spend of ${cost:.2f} flagged for confirmation)."

    elif conversions == 0 and cost >= t_waste:
        decision = "BLOCK"
        confidence = "High" if clicks >= t_clicks else "Medium"
        estimated_waste = cost
        match_proposal = "[Exact]"
        scope_rec = "Campaign-level"
        review_required = cost > 150.0  # High spend needs review confirmation
        reason = f"High spend (${cost:.2f}) with 0 conversions across {clicks} clicks exceeds waste threshold of ${t_waste:.2f}."

    elif conversions > 0 and clicks >= t_clicks and cpa > (t_cpa * t_cpa_mult) and roas < 0.8:
        decision = "BLOCK"
        confidence = "High"
        estimated_waste = max(0.0, cost - (conversions * t_cpa))
        match_proposal = "[Exact]"
        scope_rec = "Ad-group-level"
        review_required = True  # Has conversions, so reviewer should inspect before blocking
        reason = f"Severe CPA bleed (${cpa:.2f} vs ${t_cpa:.2f} target) with unprofitable {roas:.2f}x ROAS."

    # -------------------------------------------------------------
    # RULE SET 2: Proven Performers -> PROMOTE
    # (Conservative: Never promote simply because conversions > 0)
    # -------------------------------------------------------------
    elif (
        conversions >= t_conv and 
        clicks >= t_clicks and 
        (roas >= t_roas or (cpa > 0 and cpa <= t_cpa)) and 
        not is_negative_intent
    ):
        decision = "PROMOTE"
        confidence = "High" if (roas >= t_roas and cpa <= t_cpa) else "Medium"
        review_required = False
        match_proposal = "[Exact]"
        scope_rec = "Campaign-level"
        estimated_waste = 0.0
        reason = f"Strong performer with {conversions} conversions, ${cpa:.2f} CPA, and {roas:.2f}x ROAS across {clicks} clicks. Ideal for exact match scaling."

    # -------------------------------------------------------------
    # RULE SET 3: Small-Sample Conversions / Uncertain -> IGNORE or REVIEW
    # -------------------------------------------------------------
    elif conversions > 0:
        # Conversions exist, but insufficient sample size or borderline metrics
        decision = "IGNORE"
        confidence = "Low"
        review_required = True
        estimated_waste = 0.0
        match_proposal = "N/A"
        
        if clicks < t_clicks:
            reason = f"Early conversion signal ({conversions} conv, {clicks} clicks) lacks statistical significance ({t_clicks} required). Route to review before scaling."
        elif roas < t_roas:
            reason = f"Moderate ROAS ({roas:.2f}x) falls below target promotion threshold ({t_roas:.2f}x). Continue monitoring in existing match type."
        else:
            reason = f"Borderline conversion performance ({conversions} conv, ${cpa:.2f} CPA). Requires human assessment."

    # -------------------------------------------------------------
    # RULE SET 4: Normal Traffic Exploration -> IGNORE
    # -------------------------------------------------------------
    else:
        decision = "IGNORE"
        confidence = "Medium"
        review_required = False
        estimated_waste = 0.0
        match_proposal = "N/A"
        
        # Check for abnormal CTR with high impressions
        if impressions > 200 and clicks == 0:
            review_required = True
            reason = f"High impression volume ({impressions:,}) with 0 clicks on {match_type} match. Low CTR query needs review."
        elif cost > (t_waste * 0.7):
            review_required = True
            reason = f"Spend (${cost:.2f}) approaching waste limit (${t_waste:.2f}) with 0 conversions. Monitor closely."
        else:
            reason = f"Low spend (${cost:.2f}) with 0 conversions; within acceptable exploratory budget. Keep running as-is."

    # Uncertainty impact score for prioritizing human review:
    # High financial volume with low sample size generates highest priority
    if review_required:
        uncertainty_impact_score = round(cost + (conversion_value * 0.15) + (100.0 / (clicks + 1)), 2)
    else:
        uncertainty_impact_score = round(cost * 0.1, 2)

    return {
        "decision": decision,
        "reason": reason,
        "confidence": confidence,
        "review_required": review_required,
        "estimated_historical_waste": estimated_waste,
        "scope_recommendation": scope_rec,
        "match_type_proposal": match_proposal,
        "uncertainty_impact_score": uncertainty_impact_score
    }


def run_decision_engine(
    df: pd.DataFrame, 
    thresholds: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Executes the PS-01 triage decision engine across all rows in the DataFrame.
    Guarantees every row receives exactly one of: PROMOTE, BLOCK, or IGNORE.
    
    Returns:
    --------
    pd.DataFrame:
        Enriched DataFrame with triage decisions and audit fields.
    """
    if df is None or df.empty:
        return df
        
    df_out = df.copy()
    t = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        t.update(thresholds)
        
    # Extract records for fast evaluation
    records = df_out.to_dict(orient="records")
    
    decisions = []
    reasons = []
    confidences = []
    reviews_req = []
    wastes = []
    scopes = []
    match_proposals = []
    uncertainty_scores = []
    
    for r in records:
        eval_res = evaluate_term_record(r, t)
        decisions.append(eval_res["decision"])
        reasons.append(eval_res["reason"])
        confidences.append(eval_res["confidence"])
        reviews_req.append(eval_res["review_required"])
        wastes.append(eval_res["estimated_historical_waste"])
        scopes.append(eval_res["scope_recommendation"])
        match_proposals.append(eval_res["match_type_proposal"])
        uncertainty_scores.append(eval_res["uncertainty_impact_score"])
        
    df_out["decision"] = decisions
    df_out["reason"] = reasons
    df_out["confidence"] = confidences
    df_out["review_required"] = reviews_req
    df_out["estimated_historical_waste"] = wastes
    df_out["scope_recommendation"] = scopes
    df_out["match_type_proposal"] = match_proposals
    df_out["uncertainty_impact_score"] = uncertainty_scores
    
    # Initialize final_decision and reviewer_notes
    if "decision_label" in df_out.columns and df_out["decision_label"].notna().any():
        df_out["final_decision"] = df_out["decision_label"].combine_first(df_out["decision"])
    else:
        df_out["final_decision"] = df_out["decision"]
        
    if "reviewer_notes" not in df_out.columns:
        df_out["reviewer_notes"] = ""
        
    return df_out
