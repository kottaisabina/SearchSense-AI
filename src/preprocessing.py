"""
SearchSense AI - Data Preprocessing & Validation Module
------------------------------------------------------
This module validates, cleans, standardizes, and computes core PPC metrics 
(CTR, CPC, CVR, CPA, ROAS) for search term performance datasets.
Optimized for high-performance processing of large search term reports (25,000+ rows).
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np


# Required columns for the analysis
REQUIRED_COLUMNS = [
    "query",
    "campaign",
    "impressions",
    "clicks",
    "cost",
    "conversions",
    "conversion_value"
]

# Mapping dictionary to assist flexible CSV imports with common PPC column variations
COLUMN_SYNONYMS = {
    "search term": "query",
    "search_term": "query",
    "search_terms": "query",
    "keyword": "query",
    "query": "query",
    "term": "query",
    "row_id": "row_id",
    "id": "row_id",
    "match type": "match_type",
    "match_type": "match_type",
    "ad group": "ad_group",
    "ad_group": "ad_group",
    "adgroup": "ad_group",
    "campaign name": "campaign",
    "campaign_name": "campaign",
    "campaign": "campaign",
    "impr": "impressions",
    "impr.": "impressions",
    "impressions": "impressions",
    "clicks": "clicks",
    "cost": "cost",
    "spend": "cost",
    "conversions": "conversions",
    "conv": "conversions",
    "conv.": "conversions",
    "conversion value": "conversion_value",
    "conversion_value": "conversion_value",
    "conv value": "conversion_value",
    "conv. value": "conversion_value",
    "revenue": "conversion_value",
    "decision_label": "decision_label",
    "decision": "decision_label"
}


def clean_currency_or_number(val: Any) -> float:
    """
    Safely parses numeric values that may contain currency symbols, commas, or percent signs.
    Example: '$1,250.50' -> 1250.50
    """
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    
    val_str = str(val).strip()
    for char in ["$", "€", "£", "₹", ",", "%", " "]:
        val_str = val_str.replace(char, "")
    
    try:
        return float(val_str)
    except ValueError:
        return 0.0


def validate_and_preprocess(
    df_raw: pd.DataFrame, 
    preserve_rows: bool = True
) -> Tuple[Optional[pd.DataFrame], List[str], List[str]]:
    """
    Validates uploaded DataFrame, cleans data, preserves all original rows,
    and computes derived advertising metrics.
    
    Parameters:
    -----------
    df_raw : pd.DataFrame
        The raw uploaded DataFrame.
    preserve_rows : bool
        If True (default for PS-01), preserves every row intact without aggregating duplicates.
        
    Returns:
    --------
    Tuple[Optional[pd.DataFrame], List[str], List[str]]:
        - Cleaned DataFrame with computed metrics (or None if critical error)
        - List of error messages (critical blockers)
        - List of warning messages (non-critical notifications for the user)
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    if df_raw is None or df_raw.empty:
        errors.append("The uploaded file is empty. Please provide a valid CSV with search term data.")
        return None, errors, warnings
    
    # Create working copy
    df = df_raw.copy()
    
    # 1. Normalize column headers (strip spaces and lowercase)
    normalized_cols = {col: str(col).strip().lower() for col in df.columns}
    df.rename(columns=normalized_cols, inplace=True)
    
    # Map common aliases to canonical names
    mapped_cols = {}
    for col in df.columns:
        if col in COLUMN_SYNONYMS:
            mapped_cols[col] = COLUMN_SYNONYMS[col]
    df.rename(columns=mapped_cols, inplace=True)
    
    # 2. Check for missing required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(
            f"Missing required column(s): {', '.join(missing_cols)}. "
            f"Expected columns include: {', '.join(REQUIRED_COLUMNS)}."
        )
        return None, errors, warnings
        
    # 3. Clean string columns
    df["query"] = df["query"].astype(str).str.strip()
    df["campaign"] = df["campaign"].astype(str).str.strip()
    
    # Ensure ad_group and match_type exist
    if "ad_group" not in df.columns:
        df["ad_group"] = "General"
    else:
        df["ad_group"] = df["ad_group"].fillna("General").astype(str).str.strip()
        
    if "match_type" not in df.columns:
        df["match_type"] = "broad"
    else:
        df["match_type"] = df["match_type"].fillna("broad").astype(str).str.strip().str.lower()
        
    # Ensure row_id exists
    if "row_id" not in df.columns:
        df["row_id"] = np.arange(1, len(df) + 1)
        
    # Check for empty queries (do not drop rows unless completely blank)
    empty_queries = df["query"].isin(["", "nan", "none", "null", "undefined"])
    if empty_queries.sum() > 0:
        warnings.append(f"Found {empty_queries.sum()} row(s) with blank query strings. Replacing with '[unspecified_query]'.")
        df.loc[empty_queries, "query"] = "[unspecified_query]"
        
    # 4. Clean and parse numeric columns using fast vectorization
    numeric_cols = ["impressions", "clicks", "cost", "conversions", "conversion_value"]
    for col in numeric_cols:
        if df[col].dtype == object:
            df[col] = df[col].apply(clean_currency_or_number)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            
        # Ensure values are non-negative
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            warnings.append(f"Found {neg_count} negative value(s) in '{col}'. Converted them to 0.0.")
            df[col] = df[col].clip(lower=0.0)
            
    # Round impressions and clicks to integers
    df["impressions"] = df["impressions"].round().astype(int)
    df["clicks"] = df["clicks"].round().astype(int)
    
    # Adjust impressions if clicks exceed impressions
    inconsistent_clicks = df["clicks"] > df["impressions"]
    if inconsistent_clicks.sum() > 0:
        warnings.append(
            f"Found {inconsistent_clicks.sum()} row(s) where clicks exceeded impressions. "
            f"Adjusted impressions to match clicks."
        )
        df.loc[inconsistent_clicks, "impressions"] = df.loc[inconsistent_clicks, "clicks"]
        
    # Check for duplicate queries across campaign and ad_group
    dup_count = df.duplicated(subset=["campaign", "ad_group", "query"], keep=False).sum()
    if dup_count > 0:
        warnings.append(f"Detected {dup_count} duplicate row(s) across campaign, ad group, and query. Preserved all rows with unique row IDs.")
        
    # 5. Fast vectorized derivation of PPC performance metrics
    impr = df["impressions"].values
    clicks = df["clicks"].values
    cost = df["cost"].values
    conv = df["conversions"].values
    val = df["conversion_value"].values
    
    # CTR = (clicks / impressions) * 100
    df["ctr"] = np.where(impr > 0, (clicks / impr) * 100, 0.0)
    
    # CPC = cost / clicks
    df["cpc"] = np.where(clicks > 0, cost / clicks, 0.0)
    
    # CVR = (conversions / clicks) * 100
    df["cvr"] = np.where(clicks > 0, (conv / clicks) * 100, 0.0)
    
    # CPA = cost / conversions (cost if conversions == 0)
    df["cpa"] = np.where(conv > 0, cost / conv, 0.0)
    
    # ROAS = conversion_value / cost
    df["roas"] = np.where(cost > 0, val / cost, 0.0)
    
    # Profit Estimate = conversion_value - cost
    df["profit_estimate"] = val - cost
    
    # Handle decision_label if present
    if "decision_label" not in df.columns:
        df["decision_label"] = None
    else:
        df["decision_label"] = df["decision_label"].fillna("").astype(str).str.strip().str.upper()
        df["decision_label"] = df["decision_label"].replace("", None)
        
    return df, errors, warnings


def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes overall performance totals from the cleaned dataframe.
    Calculates dynamic metrics without hardcoded totals.
    """
    if df is None or df.empty:
        return {
            "total_terms": 0,
            "total_impressions": 0,
            "total_clicks": 0,
            "total_cost": 0.0,
            "total_conversions": 0,
            "total_conversion_value": 0.0,
            "overall_ctr": 0.0,
            "overall_cpc": 0.0,
            "overall_cvr": 0.0,
            "overall_cpa": 0.0,
            "overall_roas": 0.0,
            "overall_profit": 0.0
        }
        
    total_impr = int(df["impressions"].sum())
    total_clicks = int(df["clicks"].sum())
    total_cost = float(df["cost"].sum())
    total_conv = int(df["conversions"].sum())
    total_conv_val = float(df["conversion_value"].sum())
    
    overall_ctr = (total_clicks / total_impr * 100) if total_impr > 0 else 0.0
    overall_cpc = (total_cost / total_clicks) if total_clicks > 0 else 0.0
    overall_cvr = (total_conv / total_clicks * 100) if total_clicks > 0 else 0.0
    overall_cpa = (total_cost / total_conv) if total_conv > 0 else 0.0
    overall_roas = (total_conv_val / total_cost) if total_cost > 0 else 0.0
    overall_profit = total_conv_val - total_cost
    
    return {
        "total_terms": len(df),
        "total_impressions": total_impr,
        "total_clicks": total_clicks,
        "total_cost": total_cost,
        "total_conversions": total_conv,
        "total_conversion_value": total_conv_val,
        "overall_ctr": overall_ctr,
        "overall_cpc": overall_cpc,
        "overall_cvr": overall_cvr,
        "overall_cpa": overall_cpa,
        "overall_roas": overall_roas,
        "overall_profit": overall_profit
    }
