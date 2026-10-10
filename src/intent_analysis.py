"""
SearchSense AI - Intent Analysis & Query Grouping Engine
-------------------------------------------------------
This module classifies search queries into explainable intent categories using 
deterministic pattern matching and lexical token recognition tailored to commercial lending.
Includes high-performance near-duplicate query grouping.

Notice: This module utilizes transparent, rule-based heuristic intelligence.
It does NOT claim to be an opaque or unvalidated deep learning model.
"""

import re
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict
import pandas as pd


# Dictionary of intent categories with associated token patterns and explanations
INTENT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "Negative: Jobs & Careers": {
        "keywords": [
            "job", "jobs", "salary", "salaries", "career", "careers", "intern", 
            "internship", "internships", "hiring", "resume", "cv", "interview question", 
            "interview questions", "underwriter salary", "processor job", "loan officer jobs"
        ],
        "weight": 1.0,
        "is_negative": True,
        "description": "Job and career seekers looking for employment rather than financing."
    },
    "Negative: Grants & Free Money": {
        "keywords": [
            "grant", "grants", "free", "free business grants", "government grants", 
            "government grant", "covid relief", "covid relief grant", "stimulus", 
            "ein only free", "free grants"
        ],
        "weight": 1.0,
        "is_negative": True,
        "description": "Users seeking non-repayable grants or free money, not commercial loans."
    },
    "Negative: Unsupported Products": {
        "keywords": [
            "mortgage", "mortgages", "personal loan", "personal debt", 
            "debt consolidation personal", "credit card offers", "credit card", 
            "credit cards", "used car", "car loan", "auto loan", "used car financing", 
            "bank account", "checking account", "savings account", "accounting software", 
            "payroll software", "invoice template", "business plan template", 
            "template", "templates", "rental", "equipment rental", "repair"
        ],
        "weight": 1.0,
        "is_negative": True,
        "description": "Searches for consumer loans, mortgages, templates, software, or products not offered."
    },
    "Negative: Piracy & Free Software": {
        "keywords": [
            "crack", "torrent", "nulled", "keygen", "warez", "github", "download crack", "pdf download"
        ],
        "weight": 1.0,
        "is_negative": True,
        "description": "Pirated software or file downloads."
    },
    "High-Intent Commercial Lending": {
        "keywords": [
            "apply", "approval", "instant", "fast", "rates", "quote", "online", 
            "business loan", "sba loan", "sba 7a", "line of credit", "revolving credit", 
            "working capital", "equipment finance", "equipment financing", "equipment lease", 
            "equipment leasing", "truck financing", "commercial vehicle loan", 
            "merchant cash advance", "mca", "revenue based financing", "startup funding", 
            "small business funding", "bad credit", "cash advance", "short term business loan"
        ],
        "weight": 0.9,
        "is_negative": False,
        "description": "High purchase readiness looking for commercial financing products."
    },
    "Commercial Investigation": {
        "keywords": [
            "best", "top", "review", "reviews", "vs", "versus", "compare", 
            "comparison", "calculator", "requirements", "lenders", "companies", 
            "solutions", "alternatives", "limits", "checklist"
        ],
        "weight": 0.75,
        "is_negative": False,
        "description": "Prospects evaluating options, comparing lenders, or calculating loan terms."
    },
    "Informational / Educational": {
        "keywords": [
            "how to", "how do", "what is", "why is", "guide", "tutorial", 
            "definition", "meaning", "explained", "tips", "requirements 2026"
        ],
        "weight": 0.6,
        "is_negative": False,
        "description": "Educational or exploratory searches with informational intent."
    }
}

# Compile regexes for ultra-fast vectorization across 25,000+ rows
COMPILED_PATTERNS = []
for category, data in INTENT_DEFINITIONS.items():
    # Sort keywords by length descending so multi-word phrases match first
    sorted_kws = sorted(data["keywords"], key=len, reverse=True)
    pattern_str = r"\b(" + "|".join(re.escape(k) for k in sorted_kws) + r")\b"
    regex = re.compile(pattern_str, flags=re.IGNORECASE)
    COMPILED_PATTERNS.append((category, regex, data["is_negative"], data["description"]))


def analyze_query_intent_fast(query: str) -> Tuple[str, str, str, bool]:
    """
    Analyzes an individual search query string using compiled regex patterns.
    
    Returns:
    --------
    Tuple[str, str, str, bool]:
        - primary_intent: detected intent category
        - matched_tokens: comma-separated list of matched tokens
        - explanation: human-readable explanation
        - is_negative: boolean flag indicating negative/irrelevant intent
    """
    if not query or not isinstance(query, str):
        return ("General Exploratory", "None", "Query is empty or invalid.", False)
        
    matched_negative_category = None
    matched_negative_tokens = []
    
    matched_positive_category = None
    matched_positive_tokens = []
    
    for category, regex, is_neg, desc in COMPILED_PATTERNS:
        matches = regex.findall(query)
        if matches:
            clean_matches = sorted(list(set(m.lower() for m in matches)))
            if is_neg and not matched_negative_category:
                matched_negative_category = category
                matched_negative_tokens = clean_matches
                break  # Negative intent has immediate priority to stop budget bleed
            elif not is_neg and not matched_positive_category:
                matched_positive_category = category
                matched_positive_tokens = clean_matches

    # If any negative category was matched, assign it immediately
    if matched_negative_category:
        tokens_str = ", ".join(matched_negative_tokens)
        explanation = f"Matched negative trigger [{tokens_str}]. {INTENT_DEFINITIONS[matched_negative_category]['description']}"
        return (matched_negative_category, tokens_str, explanation, True)
        
    # Otherwise assign positive commercial/informational intent
    if matched_positive_category:
        tokens_str = ", ".join(matched_positive_tokens)
        explanation = f"Matched commercial signal [{tokens_str}]. {INTENT_DEFINITIONS[matched_positive_category]['description']}"
        return (matched_positive_category, tokens_str, explanation, False)
        
    return (
        "General Exploratory",
        "None",
        "No specific commercial or negative trigger tokens matched. Standard exploratory search query.",
        False
    )


def enrich_dataframe_with_intent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies high-speed query intent analysis to each search term in the DataFrame.
    """
    if df is None or df.empty:
        return df
        
    df_enriched = df.copy()
    
    # Process queries using fast list comprehension
    queries = df_enriched["query"].tolist()
    intents = []
    tokens = []
    explanations = []
    is_negatives = []
    
    for q in queries:
        cat, tok, exp, is_neg = analyze_query_intent_fast(q)
        intents.append(cat)
        tokens.append(tok)
        explanations.append(exp)
        is_negatives.append(is_neg)
        
    df_enriched["intent"] = intents
    df_enriched["intent_tokens"] = tokens
    df_enriched["intent_explanation"] = explanations
    df_enriched["is_negative_intent"] = is_negatives
    
    return df_enriched


def normalize_query_for_grouping(query: str) -> str:
    """
    Normalizes a query by lowercasing, stripping punctuation, fixing known typos,
    and sorting unique tokens alphabetically to detect semantic word-order near-duplicates.
    """
    if not query:
        return ""
    q = query.lower().strip()
    # Replace common spelling variants / typos in business lending PPC
    replacements = {
        "buiness": "business",
        "ubsiness": "business",
        "financingg": "financing",
        "advvance": "advance",
        "lineo f credit": "line of credit",
        "lie of credit": "line of credit",
        "equipement": "equipment",
    }
    for typo, fix in replacements.items():
        q = q.replace(typo, fix)
        
    tokens = re.findall(r"\b[a-z0-9]+\b", q)
    # Return sorted tokens signature
    return " ".join(sorted(tokens))


def detect_near_duplicates(df: pd.DataFrame, min_cluster_size: int = 2) -> pd.DataFrame:
    """
    Groups queries with near-duplicate phrasing, word order variations, or common misspellings.
    Returns an aggregated summary of duplicate clusters with total spend and conversions.
    """
    if df is None or df.empty:
        return pd.DataFrame()
        
    df_work = df.copy()
    df_work["token_signature"] = df_work["query"].apply(normalize_query_for_grouping)
    
    # Group by token_signature and campaign
    grouped = df_work.groupby(["campaign", "token_signature"]).agg(
        variant_count=("query", "nunique"),
        sample_queries=("query", lambda s: list(s.unique())[:5]),
        total_impressions=("impressions", "sum"),
        total_clicks=("clicks", "sum"),
        total_cost=("cost", "sum"),
        total_conversions=("conversions", "sum"),
        total_value=("conversion_value", "sum")
    ).reset_index()
    
    # Filter for clusters with 2 or more distinct queries
    clusters = grouped[grouped["variant_count"] >= min_cluster_size].copy()
    clusters = clusters.sort_values(by="total_cost", ascending=False)
    
    return clusters
