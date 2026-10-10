# PS-01 — Search Term Triage Engine · Dataset

**File:** `search_terms_report.csv` — 25,000 rows, one row per search term + match type.

The advertiser is a fictional US small-business lender, *Kestrel Business Funding*. It sells working
capital loans, equipment finance, business lines of credit, merchant cash advances and SBA loans.
It does **not** sell personal loans, mortgages, car loans, student loans or grants, and it is not hiring.
This is a 90-day report. All data is synthetic.

| Column | Meaning |
|---|---|
| row_id | Unique row number |
| search_term | What the person actually typed into Google before the ad showed |
| match_type | How the keyword matched: broad / phrase / exact |
| campaign | Campaign the ad ran in |
| ad_group | Ad group within that campaign |
| impressions | Times the ad was shown for this search term |
| clicks | Clicks on the ad |
| cost | Money spent on those clicks (USD) |
| conversions | Loan applications submitted after the click |
| conversion_value | Estimated value of those applications (USD) |

**Your job:** for every row, decide **promote** (add as a keyword we bid on deliberately),
**block** (add as a negative keyword so we stop paying for it) or **ignore**, with a one-line reason
and a confidence level. Abstain / route to human review when the data is too thin.

**Watch out for:** most rows have very few clicks. A term with 2 clicks and 1 conversion is not a
proven winner. Spelling variants of the same search appear as separate rows.
