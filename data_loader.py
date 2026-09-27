"""
Phase 5: Data Collection & Data Understanding
------------------------------------------------
DATA SOURCE
-----------
This project reuses the UCI "Online Retail" dataset already used for
Day 10's demand-forecasting project — the same 541,909 raw invoice line
items from a UK-based online gift retailer, 01/12/2010 to 09/12/2011:
    https://archive.ics.uci.edu/dataset/352/online+retail

WHY REUSE THE SAME DATASET (a deliberate choice, not a shortcut)
-------------------------------------------------------------------
A real data team doesn't get a fresh dataset for every new analysis —
the same transaction warehouse routinely supports many different
questions over time (a demand forecast one week, a cross-sell analysis
the next). This project asks a genuinely different question of the same
raw data: not "how much will we sell," but "which products get bought
TOGETHER" — market basket / association-rule analysis, a fundamentally
different technique (unsupervised pattern mining, not a trained
predictive model) from every other project in this series so far.

If you're following along fresh: download the same file used in Day 10
(`data/online_retail.csv`) — the schema is identical.
"""

import pandas as pd


def load_raw_data(path: str = "data/online_retail.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%y %H:%M")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} raw invoice line items")
    print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    print(f"\nCountry share (top 5):")
    print(df["Country"].value_counts(normalize=True).head(5).round(4))
