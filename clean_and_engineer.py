"""
Phases 5 (data quality fixes) & 7 (feature engineering)
------------------------------------------------------------
"Feature engineering" here means building the BASKET MATRIX association
rules are mined from -- a one-hot table of InvoiceNo x Product, each
cell True/False for "was this product in this basket." Every scoping
decision below has a one-line WHY comment.
"""

import pandas as pd

TOP_N_PRODUCTS = 150
MIN_BASKET_SIZE = 2

# StockCodes that are administrative/fee line items, not real products --
# verified individually against their Description (see eda.py's output
# and PROJECT_DOCUMENTATION.md Section 4 for how these were caught: the
# first version of this analysis ranked "DOTCOM POSTAGE" as the single
# strongest cross-sell partner for several genuine products, which is
# meaningless -- postage appears on almost every shipped multi-item
# order regardless of which products are in it, so it trivially
# co-occurs with everything and inflates lift without reflecting any
# real merchandising insight.
NON_PRODUCT_CODES = ["POST", "DOT", "C2", "M", "BANK CHARGES", "AMAZONFEE", "CRUK", "PADS", "D", "S"]


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # -----------------------------------------------------------------
    # Fix 1 & 2: same cancellation and stock-adjustment exclusions used
    # in Day 10, for the same reasons (verified there: cancellations are
    # always full-order reversals; the zero-price/no-description rows
    # are manual adjustments, not customer purchases).
    # -----------------------------------------------------------------
    is_cancellation = df["InvoiceNo"].astype(str).str.startswith("C")
    df = df[~is_cancellation]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[df["Description"].notna()]

    # -----------------------------------------------------------------
    # Fix 3: restrict to United Kingdom transactions (91.4% of the data).
    # WHY: market basket patterns reflect a specific customer base's
    # purchasing culture and catalog access. Mixing UK retail-gift-shop
    # customers with, say, wholesale buyers in other countries would
    # blend genuinely different purchasing behaviors into rules that
    # don't cleanly describe either group. UK alone is both the natural
    # default (it's the company's home market and the large majority of
    # transactions) and large enough on its own (485,123 rows) to
    # analyze.
    # -----------------------------------------------------------------
    df = df[df["Country"] == "United Kingdom"]

    # -----------------------------------------------------------------
    # Fix 4: exclude administrative/fee line items (see NON_PRODUCT_CODES
    # above). WHY THIS MATTERS ENOUGH TO BE ITS OWN FIX: a first pass of
    # this analysis (before this fix was added) ranked "DOTCOM POSTAGE"
    # as the single strongest cross-sell partner for several genuine
    # products, with lift scores above 20. That's not a real
    # merchandising insight -- postage is added to nearly every shipped
    # multi-item order regardless of contents, so it trivially co-occurs
    # with everything. Left in, it would have crowded out every genuine
    # product-to-product pattern from the top of the rule list.
    # -----------------------------------------------------------------
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]

    # NOTE: unlike Day 10, no quantity cap is applied here -- market
    # basket analysis uses binary PRESENCE (was this product in the
    # basket at all), not the quantity ordered, so an outlier bulk-order
    # quantity doesn't distort a presence/absence matrix the way it
    # would distort a demand-forecasting target.

    return df


def build_basket_matrix(clean_df: pd.DataFrame, top_n: int = TOP_N_PRODUCTS, min_basket_size: int = MIN_BASKET_SIZE):
    """
    Builds a one-hot basket matrix: rows = invoices, columns = the top N
    products by basket frequency, values = True/False for "was this
    product in this basket."
    """
    df = clean_df.copy()

    # WHY TOP N PRODUCTS, NOT ALL ~3,900: EDA found the large majority
    # of UK products appear in too few baskets to support statistically
    # meaningful co-occurrence rules, and Apriori's computational cost
    # grows quickly with the number of items considered. Restricting to
    # the top N by basket frequency concentrates the analysis where
    # there's enough volume to trust the resulting rules.
    product_freq = df.groupby("StockCode")["InvoiceNo"].nunique().sort_values(ascending=False)
    top_products = product_freq.head(top_n).index
    df = df[df["StockCode"].isin(top_products)]

    # A lookup from StockCode to a representative product name, for
    # human-readable output later (a single StockCode occasionally has
    # more than one slightly different Description across rows -- the
    # most common one is used).
    product_names = df.groupby("StockCode")["Description"].agg(lambda s: s.mode().iat[0])

    basket = (
        df.groupby(["InvoiceNo", "StockCode"])["Quantity"].sum().unstack(fill_value=0)
    )
    basket = basket > 0  # binary presence, not quantity

    # WHY EXCLUDE SINGLE-PRODUCT BASKETS: EDA found 8.3% of UK baskets
    # contain only 1 distinct product -- these can't contribute to a
    # "bought together" pattern by definition, and keeping them just
    # dilutes support calculations with rows that are structurally
    # incapable of confirming any 2-item rule.
    basket = basket[basket.sum(axis=1) >= min_basket_size]

    invoice_dates = df.groupby("InvoiceNo")["InvoiceDate"].min()
    invoice_dates = invoice_dates.loc[basket.index]

    return basket, product_names, invoice_dates


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw = load_raw_data()
    cleaned = clean_data(raw)
    print(f"UK, non-cancelled, real-sale rows: {len(cleaned)}")

    basket, product_names, invoice_dates = build_basket_matrix(cleaned)
    print(f"Basket matrix: {basket.shape[0]} baskets x {basket.shape[1]} products")
    print(f"Matrix density (avg products per basket / {basket.shape[1]}): {basket.mean(axis=1).mean():.3f}")
    print(f"Date range of baskets: {invoice_dates.min()} to {invoice_dates.max()}")

    basket.to_csv("outputs/basket_matrix.csv")
    product_names.to_csv("outputs/product_names.csv", header=["Description"])
    print("\nSaved -> outputs/basket_matrix.csv, outputs/product_names.csv")
