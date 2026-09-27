"""
Phase 6: Exploratory Data Analysis
------------------------------------
WHY THESE SPECIFIC CUTS: before running any association-rule algorithm,
a merchandising stakeholder needs to know whether the transaction data
can even support meaningful "bought together" patterns -- which requires
checking basket composition (how many DISTINCT products per order, not
quantity), whether mixing countries would muddy the analysis, and which
products have enough transaction volume to produce statistically
meaningful rules.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_raw_data


def run_eda(df: pd.DataFrame, out_dir: str = "outputs"):
    df = df.copy()
    is_cancel = df["InvoiceNo"].astype(str).str.startswith("C")
    sold = df[(~is_cancel) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0) & (df["Description"].notna())]

    print("=== Country concentration ===")
    print(df["Country"].value_counts(normalize=True).head(5).round(4))
    # WHY: market basket patterns are cultural/catalog-specific -- mixing
    # a UK gift-shop customer's basket with a wholesale German buyer's
    # basket would blend two different purchasing behaviors into
    # meaningless "rules." 91.4% of transactions are UK, which is both
    # the natural default scope AND large enough on its own to analyze.

    uk_sold = sold[sold["Country"] == "United Kingdom"]
    print(f"\nUK sold rows: {len(uk_sold)} ({len(uk_sold) / len(sold):.1%} of all sold rows)")

    basket_sizes = uk_sold.groupby("InvoiceNo")["StockCode"].nunique()
    print("\n=== Basket size (DISTINCT products per invoice), UK only ===")
    print(basket_sizes.describe().round(1))
    # WHY DISTINCT PRODUCTS, NOT TOTAL QUANTITY: association rules ask
    # "which products co-occur in an order," not "how many units" -- a
    # basket of 50 units of one product is a basket of size 1 for this
    # purpose, not size 50.

    single_item = (basket_sizes == 1).sum()
    print(f"\nSingle-product baskets: {single_item} of {len(basket_sizes)} ({single_item / len(basket_sizes):.1%})")
    print("-> These can't contribute to a 'bought together' rule by definition and are excluded.")

    product_freq = uk_sold.groupby("StockCode")["InvoiceNo"].nunique().sort_values(ascending=False)
    print(f"\n=== Product frequency (number of distinct baskets containing each product) ===")
    print(f"Unique products: {len(product_freq)}")
    print(f"Products in >=100 baskets: {(product_freq >= 100).sum()}")
    print(f"Products in >=50 baskets: {(product_freq >= 50).sum()}")
    print(f"\nTop 10 most frequently purchased products:")
    print(product_freq.head(10))
    # WHY THIS MATTERS FOR SCOPING: association-rule mining over all
    # ~3,900 UK products would produce an enormous, mostly-noise rule set
    # (most products appear too rarely to support statistically
    # meaningful co-occurrence patterns) and would be computationally
    # slow. Restricting to the top N by basket frequency concentrates
    # the analysis on products with enough volume to trust the resulting
    # rules -- the same "focus where it matters" scoping judgment used
    # throughout this series (e.g. Day 10's top-20-SKU forecasting scope).

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    basket_sizes.clip(upper=80).plot(kind="hist", bins=40, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Distinct products per basket (clipped at 80 for readability)")
    axes[0].set_xlabel("Distinct products")

    product_freq.head(15).plot(kind="barh", ax=axes[1], color="#55A868", title="Top 15 products by basket frequency")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_summary.png", dpi=120)
    print(f"\nSaved chart -> {out_dir}/eda_summary.png")


if __name__ == "__main__":
    data = load_raw_data()
    run_eda(data)
