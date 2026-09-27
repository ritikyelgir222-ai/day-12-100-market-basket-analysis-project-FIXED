"""
Phase 8: "Model Development" & Phase 9: Evaluation & Business Validation
---------------------------------------------------------------------------
There's no trained predictive model in the usual sense here -- Apriori
is an unsupervised pattern-mining algorithm, not something that learns
parameters from a target. "Development" in this project means choosing
defensible support/confidence/lift thresholds; "evaluation" means
checking the discovered rules actually hold up on baskets the algorithm
never saw, which most market-basket tutorials skip.
"""

import json

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from data_loader import load_raw_data
from clean_and_engineer import clean_data, build_basket_matrix
from split import split_baskets

MIN_SUPPORT = 0.02       # a rule/itemset must appear in at least 2% of baskets to be considered
MIN_CONFIDENCE = 0.30    # P(consequent | antecedent) must be at least 30%
MIN_LIFT = 1.5           # co-occurrence must be at least 1.5x more likely than chance


def compute_rules(basket: pd.DataFrame, min_support: float = MIN_SUPPORT) -> pd.DataFrame:
    """
    WHY MIN_SUPPORT = 0.02 (not a much lower or higher number): with
    ~150 products and baskets averaging only a handful of items each,
    most 2-item combinations occur in well under 1% of baskets by
    chance alone. Requiring at least 2% support keeps the frequent-
    itemset search from exploding combinatorially (Apriori's runtime is
    highly sensitive to this threshold) while still catching the
    genuinely common pairings a merchandising team could act on -- a
    rule true for 1 basket in 500 isn't a pattern worth building a
    cross-sell campaign around anyway.
    """
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    if len(frequent_itemsets) == 0:
        return pd.DataFrame()
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules = rules[rules["lift"] >= MIN_LIFT]
    # WHY FILTER ON LIFT TOO, NOT JUST CONFIDENCE: a rule can have high
    # confidence simply because the consequent is a universally popular
    # product everyone buys anyway, regardless of the antecedent (e.g.
    # "if you buy anything, you probably also bought the best-seller").
    # Lift corrects for that baseline popularity -- lift > 1 means the
    # pairing happens MORE than you'd expect from each product's
    # individual popularity alone, which is the actual "these go
    # together" signal a cross-sell recommendation needs.
    return rules.sort_values("lift", ascending=False)


def validate_rules_out_of_sample(rules: pd.DataFrame, basket_test: pd.DataFrame) -> pd.DataFrame:
    """
    THE HONEST EVALUATION STEP: for every rule mined from the training
    baskets, recompute its actual support/confidence/lift on the TEST
    baskets (never used for mining) and report whether it still holds.
    """
    results = []
    n_test = len(basket_test)
    for _, rule in rules.iterrows():
        antecedent = list(rule["antecedents"])
        consequent = list(rule["consequents"])
        cols = antecedent + consequent
        if not all(c in basket_test.columns for c in cols):
            continue

        has_antecedent = basket_test[antecedent].all(axis=1)
        has_both = basket_test[cols].all(axis=1)

        n_antecedent = int(has_antecedent.sum())
        test_support = float(has_both.sum() / n_test) if n_test > 0 else 0.0
        test_confidence = float(has_both.sum() / n_antecedent) if n_antecedent > 0 else 0.0
        consequent_support_test = float(basket_test[consequent].all(axis=1).sum() / n_test) if n_test > 0 else 0.0
        test_lift = float(test_confidence / consequent_support_test) if consequent_support_test > 0 else 0.0

        results.append({
            "antecedent": ", ".join(antecedent),
            "consequent": ", ".join(consequent),
            "train_support": round(rule["support"], 4),
            "train_confidence": round(rule["confidence"], 4),
            "train_lift": round(rule["lift"], 4),
            "test_n_baskets_with_antecedent": n_antecedent,
            "test_support": round(test_support, 4),
            "test_confidence": round(test_confidence, 4),
            "test_lift": round(test_lift, 4),
            "held_up": bool(test_lift >= 1.2),  # a slightly looser bar than the MIN_LIFT used for mining -- see WHY below
        })
    return pd.DataFrame(results)


def train_and_evaluate():
    raw = load_raw_data()
    cleaned = clean_data(raw)
    basket, product_names, invoice_dates = build_basket_matrix(cleaned)
    basket_train, basket_test, cutoff_date = split_baskets(basket, invoice_dates)

    print(f"Train baskets: {len(basket_train)} (through {cutoff_date.date()})")
    print(f"Test baskets:  {len(basket_test)} (after {cutoff_date.date()})")

    rules = compute_rules(basket_train)
    print(f"\nRules mined from training baskets (support >= {MIN_SUPPORT}, confidence >= {MIN_CONFIDENCE}, lift >= {MIN_LIFT}): {len(rules)}")

    validated = validate_rules_out_of_sample(rules, basket_test)

    # -----------------------------------------------------------------
    # Business validation: WHY test_lift >= 1.2 (not the same 1.5 used
    # for MINING) counts as "held up": mining thresholds are
    # deliberately strict to avoid drowning in noisy candidate rules
    # from ~150*149 possible pairs. The out-of-sample bar for "is this
    # still a real, useful signal" is intentionally a bit more lenient
    # (still meaningfully above the lift=1.0 "no better than chance"
    # line) -- a rule doesn't need to be EXACTLY as strong on a smaller
    # test sample to still be a rule worth acting on; it needs to still
    # show a genuine positive association.
    # -----------------------------------------------------------------
    held_up_rate = float(validated["held_up"].mean()) if len(validated) > 0 else 0.0
    avg_train_lift = float(validated["train_lift"].mean()) if len(validated) > 0 else 0.0
    avg_test_lift = float(validated["test_lift"].mean()) if len(validated) > 0 else 0.0

    business_summary = {
        "n_rules_mined": len(rules),
        "n_rules_validated": len(validated),
        "pct_rules_held_up_out_of_sample": round(held_up_rate, 4),
        "avg_train_lift": round(avg_train_lift, 4),
        "avg_test_lift": round(avg_test_lift, 4),
        "min_support": MIN_SUPPORT,
        "min_confidence": MIN_CONFIDENCE,
        "min_lift_for_mining": MIN_LIFT,
        "note": (
            "pct_rules_held_up_out_of_sample is the key trust metric for this "
            "project -- it answers 'of the cross-sell rules we'd hand to "
            "merchandising, what share are genuine patterns rather than noise "
            "specific to the training period?' This is a real, measured "
            "comparison on held-out baskets, not an assumption."
        ),
    }

    print("\n=== Business Validation Summary (Phase 9) ===")
    for k, v in business_summary.items():
        print(f"{k}: {v}")

    with open("outputs/business_validation.json", "w") as f:
        json.dump(business_summary, f, indent=2)

    # Add readable product names for the top rules and save
    def name_of(code):
        return product_names.get(code, code)

    validated["antecedent_name"] = validated["antecedent"].apply(
        lambda codes: ", ".join(name_of(c) for c in codes.split(", "))
    )
    validated["consequent_name"] = validated["consequent"].apply(
        lambda codes: ", ".join(name_of(c) for c in codes.split(", "))
    )
    validated = validated.sort_values("test_lift", ascending=False)
    validated.to_csv("outputs/validated_rules.csv", index=False)

    print("\n=== Top 10 rules by TEST lift (held-out validated) ===")
    print(validated.head(10)[["antecedent_name", "consequent_name", "train_lift", "test_lift", "held_up"]].to_string(index=False))

    # Save only the rules that genuinely held up out-of-sample as the
    # deployable recommendation table -- this is what app.py serves.
    deployable = validated[validated["held_up"]].copy()
    deployable.to_csv("outputs/deployable_rules.csv", index=False)
    print(f"\nSaved {len(deployable)} out-of-sample-validated rules -> outputs/deployable_rules.csv")
    print("Saved all validated rules -> outputs/validated_rules.csv")


if __name__ == "__main__":
    train_and_evaluate()
