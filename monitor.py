"""
Phase 13: Monitoring & Maintenance
--------------------------------------
WHAT "MONITORING" MEANS FOR AN ASSOCIATION-RULE SYSTEM (different from
every predictive-model monitor.py in this series): there's no accuracy
metric to track drift on -- instead, the question is whether the
deployed rule set still reflects current buying behavior, or whether a
product's association pattern has shifted (a seasonal item falls out of
a bundle, a product gets discontinued, a new bundle emerges) enough that
the rule table needs re-mining.

WHY THIS SCRIPT REUSES THE SAME TEST-SET VALIDATION LOGIC as
train_model.py, rather than a genuinely new time period: like the
cross-sectional projects in this series, this dataset has a fixed end
date (Dec 2011) with no further data beyond it -- there is no real
"next quarter" to check against. Re-running the validation confirms the
CODE works and shows what a periodic re-validation cycle should look
like in production, not that real rule decay is being caught here.
"""

import json

import pandas as pd

from data_loader import load_raw_data
from clean_and_engineer import clean_data, build_basket_matrix
from split import split_baskets
from train_model import compute_rules, validate_rules_out_of_sample

RULE_DECAY_LIFT_THRESHOLD = 1.2  # a rule whose out-of-sample lift falls below this is no longer worth serving


def run_monitoring_check():
    raw = load_raw_data()
    cleaned = clean_data(raw)
    basket, product_names, invoice_dates = build_basket_matrix(cleaned)
    basket_train, basket_test, cutoff_date = split_baskets(basket, invoice_dates)

    rules = compute_rules(basket_train)
    validated = validate_rules_out_of_sample(rules, basket_test)

    with open("outputs/business_validation.json") as f:
        baseline = json.load(f)

    current_held_up_rate = float(validated["held_up"].mean()) if len(validated) > 0 else 0.0
    baseline_held_up_rate = baseline["pct_rules_held_up_out_of_sample"]

    print(f"=== Rule health check ===")
    print(f"Baseline (last training run) held-up rate: {baseline_held_up_rate:.1%}")
    print(f"Current run held-up rate:                   {current_held_up_rate:.1%}")

    n_decayed = int((validated["test_lift"] < RULE_DECAY_LIFT_THRESHOLD).sum())
    print(f"\nRules currently below the lift={RULE_DECAY_LIFT_THRESHOLD} decay threshold: {n_decayed} of {len(validated)}")

    drop = baseline_held_up_rate - current_held_up_rate
    if drop > 0.10:
        print(f"\n⚠️  RE-MINING RECOMMENDED: held-up rate dropped {drop:.1%} vs. baseline")
    else:
        print(f"\n✅ No re-mining needed (held-up rate change: {-drop:+.1%})")

    print("\nNOTE: in production, this check should run monthly against a")
    print("genuinely new period of baskets (not the same held-out test set")
    print("used at training time) -- this dataset has no further data beyond")
    print("Dec 2011 to demonstrate that with.")


if __name__ == "__main__":
    run_monitoring_check()
