"""
Part of Phase 7: train/test split
------------------------------------------------
WHY A TIME-BASED SPLIT FOR ASSOCIATION RULES (a step most market-basket
tutorials skip entirely): the standard tutorial approach is to mine
rules from ALL the data and report them as findings, with no check that
the discovered patterns actually hold up on baskets the algorithm never
saw. That's equivalent to reporting training accuracy with no test set
-- a rule with high support and confidence on the data it was mined from
could still be a fluke of that specific period. Splitting baskets
chronologically (mine rules on the earlier period, validate confidence
and lift against the LATER period) gives an honest read on whether a
rule reflects a durable buying pattern or just noise in one time window
-- the same train/test discipline used for every predictive model in
this series, applied here to an unsupervised technique that doesn't
usually get it.
"""

import pandas as pd


def split_baskets(basket: pd.DataFrame, invoice_dates: pd.Series, test_frac: float = 0.25):
    invoice_dates = invoice_dates.sort_values()
    cutoff_idx = int(len(invoice_dates) * (1 - test_frac))
    cutoff_date = invoice_dates.iloc[cutoff_idx]

    train_invoices = invoice_dates[invoice_dates <= cutoff_date].index
    test_invoices = invoice_dates[invoice_dates > cutoff_date].index

    basket_train = basket.loc[basket.index.intersection(train_invoices)]
    basket_test = basket.loc[basket.index.intersection(test_invoices)]

    return basket_train, basket_test, cutoff_date
