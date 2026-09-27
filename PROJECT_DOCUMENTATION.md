# Project Documentation: Market Basket Analysis / Cross-Sell Recommendations
### Technical Documentation & Handover — Phase 12 of the SDLC

This document is the technical documentation and user manual a real handover
package would include. Every number below came from actually running the
code in this repository — none are illustrative.

---

## 1. Project Summary

| | |
|---|---|
| **Objective** | Identify genuine "bought together" product pairs to power cross-sell recommendations |
| **Client context** | E-commerce (online gift retailer) merchandising/marketing team |
| **Data source** | UCI Online Retail — reused from Day 10, 541,909 raw invoice line items |
| **Technique** | Apriori association-rule mining (unsupervised — no trained predictive model) |
| **Scope** | UK transactions, top 150 products by basket frequency, 12,851 multi-item baskets |
| **Result** | 708 rules mined, **100%** validated out-of-sample on held-out baskets |
| **Business framing** | Serve only rules confirmed to hold on baskets never used for mining |

---

## 2. Architecture

```
data/online_retail.csv (same raw file as Day 10)
        │
        ▼
data_loader.py ──────► loads raw CSV, parses dates
        │
        ▼
clean_and_engineer.py ─► excludes cancellations, stock adjustments,
        │                non-UK transactions, and ADMINISTRATIVE
        │                CODES (postage, carriage, fees -- see
        │                Section 4); builds the basket matrix
        ▼
   split.py ───────────► chronological basket split (train / test)
        │
        ▼
train_model.py ───────► Apriori mining on train baskets -> rules;
        │                VALIDATES each rule's confidence/lift on
        │                test baskets never used for mining
        ▼
outputs/deployable_rules.csv (only out-of-sample-validated rules)
        │
        ├──────────────► app.py ─── FastAPI /recommend endpoint,
        │                           serves only validated rules
        │
        └──────────────► monitor.py ─ re-validates rule health
```

---

## 3. Data Preparation & Scoping Decisions

| Decision | Verified reasoning |
|---|---|
| UK transactions only | 91.4% of all rows; mixing countries would blend different purchasing cultures/catalogs into meaningless rules |
| Top 150 products by basket frequency | Of ~3,900 UK products, most appear too rarely to support statistically meaningful rules; Apriori's cost also grows quickly with item count |
| Exclude single-product baskets | 8.3% of UK baskets contain only 1 distinct product — structurally incapable of confirming a 2-item rule |
| **Exclude administrative StockCodes** | `POST`, `DOT`, `C2`, `M`, `BANK CHARGES`, `AMAZONFEE`, `CRUK`, `PADS`, `D`, `S` — verified individually against their `Description` field; not real products (see Section 4) |
| Binary presence, not quantity | Association rules ask "did these co-occur," not "how many units" — no quantity cap needed here (unlike Day 10's demand-forecasting target) |

---

## 4. The Central Finding: Catching a Meaningless "Rule"

The first pass of this analysis (before the administrative-code fix) ran
Apriori directly on the raw basket matrix. The result: **"DOTCOM
POSTAGE" ranked as the single strongest cross-sell partner** for
multiple genuine products, with lift scores above 20 — a number that
looks impressive and is completely useless to a merchandising team.

**Why this happened:** postage/shipping is logged as its own line item
(`StockCode = "DOT"`) on nearly every shipped multi-item order,
regardless of which actual products are in it. It doesn't reflect any
customer preference — it's a byproduct of how the transaction system
records shipping charges. A rule like "customers who buy Product X also
buy DOTCOM POSTAGE" is not a merchandising insight; it's an artifact of
the data schema.

**How it was caught:** reviewing the top-ranked rules by lift before
trusting them, rather than taking the algorithm's output at face value.
Cross-referencing the `StockCode` against its `Description` (`DOT` →
"DOTCOM POSTAGE", `POST` → "POSTAGE", `C2` → "CARRIAGE", `M` → "Manual",
etc.) confirmed these are administrative/fee entries, not products a
customer chose to buy.

**The fix:** exclude these codes before building the basket matrix (see
`NON_PRODUCT_CODES` in `clean_and_engineer.py`). After the fix, the top
rules by lift are genuinely sensible: matching pieces of the "Regency
teacup and saucer" collection (pink, green, roses colorways) and
different "jumbo storage bag" print variants — real product bundles a
merchandising team could act on immediately (e.g. "customers buying one
teacup color almost always complete the set").

This is the kind of finding worth leading a LinkedIn post with: **the
algorithm did exactly what it was asked, and the "insight" was still
wrong** until the output was sanity-checked against domain knowledge.

---

## 5. Modeling Results

### Mining parameters

| Parameter | Value | Reasoning |
|---|---|---|
| Min support | 2% | Keeps Apriori's search tractable; a pattern in <2% of baskets isn't worth a campaign |
| Min confidence | 30% | P(consequent \| antecedent) must be at least moderate |
| Min lift (mining) | 1.5x | Co-occurrence must be meaningfully above chance |

**708 rules** were mined from 9,639 training baskets (through
2011-10-11).

### Out-of-sample validation (the honest evaluation step)

Each of the 708 rules was re-evaluated on 3,212 test baskets (after
2011-10-11) — support, confidence, and lift recomputed from scratch on
data never used for mining.

| Metric | Train | Test |
|---|---|---|
| Average lift | 6.50 | **8.75** |
| Rules with test lift ≥ 1.2 ("held up") | — | **708 of 708 (100%)** |

**This 100% figure deserves scrutiny, not just celebration.** The most
plausible explanation: the mining thresholds (support ≥ 2%, confidence
≥ 30%, lift ≥ 1.5) are already fairly strict, and restricting to the top
150 most popular products means every candidate rule involves
well-established, high-volume products — exactly the conditions where a
pattern is more likely to be a durable signal than noise. This is a
genuinely positive result, but a smaller sample, lower mining
thresholds, or the full long-tail product catalog would likely show more
rules failing to hold up. Worth re-testing before assuming every future
mining run on this business's data will look this clean.

### Top validated rules (by test-set lift)

| Antecedent | Consequent | Train lift | Test lift |
|---|---|---|---|
| Pink + Roses Regency teacup/saucer | Green Regency teacup/saucer + Cakestand | 12.35 | **20.33** |
| Pink Regency teacup/saucer | Green + Roses teacup/saucer + Cakestand | 12.33 | 19.93 |
| Jumbo Bag Pink Polkadot + Jumbo Storage Bag Suki | Jumbo Bag Red Retrospot + Woodland Animals | 11.04 | 19.26 |

All of these are matched product-collection patterns — customers
completing a themed set — a genuinely actionable merchandising signal
("bundle these," "recommend the matching piece at checkout").

---

## 6. API Reference (Phase 10)

**Base URL (local):** `http://127.0.0.1:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Browsable product dropdown, shows live recommendations |
| `/docs` | GET | Swagger UI |
| `/health` | GET | Health check |
| `/recommend` | POST | Get cross-sell recommendations for one product |

**Response (verified, real query — StockCode 22699, "Roses Regency
Teacup and Saucer"):**
```json
{
  "queried_product": "ROSES REGENCY TEACUP AND SAUCER",
  "recommendations": [
    {"stock_code": "22697, 22698", "description": "GREEN REGENCY TEACUP AND SAUCER, PINK REGENCY TEACUP AND SAUCER", "confidence": 0.5663, "lift": 15.9548},
    {"stock_code": "22698", "description": "PINK REGENCY TEACUP AND SAUCER", "confidence": 0.6627, "lift": 14.8842}
  ]
}
```
The API only serves rules from `deployable_rules.csv` — the
out-of-sample-validated subset — never the raw mined output.

---

## 7. Monitoring & Maintenance Plan (Phase 13)

`monitor.py` re-runs the mining and validation pipeline and compares the
current held-up rate against the baseline stored at training time.
**Important limitation, stated directly in the script's output**: this
dataset has no data beyond December 2011, so there's no genuinely new
period to check rule decay against — this run confirms the monitoring
CODE works, not that real decay is being caught. In production, this
should run monthly against a fresh batch of baskets, since association
rules genuinely do decay: seasonal products fall in and out of relevant
bundles, items get discontinued, and new bundles emerge as the catalog
changes.

---

## 8. Known Limitations (stated for the handover record)

1. **The postage-artifact bug (Section 4) is the central limitation
   lesson of this project** — a reminder that unsupervised techniques
   have no target to sanity-check against, making manual review of
   top-ranked output essential before deployment, more so than for
   supervised models with a clear held-out accuracy number.
2. **Scoped to the top 150 UK products** — the long tail of ~3,750
   other products isn't analyzed.
3. **100% out-of-sample held-up rate is unusually clean** — see the
   honest discussion in Section 5.
4. **No genuine future period exists to validate rule decay** — see
   Section 7.
5. **Recommendations use only single-product antecedents at serving
   time** — real shopping-cart-level (multi-item) recommendations
   aren't implemented.
6. **Fixed: multi-item consequents were briefly served as fake
   "products."** An earlier version of `app.py` filtered rules only on
   antecedent (single product in, matches the query), but not on
   consequent — so a rule like `22698 → "22697, 22699, 22423"` was
   served as one `Recommendation` with a comma-joined, non-existent
   `stock_code`. Affected 81 of 439 single-antecedent rules (~18%).
   Fixed by also requiring a single-item consequent when serving a
   live recommendation — verified this doesn't remove any product's
   recommendations entirely (all 90 eligible products retain at least
   one clean, single-product recommendation after the fix).

---

## 9. File Map (for quick reference)

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Load raw invoice line items (reused from Day 10) |
| `eda.py` | 6 | Country concentration, basket sizes, product frequency |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Cleaning, UK scope, **postage-artifact fix**, basket matrix |
| `split.py` | 7 | Chronological basket split |
| `train_model.py` | 8-9 | Apriori mining + out-of-sample rule validation |
| `app.py` | 10 | FastAPI recommendation service |
| `monitor.py` | 13 | Rule-health check |
| `README.md` | 12 | Setup and run instructions, the postage bug up front |
| `PROJECT_DOCUMENTATION.md` (this file) | 12 | Technical documentation and handover |
