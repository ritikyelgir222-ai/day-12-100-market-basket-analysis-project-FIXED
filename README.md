# Market Basket Analysis / Cross-Sell Recommendations — Full Project Code

Companion code to Day 12 of the 100-day LinkedIn series. Reuses the real
UCI Online Retail transaction data from Day 10, but asks a fundamentally
different question of it: not "how much will we sell" but "which
products get bought together" — association-rule mining
(Apriori), the first **unsupervised** technique in this series.

## Data source

Same file as Day 10: **UCI Online Retail** — 541,909 raw UK e-commerce
invoice line items. See `data_loader.py` for why reusing the same
dataset for a different technique is a deliberate, realistic choice, not
a shortcut.

## The bug this project caught (worth reading before trusting any rule below)

The first version of this analysis ranked **"DOTCOM POSTAGE"** as the
single strongest cross-sell partner for several genuine products, with
lift scores above 20. That's not a real merchandising insight — postage
gets added as a line item to nearly every shipped multi-item order
regardless of what's actually in it, so it trivially co-occurs with
everything. Left in, it would have crowded out every genuine
product-to-product pattern from the top of the results. Fixed by
identifying and excluding all administrative/fee `StockCode`s (`POST`,
`DOT`, `C2` = carriage, `M` = manual adjustment, `BANK CHARGES`,
`AMAZONFEE`, `CRUK` = charity commission) — see `clean_and_engineer.py`.
After the fix, the top rules make genuine merchandising sense: matching
pieces of the "Regency teacup and saucer" collection (pink/green/roses)
and "jumbo storage bag" variants — real product bundles, not a shipping
artifact.

## What's genuinely different about this project

1. **No trained predictive model** — Apriori is unsupervised pattern
   mining, not something with a target to predict.
2. **A held-out validation step most market-basket tutorials skip
   entirely.** `train_model.py` mines rules from the earlier ~75% of
   baskets (chronologically) and checks whether each rule's
   confidence/lift still holds on baskets from the later period the
   algorithm never saw — the same train/test discipline as every
   predictive model in this series, applied to a technique that
   usually just reports rules from 100% of the data with no check at
   all.
3. **100% of the 708 mined rules held up out-of-sample.** This is
   unusually clean and is called out directly in
   `PROJECT_DOCUMENTATION.md` rather than presented uncritically — the
   likely explanation is that strict mining thresholds (support ≥ 2%,
   confidence ≥ 30%, lift ≥ 1.5) combined with restricting to the top
   150 most popular products already filters out marginal, noise-prone
   candidate rules before validation even runs.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data_loader.py         # Phase 5 — loads the raw invoice line items
python eda.py                  # Phase 6 — country concentration, basket sizes, product frequency + outputs/eda_summary.png
python clean_and_engineer.py   # Phase 5 (fixes) + 7 — cleaning, UK-only scope, non-product exclusion, basket matrix
python train_model.py          # Phase 8-9 — Apriori mining + out-of-sample rule validation
python monitor.py              # Phase 13 — rule-health check
```

## Serve the model (Phase 10)

```bash
uvicorn app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"stock_code": "22699", "top_n": 5}'
```

Expected: recommendations dominated by the matching Regency teacup/
saucer/cakestand pieces, each with confidence and lift figures.

## File map

| File | SDLC Phase | What it does |
|---|---|---|
| `data_loader.py` | 5 | Loads the raw invoice line items (reused from Day 10) |
| `eda.py` | 6 | Country concentration, basket size distribution, product frequency |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Cleaning, UK scope, **the postage-artifact fix**, basket-matrix construction |
| `split.py` | 7 | Chronological basket split for rule validation |
| `train_model.py` | 8-9 | Apriori mining + **out-of-sample rule validation** (most tutorials skip this) |
| `app.py` | 10 | FastAPI recommendation service, serves only validated rules |
| `monitor.py` | 13 | Rule-health check |

## Outputs produced (in `outputs/`)

- `basket_matrix.csv`, `product_names.csv`, `eda_summary.png`
- `business_validation.json` — the out-of-sample held-up rate and lift figures
- `validated_rules.csv` — every mined rule with train AND test metrics
- `deployable_rules.csv` — only the rules that held up out-of-sample (what `app.py` serves)

## Known limitations (stated honestly, not hidden)

- **Scoped to UK transactions and the top 150 products** — the same
  "focus where it matters" tradeoff made in Day 10, here applied to a
  different technique.
- **100% held-up rate is unusually clean** — see the discussion above;
  worth re-validating with looser mining thresholds to see if the rate
  holds, before assuming every future rule-mining run will look this good.
- **No genuinely new time period exists to test rule decay against** —
  `monitor.py` reuses the same held-out test set as training-time
  validation, which confirms the monitoring code works but not that
  real rule decay is being caught (same limitation Day 2's fraud
  project had for a similar reason).
- **Recommendations only use single-product antecedents** at serving
  time — multi-item basket-level recommendations (based on everything
  currently in a cart) aren't implemented in this demo API.
