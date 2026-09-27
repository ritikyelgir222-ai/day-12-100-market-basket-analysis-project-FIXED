"""
Phase 10: MLOps & Deployment
--------------------------------
A minimal FastAPI service that returns cross-sell recommendations for a
given product, backed by the out-of-sample-VALIDATED rule table
(outputs/deployable_rules.csv) rather than the full raw rule-mining
output -- only rules that held up on baskets the algorithm never saw
during mining are ever served.

Run with:  uvicorn app:app --reload
"""

import json

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Cross-Sell Recommendation API", version="1.0")

RULES = pd.read_csv("outputs/deployable_rules.csv")
PRODUCT_NAMES = pd.read_csv("outputs/product_names.csv", index_col=0)["Description"]
ALL_PRODUCTS = sorted(PRODUCT_NAMES.items(), key=lambda kv: kv[1])


class RecommendationRequest(BaseModel):
    stock_code: str
    top_n: int = 5


class Recommendation(BaseModel):
    stock_code: str
    description: str
    confidence: float
    lift: float


class RecommendationResponse(BaseModel):
    queried_product: str
    recommendations: list[Recommendation]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    options = "".join(f'<option value="{code}">{name}</option>' for code, name in ALL_PRODUCTS)
    available_counts = (
        RULES[~RULES["consequent"].str.contains(",")]
        .groupby("antecedent")
        .size()
        .to_dict()
    )
    return f"""
    <html>
    <head><title>Cross-Sell Recommendations</title></head>
    <body style="font-family: sans-serif; max-width: 640px; margin: 40px auto;">
        <h2>Cross-Sell Recommendations — Test Form</h2>
        <p>Pick a product to see what's genuinely (out-of-sample-validated)
           bought alongside it. Full API docs at <a href="/docs">/docs</a>.</p>
        <form id="recForm">
            <label>Product<br>
                <input name="stock_code" list="productCodes" placeholder="Type a stock code, e.g. 22699"
                       style="width:100%; box-sizing:border-box" required>
                <datalist id="productCodes">{options}</datalist>
            </label><br><br>
            <label>Number of recommendations<br>
                <input name="top_n" type="number" min="1" value="5" required>
                <span id="availableCount"></span>
            </label><br><br>
            <button type="submit">Get recommendations</button>
        </form>
        <div id="result"></div>
        <script>
        const availableCounts = {json.dumps(available_counts)};
        const stockCodeInput = document.querySelector('[name="stock_code"]');
        const topNInput = document.querySelector('[name="top_n"]');
        const availableCount = document.getElementById("availableCount");

        function updateAvailableCount() {{
            const count = availableCounts[stockCodeInput.value.trim()] || 0;
            topNInput.max = count || 1;
            topNInput.value = Math.min(Number(topNInput.value) || 1, count || 1);
            availableCount.innerText = count ? " (max " + count + ")" : " (no recommendations available)";
        }}

        stockCodeInput.addEventListener("input", updateAvailableCount);
        updateAvailableCount();

        document.getElementById("recForm").addEventListener("submit", async function(e) {{
            e.preventDefault();
            const form = new FormData(e.target);
            const maxAvailable = availableCounts[form.get("stock_code").trim()] || 1;
            const res = await fetch("/recommend", {{
                method: "POST",
                headers: {{"Content-Type": "application/json"}},
                body: JSON.stringify({{
                    stock_code: form.get("stock_code").trim(),
                    top_n: Math.min(Number(form.get("top_n")), maxAvailable)
                }})
            }});
            if (!res.ok) {{
                document.getElementById("result").innerText = "Error " + res.status + ": " + await res.text();
                return;
            }}
            const data = await res.json();
            let html = "<h3>Also bought with: " + data.queried_product + "</h3><ul>";
            for (const r of data.recommendations) {{
                html += "<li>" + r.description + " (confidence " + (r.confidence*100).toFixed(0) + "%, lift " + r.lift.toFixed(1) + "x)</li>";
            }}
            html += "</ul>";
            document.getElementById("result").innerHTML = html;
        }});
        </script>
    </body>
    </html>
    """


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(record: RecommendationRequest):
    code = record.stock_code
    queried_name = PRODUCT_NAMES.get(code, code)

    # A rule "holds" for this query if the ENTIRE antecedent is a single
    # product matching the requested code -- WHY single-item antecedents
    # only for a live recommendation: a real-time "you're viewing this
    # product" context only has one product to anchor on, not the
    # multi-item combinations some mined rules use as their antecedent
    # (those are more useful for basket/cart-level analysis than a
    # single product page).
    #
    # BUG FIX: the same "one product to anchor on" reasoning applies
    # symmetrically to what gets RECOMMENDED, not just to what's being
    # queried. Some rules' antecedent is a single product, but their
    # CONSEQUENT is a multi-item bundle (e.g. "22697, 22699, 22423") --
    # serving that as one Recommendation produced a fake, non-clickable
    # "stock_code" that isn't a real product at all (checked: this
    # affected 81 of 439 single-antecedent rules, ~18%, so it showed up
    # for a real share of queries, not just an edge case). A cross-sell
    # slot on a single product page needs to link to ONE real product,
    # so multi-item-consequent rules are excluded here. Checked that
    # every one of the 90 products with any single-antecedent rule still
    # has at least one clean, single-consequent rule after this filter --
    # no product loses recommendations entirely from this fix.
    matches = RULES[
        (RULES["antecedent"] == code) & (~RULES["consequent"].str.contains(","))
    ].copy()
    matches = matches.sort_values("test_lift", ascending=False).head(record.top_n)

    recs = [
        Recommendation(
            stock_code=row["consequent"],
            description=row["consequent_name"],
            confidence=round(row["test_confidence"], 4),
            lift=round(row["test_lift"], 4),
        )
        for _, row in matches.iterrows()
    ]

    return RecommendationResponse(queried_product=queried_name, recommendations=recs)
