#!/usr/bin/env python3
"""80_megatrend_normalize_insider_news.py — RC-4 Step 3 normalization.

PURPOSE
-------
Normalize insider selling and operational-news context without silently using
look-ahead data or confusing data absence with a quiet signal. This is a
research/data-quality artifact. It does not alter Script 74 thresholds and does
not produce trading orders.

POINT-IN-TIME CONTRACT
----------------------
For each month-end asof date, only filings/articles with date <= asof are
visible. Historical baselines use strictly earlier monthly observations.
No current market cap, current shares outstanding, or retrospective vendor
fundamental snapshot is used: the caches do not contain a point-in-time market
cap series.

INSIDER NORMALIZATION
---------------------
- exact open-market Form 4 P-Purchase and S-Sale transactions only;
- remove exact/near duplicate transaction rows;
- exclude awards, gifts, option exercises, exempt, derivative and other codes;
- use company-relative historical percentile of trailing-90d sell dollars;
- report seller/buyer counts, cross-company seller breadth, dollar net flow,
  coverage, and number of companies with an unusually high selling score;
- candidate status is based on >=2 companies with unusually high selling and
  >=2 independent sellers, not on raw theme dollars;
- raw dollars are retained for audit only because vendor values can contain
  split/entity anomalies and no point-in-time market-cap denominator exists;
- a candidate cluster requires >=2 companies and >=2 independent sellers, but
  this is descriptive only.

NEWS NORMALIZATION
------------------
- deduplicate to one event per company/day/category/direction;
- fixed operational taxonomy: demand/orders, guidance/financial, capex/
  capacity, supply/pricing, labor/operations, balance-sheet/solvency;
- count negative/positive company-days rather than raw articles;
- normalize each company's current 90d negative event count against its own
  prior monthly history; report raw article/company-day denominators, event
  rates, and missing text;
- source provenance is explicitly unavailable in this cache and is reported.

OUTPUT
------
- archive/experiments/rc4_normalized_insider_news_monthly.csv
- archive/experiments/rc4_normalized_insider_news_summary.json

Interpretation is intentionally conservative. This script establishes usable
features and audit fields; a separate pre-registered episode study is required
before any warning threshold can be promoted.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
load_dotenv(ROOT / ".env")
DB_INSIDER = ROOT / "01_data" / "db_insider_megatrend.h5"
DB_NEWS = ROOT / "01_data" / "db_news.h5"
OUT_CSV = HERE / "archive" / "experiments" / "rc4_normalized_insider_news_monthly.csv"
OUT_JSON = HERE / "archive" / "experiments" / "rc4_normalized_insider_news_summary.json"

THEMES = {
    "AI/hyperscale": ["MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL"],
    "clean_energy": ["FSLR", "ENPH", "SEDG", "NEE", "RUN", "PLUG"],
    "crypto": ["MSTR", "COIN", "RIOT", "MARA", "CLSK"],
}

# The taxonomy is deliberately narrow. A category can match both directions;
# direction is resolved at the article level and then counted once per day.
NEWS_RULES = {
    "demand_orders": {
        "negative": r"(?:weak|soft|slow|slowing|collapse|declin|reduc|cancel|delay|cut).{0,35}(?:demand|orders?|backlog|bookings?)|(?:demand|orders?|backlog|bookings?).{0,35}(?:weak|soft|slow|slowing|collapse|declin|reduc|cancel|delay|cut)",
        "positive": r"(?:strong|robust|record|accelerat|increas|expand|rais).{0,35}(?:demand|orders?|backlog|bookings?)|(?:demand|orders?|backlog|bookings?).{0,35}(?:strong|robust|record|accelerat|increas|expand)",
    },
    "guidance_financial": {
        "negative": r"(?:lower|cut|reduc|withdraw|miss|weak).{0,25}(?:guidance|outlook|forecast|revenue|earnings|margin)|(?:guidance|outlook|forecast).{0,25}(?:lower|cut|reduc|withdraw|miss|weak)",
        "positive": r"(?:rais|increas|boost|record|beat|strong).{0,25}(?:guidance|outlook|forecast|revenue|earnings|margin)|(?:guidance|outlook|forecast).{0,25}(?:rais|increas|boost|record|beat|strong)",
    },
    "capex_capacity": {
        "negative": r"(?:cut|reduc|lower|slash|delay|cancel).{0,35}(?:capex|capital expenditure|capacity|expansion|factory|plant)|(?:capex|capital expenditure|capacity|expansion|factory|plant).{0,35}(?:cut|reduc|lower|slash|delay|cancel)",
        "positive": r"(?:increas|boost|rais|expand|build|open|add|invest).{0,35}(?:capex|capital expenditure|capacity|expansion|factory|plant)|(?:capex|capital expenditure|capacity|expansion|factory|plant).{0,35}(?:increas|boost|rais|expand|build|open|add|invest)",
    },
    "supply_pricing": {
        "negative": r"(?:oversupply|overcapacity|inventory glut|pricing pressure|price cut|price war|commodit|excess inventory|surplus)",
        "positive": r"(?:pricing power|price increas|price rais|tight supply|shortage|capacity constraint)",
    },
    "labor_operations": {
        "negative": r"(?:layoff|job cut|workforce reduc|strike|shutdown|production cut|operational disruption|restructur|bankrupt|insolv|recall|investigat)",
        "positive": r"(?:hire|hiring|new jobs|production increas|ramp|operational improv|efficien|productivity)",
    },
    "balance_sheet_solvency": {
        "negative": r"(?:liquidity|cash burn|debt default|covenant|going concern|bankrupt|insolv|distress|capital rais|dilution|credit downgrade|downgrade)",
        "positive": r"(?:deleverag|debt repay|cash flow positive|liquidity increas|credit upgrade|refinanc|capital return)",
    },
}
COMPILED_RULES = {
    cat: {direction: re.compile(pattern, re.I | re.S) for direction, pattern in rules.items()}
    for cat, rules in NEWS_RULES.items()
}


def asof_months(start: str = "2015-01-31") -> pd.DatetimeIndex:
    dates = []
    for path, col in [(DB_INSIDER, "filingDate"), (DB_NEWS, "publishedDate")]:
        if not path.exists():
            continue
        with pd.HDFStore(path, mode="r") as store:
            for key in list(store.keys()):
                d = pd.to_datetime(store[key][col], errors="coerce")
                if len(d):
                    dates.append(d.min())
                    dates.append(d.max())
    end = max(dates).normalize() if dates else pd.Timestamp.today().normalize()
    # Label the current partial month by its calendar month-end, but the
    # as-of contract remains <= the latest actually available date.
    end = end.to_period("M").to_timestamp("M")
    return pd.date_range(start, end, freq="ME")


def clean_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()


def load_insider_events() -> tuple[pd.DataFrame, dict]:
    rows = []
    audit = {"keys": 0, "rows_raw": 0, "rows_open_market": 0, "rows_dedup": 0,
             "excluded_non_open_market": 0, "invalid_dates": 0, "symbols": {}}
    with pd.HDFStore(DB_INSIDER, mode="r") as store:
        keys = list(store.keys()); audit["keys"] = len(keys)
        for key in keys:
            sym = key.rsplit("/", 1)[-1]
            d = store[key].copy(); audit["rows_raw"] += len(d)
            if d.empty: continue
            d["symbol"] = sym
            d["filingDate"] = clean_date(d["filingDate"])
            audit["invalid_dates"] += int(d.filingDate.isna().sum())
            typ = d.transactionType.astype(str)
            # Do not infer open-market status from acquisition/disposition.
            # FMP's exact P-Purchase/S-Sale code is the conservative contract.
            d["direction"] = np.select([typ.eq("P-Purchase"), typ.eq("S-Sale")], ["buy", "sell"], default="")
            open_market = d.direction.ne("") & d.filingDate.notna()
            audit["excluded_non_open_market"] += int((~open_market).sum())
            d = d[open_market].copy(); audit["rows_open_market"] += len(d)
            d["value"] = pd.to_numeric(d.get("value", 0), errors="coerce").fillna(0.0)
            d["reportingName"] = d.get("reportingName", "").fillna("").astype(str).str.strip()
            # FMP cache can contain repeated rows from overlapping pages and
            # amended records. Keep one identical economic transaction.
            dedup_cols = [c for c in ["symbol", "filingDate", "transactionDate", "direction",
                                      "reportingName", "securitiesTransacted", "price",
                                      "value", "securityName", "acquisitionOrDisposition"] if c in d.columns]
            d = d.drop_duplicates(dedup_cols)
            audit["rows_dedup"] += len(d)
            audit["symbols"][sym] = {"rows": len(d), "first": str(d.filingDate.min().date()), "last": str(d.filingDate.max().date())}
            rows.append(d[["symbol", "filingDate", "direction", "reportingName", "value"]])
    if not rows:
        return pd.DataFrame(columns=["symbol", "filingDate", "direction", "reportingName", "value"]), audit
    return pd.concat(rows, ignore_index=True), audit


def rolling_insider_snapshot(events: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    out = []
    for theme, members in THEMES.items():
        ev = events[events.symbol.isin(members)].copy()
        company_month = []
        for sym in members:
            e = ev[ev.symbol.eq(sym)]
            vals = []
            for asof in months:
                x = e[(e.filingDate <= asof) & (e.filingDate > asof - pd.Timedelta(days=90))]
                sell = x[x.direction.eq("sell")]; buy = x[x.direction.eq("buy")]
                vals.append({"sell_dollars": sell.value.sum(), "buy_dollars": buy.value.sum(),
                             "seller_count": sell.reportingName[sell.reportingName.ne("")].nunique(),
                             "buyer_count": buy.reportingName[buy.reportingName.ne("")].nunique(),
                             "has_sell": int(not sell.empty), "has_buy": int(not buy.empty)})
            q = pd.DataFrame(vals, index=months); q["symbol"] = sym; company_month.append(q)
        cm = pd.concat(company_month)
        # Baselines use strictly earlier monthly rows for each company.
        for asof in months:
            cur = cm.loc[cm.index == asof]
            prior = cm.loc[cm.index < asof]
            current = cur.set_index("symbol"); row = {"month": asof, "theme": theme}
            row["companies_available"] = len(members)
            row["companies_with_sell"] = int((current.sell_dollars >= 50000).sum())
            row["companies_with_buy"] = int((current.buy_dollars >= 50000).sum())
            row["seller_count"] = int(current.seller_count.sum()); row["buyer_count"] = int(current.buyer_count.sum())
            row["sell_transaction_count"] = int((current.sell_dollars > 0).sum())
            row["sell_insider_count"] = int(current.seller_count.sum())
            row["sell_dollars"] = float(current.sell_dollars.sum()); row["buy_dollars"] = float(current.buy_dollars.sum())
            row["net_dollars"] = row["buy_dollars"] - row["sell_dollars"]
            unusual = 0; valid_baselines = 0; company_scores = []
            for sym in members:
                hist = prior[prior.symbol.eq(sym)].sell_dollars
                # Require a meaningful history and ignore zero-only baselines.
                hist = hist[hist.notna()]
                val = float(current.loc[sym, "sell_dollars"])
                # A row of zero-valued months is not historical coverage. The
                # underlying company must have at least 12 distinct filing
                # months before it is eligible for a company-relative baseline.
                source = ev[ev.symbol.eq(sym)]
                source_months = source.loc[source.filingDate < asof, "filingDate"].dt.to_period("M").nunique()
                if len(hist) >= 12 and source_months >= 12:
                    valid_baselines += 1
                    pct = float((hist < val).mean()) if val > 0 else 0.0
                    company_scores.append(pct)
                    # Require breadth of independent sellers. This prevents a
                    # single anomalous vendor dollar from being a cluster.
                    current_sellers = int(current.loc[sym, "seller_count"])
                    if val >= 50000 and pct >= .90 and current_sellers >= 2: unusual += 1
            row["baseline_companies"] = valid_baselines
            row["unusual_seller_companies"] = unusual
            row["median_company_sell_percentile"] = float(np.median(company_scores)) if company_scores else np.nan
            row["cluster_warning_candidate"] = bool(unusual >= 2 and row["seller_count"] >= 2)
            row["insider_coverage_ok"] = bool(valid_baselines >= max(2, len(members) // 2))
            row["insider_source_companies"] = int(sum(not ev[ev.symbol.eq(sym)].empty for sym in members))
            out.append(row)
    return pd.DataFrame(out)


def classify_news(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame(columns=["symbol", "publishedDate", "text", "category", "direction"])
    d = d.copy(); d["publishedDate"] = clean_date(d["publishedDate"])
    d = d[d.publishedDate.notna()].copy()
    d["title"] = d.get("title", "").fillna("").astype(str)
    d["text"] = d.get("text", "").fillna("").astype(str)
    d["body"] = (d.title + " " + d.text).str.replace(r"\s+", " ", regex=True).str.strip()
    d["title_key"] = d.title.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    d = d.drop_duplicates(["symbol", "publishedDate", "title_key", "body"])
    rows = []
    for r in d.itertuples(index=False):
        day = r.publishedDate
        for category, rules in COMPILED_RULES.items():
            neg = bool(rules["negative"].search(r.body)); pos = bool(rules["positive"].search(r.body))
            if neg or pos:
                # If both occur, retain both as separate directional evidence;
                # monthly aggregation deduplicates company/day/category/direction.
                if neg: rows.append((r.symbol, day, category, "negative", r.body))
                if pos: rows.append((r.symbol, day, category, "positive", r.body))
    return pd.DataFrame(rows, columns=["symbol", "publishedDate", "category", "direction", "text"])


def load_news_events() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []; audit = {"keys": 0, "articles_raw": 0, "articles_dedup": 0, "classified_events": 0,
                        "missing_title": 0, "missing_text": 0, "symbols": {}}
    with pd.HDFStore(DB_NEWS, mode="r") as store:
        keys = list(store.keys()); audit["keys"] = len(keys)
        for key in keys:
            sym = key.rsplit("/", 1)[-1]; d = store[key].copy(); audit["articles_raw"] += len(d)
            if d.empty: continue
            audit["missing_title"] += int(d.title.isna().sum()); audit["missing_text"] += int(d.text.isna().sum())
            d["symbol"] = sym; d["publishedDate"] = clean_date(d.publishedDate)
            d["title"] = d.title.fillna("").astype(str); d["text"] = d.text.fillna("").astype(str)
            d["title_key"] = d.title.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
            d["body"] = (d.title + " " + d.text).str.replace(r"\s+", " ", regex=True).str.strip()
            d = d.drop_duplicates(["symbol", "publishedDate", "title_key", "body"])
            audit["articles_dedup"] += len(d)
            audit["symbols"][sym] = {"articles": len(d), "first": str(d.publishedDate.min().date()), "last": str(d.publishedDate.max().date())}
            rows.append(d[["symbol", "publishedDate", "title", "text"]])
    raw = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["symbol", "publishedDate", "title", "text"])
    events = classify_news(raw); audit["classified_events"] = len(events)
    return raw, events, audit


def rolling_news_snapshot(raw: pd.DataFrame, events: pd.DataFrame, raw_audit: dict, months: pd.DatetimeIndex) -> pd.DataFrame:
    out = []
    for theme, members in THEMES.items():
        ev = events[events.symbol.isin(members)].copy()
        raw_theme = raw[raw.symbol.isin(members)].copy()
        # Build monthly snapshots per company first, then theme aggregate.
        by_company = []
        for sym in members:
            e = ev[ev.symbol.eq(sym)]
            a = raw_theme[raw_theme.symbol.eq(sym)]
            vals = []
            for asof in months:
                lo = asof - pd.Timedelta(days=90)
                x = e[(e.publishedDate <= asof) & (e.publishedDate > lo)]
                all_articles = a[(a.publishedDate <= asof) & (a.publishedDate > lo)]
                neg = x[x.direction.eq("negative")]; pos = x[x.direction.eq("positive")]
                # Category-day is retained for breadth; company-day is the
                # primary warning unit and cannot be inflated by taxonomy hits.
                neg_company_days = neg[["symbol", "publishedDate"]].drop_duplicates().shape[0]
                pos_company_days = pos[["symbol", "publishedDate"]].drop_duplicates().shape[0]
                vals.append({"negative_days": neg[["publishedDate", "category"]].drop_duplicates().shape[0],
                             "positive_days": pos[["publishedDate", "category"]].drop_duplicates().shape[0],
                             "negative_company_days": neg_company_days,
                             "positive_company_days": pos_company_days,
                             "negative_categories": neg.category.nunique(), "positive_categories": pos.category.nunique(),
                             "operational_days": x.publishedDate.nunique(),
                             "raw_articles": len(all_articles),
                             "raw_company_days": all_articles[["symbol", "publishedDate"]].drop_duplicates().shape[0],
                             "raw_text_articles": int((all_articles.text.fillna("").str.len() > 0).sum())})
            q = pd.DataFrame(vals, index=months); q["symbol"] = sym; by_company.append(q)
        cm = pd.concat(by_company)
        for asof in months:
            cur = cm.loc[cm.index == asof]; prior = cm.loc[cm.index < asof]
            row = {"month": asof, "theme": theme, "news_companies_available": len(members)}
            for col in ["negative_days", "positive_days", "negative_company_days", "positive_company_days",
                        "operational_days", "raw_articles", "raw_company_days", "raw_text_articles",
                        "negative_categories", "positive_categories"]:
                row[col] = int(cur[col].sum())
            row["negative_companies"] = int((cur.negative_company_days > 0).sum()); row["positive_companies"] = int((cur.positive_company_days > 0).sum())
            row["negative_company_day_rate"] = row["negative_company_days"] / row["raw_company_days"] if row["raw_company_days"] else np.nan
            row["text_coverage_rate"] = row["raw_text_articles"] / row["raw_articles"] if row["raw_articles"] else np.nan
            scores = []; unusual = 0; baseline_companies = 0
            for sym in members:
                hist = prior[prior.symbol.eq(sym)].negative_company_days
                val = float(cur.loc[cur.symbol.eq(sym), "negative_company_days"].iloc[0])
                source = raw_theme[raw_theme.symbol.eq(sym)]
                source_months = source.loc[source.publishedDate < asof, "publishedDate"].dt.to_period("M").nunique()
                if len(hist) >= 12 and source_months >= 12:
                    baseline_companies += 1; pct = float((hist < val).mean()); scores.append(pct)
                    if val >= 2 and pct >= .90: unusual += 1
            row["baseline_news_companies"] = baseline_companies
            row["unusual_negative_companies"] = unusual
            row["median_negative_percentile"] = float(np.median(scores)) if scores else np.nan
            row["news_coverage_ok"] = bool(baseline_companies >= max(2, len(members) // 2) and row["raw_company_days"] >= 2)
            row["news_warning_candidate"] = bool(unusual >= 2 and row["negative_companies"] >= 2 and row["negative_company_days"] > row["positive_company_days"] and row["news_coverage_ok"])
            row["news_cache_symbols_present"] = int(sum(1 for sym in members if sym in raw_audit.get("symbols", {})))
            row["news_source_companies"] = int(sum(not raw_theme[raw_theme.symbol.eq(sym)].empty for sym in members))
            row["source_provenance_available"] = False
            out.append(row)
    return pd.DataFrame(out)


def main():
    print("=" * 92)
    print("RC-4 STEP 3 — NORMALIZED INSIDER SELLING + OPERATIONAL NEWS")
    print("=" * 92)
    months = asof_months()
    print(f"[1] point-in-time month range: {months[0].date()} -> {months[-1].date()} ({len(months)} months)")
    print("[2] loading and auditing Form 4 cache ...")
    insider_events, insider_audit = load_insider_events()
    print(f"    raw={insider_audit['rows_raw']:,} open-market={insider_audit['rows_open_market']:,} dedup={insider_audit['rows_dedup']:,} excluded={insider_audit['excluded_non_open_market']:,}")
    insider = rolling_insider_snapshot(insider_events, months)
    print("[3] loading and classifying news cache ...")
    news_raw, news_events, news_audit = load_news_events()
    print(f"    raw={news_audit['articles_raw']:,} dedup={news_audit['articles_dedup']:,} operational events={news_audit['classified_events']:,} missing_text={news_audit['missing_text']:,}")
    news = rolling_news_snapshot(news_raw, news_events, news_audit, months)
    result = insider.merge(news, on=["month", "theme"], how="outer")
    result["month"] = pd.to_datetime(result.month).dt.strftime("%Y-%m-%d")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True); result.to_csv(OUT_CSV, index=False)
    summary = {"script": "80_megatrend_normalize_insider_news.py", "status": "research_only",
        "point_in_time": True, "market_cap_normalization": "not_available_without_point_in_time_series",
        "insider_audit": insider_audit, "news_audit": news_audit,
        "taxonomy": list(NEWS_RULES), "rows": len(result),
        "candidate_counts": {"insider": int(result.cluster_warning_candidate.sum()), "news": int(result.news_warning_candidate.sum())},
        "latest": result.sort_values(["month", "theme"]).tail(len(THEMES)).replace({np.nan: None}).to_dict("records")}
    OUT_JSON.write_text(json.dumps(summary, indent=1, default=str))
    print("\n[4] candidate diagnostic counts (not approved signals)")
    print(f"    insider cluster candidates: {summary['candidate_counts']['insider']} (breadth + within-company percentile; raw dollars not used)")
    print(f"    news warning candidates:     {summary['candidate_counts']['news']} (company-day rate + within-company percentile + coverage gate)")
    print("\n[5] output")
    print(f"    {OUT_CSV.relative_to(ROOT.parent)}")
    print(f"    {OUT_JSON.relative_to(ROOT.parent)}")
    print("\n[6] status: normalized features only; no Script 74 threshold or allocation rule changed.")

if __name__ == "__main__":
    main()
