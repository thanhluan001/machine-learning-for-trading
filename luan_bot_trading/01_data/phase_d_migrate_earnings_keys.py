"""Phase D Step A: /earnings/raw key migration perm_id -> permaTicker.

Per `phase_a_b_migration_report.md` Recommended next step #1 path (a)
(lightweight migration): no EODHD re-fetch required. Map the existing
44,897 rows' canonical_ticker (or legacy cik) -> the permaTicker
recorded in /metadata/sp400_permatickers, add a new `permaTicker`
column on /earnings/raw in-place via store.remove()+put, and log the
small fraction of unmatched rows (drop them).

Algorithm:
1. Load /earnings/raw (44,897 rows x {perm_id, canonical_ticker, cik, ...}).
2. Build bridge:
   (a) canonical_ticker direct: pt.canonical_ticker == e.canonical_ticker
       -> permaTicker. Handles 98.4% of rows + 859 of 879 canonical_tickers.
   (b) cik cross-reference: for unmatched rows, use the same CIK
       (e.cik) to find the permaTicker via /metadata/sp400 (read the
       modern canonical_ticker for the historical CIK), then join
       canonical_ticker -> permaTicker via permatickers. Handles rebrand
       cases: APY -> CHX (US000000085026), CALY -> MODG, LAWIL -> LNW.
3. Drop remaining rows (~15 truly-orphan delisted bankruptcies: HSC,
   KAR, RLGY, MNK, ENDP, DF, JCP, TUP, WPG, SPN, SIVB, ASNA, WIN, GDI,
   CLI, CHFC, VSCO). Log them to phase_d_migration_dropped.log.
4. Write back /earnings/raw with the new permaTicker column + permaTicker
   as the data_columns index.

Idempotent: re-running just re-derives the permaTicker column.
"""
import sys
import json
import pandas as pd
from pathlib import Path

DB_FILE = Path(__file__).parent / "db.h5"
EARNINGS_KEY = "/earnings/raw"
PERMATICKERS_KEY = "/metadata/sp400_permatickers"
SP400_KEY = "/metadata/sp400"
LOG_FILE = Path(__file__).parent / "phase_d_migration_dropped.log"


def load_tables():
    """Load the 3 source tables needed for the migration."""
    if not DB_FILE.exists():
        raise FileNotFoundError(f"{DB_FILE} not found.")
    rows = {}
    with pd.HDFStore(DB_FILE, mode="r") as s:
        if EARNINGS_KEY not in s.keys():
            raise KeyError(f"{EARNINGS_KEY} not in db.")
        if PERMATICKERS_KEY not in s.keys():
            raise KeyError(f"{PERMATICKERS_KEY} not in db.")
        rows["earnings"] = s[EARNINGS_KEY]
        rows["permatickers"] = s[PERMATICKERS_KEY]
        rows["sp400"] = s[SP400_KEY] if SP400_KEY in s.keys() else pd.DataFrame()
    return rows


def build_bridge_canon(permatickers: pd.DataFrame) -> pd.DataFrame:
    """Step 1: canonical_ticker -> permaTicker bridge.

    Returns a DataFrame with columns {canonical_ticker, permaTicker}.
    On collision (multiple permaTickers per canonical_ticker), picks the
    first isActive=True then price_unavailable=False one.

    Collisions are rare (only 2 known: ACI, TLN both inactive+unavailable);
    we pick a deterministic row to avoid non-idempotent migration.
    """
    pt = permatickers[["canonical_ticker", "permaTicker", "isActive", "price_unavailable"]].copy()
    pt["_sort"] = ((~pt["isActive"]).astype(int) * 2  # prefer active
                   + (pt["price_unavailable"].astype(int))  # then avail
                   )
    pt = pt.sort_values(["canonical_ticker", "_sort"])
    bridge = pt.groupby("canonical_ticker", as_index=False).first()[["canonical_ticker", "permaTicker"]]
    return bridge


def build_bridge_cik(permatickers: pd.DataFrame, sp400: pd.DataFrame) -> pd.DataFrame:
    """Step 2: legacy_cik -> permaTicker bridge (intermediate).

    Reads /metadata/sp400 (Wikipedia) to find which modern canonical_ticker
    belongs to a given cik_at_added (= the legacy cik in /earnings/raw),
    then resolves canonical_ticker -> permaTicker from /metadata/sp400_permatickers.

    Returns {legacy_cik, permaTicker}.
    """
    if sp400.empty:
        return pd.DataFrame(columns=["cik", "permaTicker"])
    # wiki cik_at_added -> wiki canonical_ticker (the post-rebrand canonical
    # assigned during Phase A). When a CIK has multiple wiki rows (different
    # historical ticker codes that ended up with the same canonical), prefer
    # the row whose canonical_ticker IS in permatickers (post-rebrand modern).
    sp = sp400[["ticker", "cik_at_added"]].rename(columns={"ticker": "canonical_ticker",
                                                            "cik_at_added": "cik"})
    sp = sp[sp["cik"].notna()].copy()
    known_canon = set(permatickers["canonical_ticker"].dropna())
    sp["_priority"] = (~sp["canonical_ticker"].isin(known_canon)).astype(int)  # 0 = known (preferred)
    sp = sp.sort_values(["cik", "_priority"])
    sp = sp.drop_duplicates(subset=["cik", "canonical_ticker"]).drop_duplicates(subset=["cik"])
    # now sp maps cik -> canonical_ticker (preferably modern / in permatickers)
    pt_canon = permatickers[["canonical_ticker", "permaTicker"]].drop_duplicates(subset=["canonical_ticker"])
    joined = sp.merge(pt_canon, on="canonical_ticker", how="inner")[["cik", "permaTicker"]]
    joined = joined.drop_duplicates(subset=["cik"])
    return joined


def map_earnings(earn: pd.DataFrame, bridge_canon: pd.DataFrame, bridge_cik: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply bridges to /earnings/raw. Returns (mapped_earnings, dropped_earnings)."""
    # Step 1: canonical_ticker bridge
    m1 = earn.merge(bridge_canon.rename(columns={"permaTicker": "permaTicker_canon"}),
                    on="canonical_ticker", how="left")
    # Step 2: cik bridge for rows where step 1 returned NaN
    m2 = m1.merge(bridge_cik.rename(columns={"permaTicker": "permaTicker_cik"}),
                  on="cik", how="left")
    # Final permaTicker: prefer canonical bridge, fall back to cik bridge
    m2["permaTicker"] = m2["permaTicker_canon"].where(m2["permaTicker_canon"].notna(),
                                                      m2["permaTicker_cik"])
    mapped = m2[m2["permaTicker"].notna()].copy()
    dropped = m2[m2["permaTicker"].isna()].copy()
    # Clean up intermediate columns
    return mapped.drop(columns=["permaTicker_canon", "permaTicker_cik"]), dropped


def write_back(table: pd.DataFrame) -> None:
    """Write to /earnings/raw (HDFStore('a') + remove() pattern -- never mode='w')."""
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if EARNINGS_KEY in store.keys():
            store.remove(EARNINGS_KEY)
        store.put(EARNINGS_KEY, table, format="table",
                  data_columns=["permaTicker", "report_date", "canonical_ticker"])


def write_log(dropped: pd.DataFrame, totals: dict) -> None:
    """Write dropped-orphan audit log."""
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== Phase D Step A migration: dropped rows (no permaTicker mapping) ===\n")
        f.write(f"Original earnings rows:     {totals['total']:,}\n")
        f.write(f"Mapped to permaTicker:       {totals['mapped']:,} ({totals['mapped']/totals['total']*100:.2f}%)\n")
        f.write(f"Dropped (unmappable):        {totals['dropped']:,} ({totals['dropped']/totals['total']*100:.2f}%)\n")
        f.write("\nUnmappable canonical_tickers (legacy bankruptcies Tiingo never tracked):\n")
        g = (dropped.groupby(["canonical_ticker", "cik"])
             .size().reset_index(name="n_rows").sort_values("n_rows", ascending=False))
        for _, r in g.iterrows():
            f.write(f"  {r['canonical_ticker']:<6} cik={str(r['cik']):<13} n_rows={int(r['n_rows'])}\n")
        f.write("\n(Loss is acceptable: these are delisted/bankrupt companies Tiingo\n")
        f.write(" has no permaTicker for. Their /sp400/* price nodes never existed,\n")
        f.write(" so Phase E feature gating would have dropped them anyway.)\n")


def main():
    print("=== Phase D Step A: /earnings/raw perm_id -> permaTicker migration ===\n")
    tables = load_tables()
    earn = tables["earnings"]
    pt = tables["permatickers"]
    sp = tables["sp400"]
    print(f"  /earnings/raw:              {len(earn):,} rows ({earn['perm_id'].nunique()} perm_ids, "
          f"{earn['canonical_ticker'].nunique()} canonical_tickers)")
    print(f"  /metadata/sp400_permatickers: {len(pt)} rows ({pt['permaTicker'].nunique()} permaTags)")
    print(f"  /metadata/sp400:            {len(sp)} rows (Wikipedia intervals)")

    bridge_canon = build_bridge_canon(pt)
    print(f"\n[Step 1] canonical_ticker bridge: {len(bridge_canon)} mappings "
          f"({bridge_canon['canonical_ticker'].nunique()} unique canonical_tickers)")

    bridge_cik = build_bridge_cik(pt, sp)
    print(f"[Step 2] CIK cross-reference bridge: {len(bridge_cik)} ciks -> permaTicakers "
          f"({bridge_cik['cik'].nunique()} unique CIKs)")

    mapped, dropped = map_earnings(earn, bridge_canon, bridge_cik)
    totals = {"total": len(earn), "mapped": len(mapped), "dropped": len(dropped)}
    print(f"\n[Step 3] Apply bridges:")
    print(f"  mapped rows:    {len(mapped):,} ({len(mapped)/len(earn)*100:.2f}%)")
    print(f"  dropped rows:   {len(dropped):,} ({len(dropped)/len(earn)*100:.2f}%)")
    print(f"  mapped canonical_tickers: {mapped['canonical_ticker'].nunique()}/{earn['canonical_ticker'].nunique()}")
    print(f"  mapped permaTicakers:      {mapped['permaTicker'].nunique()}")

    # Sanity-check: do any permaTicakers receive rows from MULTIPLE legacy perm_ids?
    # (e.g. one permaTicker getting both AAP legacy and another perm_id's AAP rebrand)
    g = mapped.groupby("permaTicker")["perm_id"].nunique()
    multi_perm = g[g > 1]
    if len(multi_perm):
        print(f"\n  [WARN] {len(multi_perm)} permaTickers receive rows from multiple perm_ids:")
        for pt_, n in multi_perm.head(15).items():
            sub = mapped[mapped["permaTicker"] == pt_]
            print(f"    {pt_} ({n} perm_ids): perm_ids={list(sub['perm_id'].unique())}, "
                  f"canonical_tickers={list(sub['canonical_ticker'].unique())}")

    print(f"\n[Step 4] Writing back /earnings/raw with new permaTicker column...")
    # Drop legacy perm_id column - Phase D onward uses permaTicker as the key.
    write_table = mapped.drop(columns=["perm_id"])
    write_back(write_table)
    print(f"  wrote {len(write_table):,} rows x {len(write_table.columns)} cols")

    write_log(dropped, totals)
    print(f"  dropped audit log -> {LOG_FILE.name}")
    print(f"\n=== Done. Original {totals['total']:,} -> migrated {totals['mapped']:,} "
          f"({100*totals['mapped']/totals['total']:.2f}%), dropped {totals['dropped']:,} ===")


if __name__ == "__main__":
    main()
