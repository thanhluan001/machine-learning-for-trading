#!/usr/bin/env python3
"""
Analyst Revision Momentum Probe.

Question: When analysts UPGRADE a stock (raise their rating), does the stock
drift up over the following weeks/months? And do DOWNGRADES drift down?

The academic literature (Chan 1996, Womack 1996, Jegadeesh & Kim 2010) finds:
  - Upgrades generate +2-4% abnormal drift over 1-3 months
  - Downgrades generate -2-4% abnormal drift over 1-3 months
  - Markets underreact to the information content of analyst revisions

We test this on FMP grades data (/analyst/grades/) with various hold periods,
market-adjusted against IJH.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np, pandas as pd

DB = 'C:/Users/thanh/Projects/machine-learning-for-trading/luan_bot_trading/01_data/db.h5'

# Hold periods to test (in trading days)
HOLD_PERIODS = [5, 10, 21, 42, 63]  # 1wk, 2wk, 1mo, 2mo, 3mo

# Ordinal grade scale (1=bearish, 5=bullish)
GRADE_ORDINAL = {
    'Strong Buy': 5, 'Buy': 5,
    'Outperform': 4, 'Overweight': 4, 'Positive': 4, 'Accumulate': 4,
    'Equal Weight': 3, 'Market Perform': 3, 'Sector Perform': 3, 'Sector Weight': 3,
    'Neutral': 3, 'Hold': 3, 'In Line': 3, 'Perform': 3,
    'Underweight': 2, 'Underperform': 2, 'Reduce': 2,
    'Sell': 1,
}


def load_all_grades():
    """Load all grade nodes into one DataFrame."""
    all_grades = []
    with pd.HDFStore(DB, mode='r') as s:
        grade_keys = [k for k in s.keys() if k.startswith('/analyst/grades/')]
        for key in grade_keys:
            try:
                d = s[key]
                all_grades.append(d)
            except Exception:
                continue
    grades = pd.concat(all_grades, ignore_index=True)
    grades['date'] = pd.to_datetime(grades['date'])
    # Map grades to ordinal
    grades['prev_ordinal'] = grades['previous_grade'].map(GRADE_ORDINAL)
    grades['new_ordinal'] = grades['new_grade'].map(GRADE_ORDINAL)
    # Identify revision events (ordinal change or explicit action)
    grades['ordinal_change'] = grades['new_ordinal'] - grades['prev_ordinal']
    # An "upgrade" is: action=='upgrade' OR ordinal_change > 0
    # A "downgrade" is: action=='downgrade' OR ordinal_change < 0
    grades['revision_type'] = 'maintain'
    grades.loc[(grades['action'] == 'upgrade') | (grades['ordinal_change'] > 0), 'revision_type'] = 'upgrade'
    grades.loc[(grades['action'] == 'downgrade') | (grades['ordinal_change'] < 0), 'revision_type'] = 'downgrade'
    # Drop events where we can't map grades
    grades = grades.dropna(subset=['new_ordinal'])
    return grades


def compute_returns_for_events(events_df, db_path, hold_periods):
    """For each revision event, compute raw and market-adjusted returns."""
    # Load IJH benchmark
    with pd.HDFStore(db_path, mode='r') as s:
        ijh = s['/macros/IJH'].copy()
    ijh['Date'] = pd.to_datetime(ijh['Date'])
    ijh = ijh.sort_values('Date').reset_index(drop=True)
    ijh_close_col = 'Adj_Close' if 'Adj_Close' in ijh.columns else 'Close'
    ijh_dates = ijh['Date'].values
    ijh_prices = ijh[ijh_close_col].values

    # Group price data by permaTicker for efficiency
    price_cache = {}
    with pd.HDFStore(db_path, mode='r') as s:
        sp400_keys = set(k for k in s.keys() if k.startswith('/sp400/'))
        # We'll lazy-load per permaTicker

    results = []
    with pd.HDFStore(db_path, mode='r') as s:
        for idx, ev in events_df.iterrows():
            pt = ev['permaTicker']
            key = f'/sp400/{pt}'
            if key not in sp400_keys:
                continue
            if pt not in price_cache:
                try:
                    p = s[key].copy()
                    p['Date'] = pd.to_datetime(p['Date'])
                    p = p.sort_values('Date').reset_index(drop=True)
                    price_cache[pt] = p
                except Exception:
                    price_cache[pt] = None

            p = price_cache[pt]
            if p is None:
                continue

            event_date = ev['date']
            close_col = 'Adj_Close' if 'Adj_Close' in p.columns else 'Close'

            # Find event date index (first trading day on or after event)
            p_dates = p['Date'].values
            ev_mask = p_dates >= event_date.to_datetime64()
            if not ev_mask.any():
                continue
            t0 = int(np.argmax(ev_mask))

            entry_price = p.iloc[t0][close_col]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            # IJH return for same window
            ijh_mask = ijh_dates >= p_dates[t0]
            if not ijh_mask.any():
                continue
            ijh_t0 = int(np.argmax(ijh_mask))
            ijh_entry = ijh_prices[ijh_t0]
            if pd.isna(ijh_entry) or ijh_entry <= 0:
                continue

            row_result = {
                'permaTicker': pt,
                'ticker': ev.get('symbol', pt),
                'date': event_date,
                'year': event_date.year,
                'revision_type': ev['revision_type'],
                'ordinal_change': ev['ordinal_change'],
                'grading_company': ev.get('grading_company', ''),
                'prev_grade': ev.get('previous_grade', ''),
                'new_grade': ev.get('new_grade', ''),
            }

            for hold in hold_periods:
                tN = t0 + hold
                if tN >= len(p):
                    row_result[f'ret_{hold}d'] = np.nan
                    row_result[f'abn_{hold}d'] = np.nan
                    continue

                exit_price = p.iloc[tN][close_col]
                if pd.isna(exit_price) or exit_price <= 0:
                    row_result[f'ret_{hold}d'] = np.nan
                    row_result[f'abn_{hold}d'] = np.nan
                    continue

                stock_ret = float(exit_price / entry_price - 1.0)

                # IJH return for same window
                ijh_tN = ijh_t0 + hold
                if ijh_tN >= len(ijh_prices):
                    row_result[f'ret_{hold}d'] = stock_ret
                    row_result[f'abn_{hold}d'] = np.nan
                    continue

                ijh_exit = ijh_prices[ijh_tN]
                if pd.isna(ijh_exit) or ijh_exit <= 0:
                    row_result[f'ret_{hold}d'] = stock_ret
                    row_result[f'abn_{hold}d'] = np.nan
                    continue

                mkt_ret = float(ijh_exit / ijh_entry - 1.0)
                abn_ret = stock_ret - mkt_ret

                row_result[f'ret_{hold}d'] = stock_ret
                row_result[f'abn_{hold}d'] = abn_ret

            results.append(row_result)

    return pd.DataFrame(results)


def analyze_group(df, hold, label, col_prefix='abn'):
    """Analyze a group of events for a given hold period."""
    col = f'{col_prefix}_{hold}d'
    r = df[col].dropna()
    if len(r) == 0:
        print(f"    {label}: N=0")
        return
    n = len(r)
    wins = r[r > 0]
    losses = r[r <= 0]
    wr = len(wins) / n * 100
    avg = r.mean() * 100
    med = r.median() * 100
    from scipy import stats
    t_stat, p_val = stats.ttest_1samp(r, 0)
    sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''
    print(f"    {label:<30} N={n:>5}  Win={wr:>5.1f}%  Avg={avg:>+6.2f}%  "
          f"Med={med:>+6.2f}%  t={t_stat:>+5.2f} p={p_val:.4f} {sig}")


def main():
    print("=" * 100)
    print("ANALYST REVISION MOMENTUM PROBE")
    print("=" * 100)

    print("\n  Loading grades data ...")
    grades = load_all_grades()
    print(f"  Total grade actions: {len(grades):,}")

    print(f"\n  Revision type breakdown:")
    for rt, cnt in grades['revision_type'].value_counts().items():
        print(f"    {rt}: {cnt:,}")

    # Filter to revision events only (upgrades + downgrades)
    revisions = grades[grades['revision_type'].isin(['upgrade', 'downgrade'])].copy()
    # Filter to 2015+ (our reliable data window)
    revisions = revisions[revisions['date'] >= '2015-01-01']
    print(f"\n  Revision events (2015+): {len(revisions):,}")
    print(f"    Upgrades:   {(revisions['revision_type']=='upgrade').sum():,}")
    print(f"    Downgrades: {(revisions['revision_type']=='downgrade').sum():,}")

    print(f"\n  Computing returns for {len(revisions):,} events ...")
    results = compute_returns_for_events(revisions, DB, HOLD_PERIODS)
    print(f"  Events with price data: {len(results):,}")

    upgrades = results[results['revision_type'] == 'upgrade']
    downgrades = results[results['revision_type'] == 'downgrade']

    # ===== 1. MAIN RESULTS: ABNORMAL RETURNS BY HOLD PERIOD =====
    print(f"\n{'='*100}")
    print("1. ABNORMAL RETURNS (market-adjusted vs IJH) BY HOLD PERIOD")
    print(f"{'='*100}")

    for hold in HOLD_PERIODS:
        print(f"\n  Hold period: {hold} trading days ({hold/5:.0f} weeks)")
        analyze_group(upgrades, hold, "UPGRADES", 'abn')
        analyze_group(downgrades, hold, "DOWNGRADES", 'abn')

    # ===== 2. RAW RETURNS (for reference) =====
    print(f"\n{'='*100}")
    print("2. RAW RETURNS (not market-adjusted) BY HOLD PERIOD")
    print(f"{'='*100}")

    for hold in HOLD_PERIODS:
        print(f"\n  Hold period: {hold} trading days ({hold/5:.0f} weeks)")
        analyze_group(upgrades, hold, "UPGRADES", 'ret')
        analyze_group(downgrades, hold, "DOWNGRADES", 'ret')

    # ===== 3. YEAR-BY-YEAR (best hold period) =====
    # Find the best hold period from section 1
    print(f"\n{'='*100}")
    print("3. YEAR-BY-YEAR ABNORMAL RETURNS (21-day hold = 1 month)")
    print(f"{'='*100}")

    print(f"\n  {'Year':>6}  {'UPGRADES':>40}  {'DOWNGRADES':>40}")
    print(f"  {'':>6}  {'N':>5} {'Win%':>6} {'Avg':>8} {'Med':>8}  {'N':>5} {'Win%':>6} {'Avg':>8} {'Med':>8}")
    print("  " + "-" * 95)
    for yr in sorted(results['year'].unique()):
        up_yr = upgrades[(upgrades['year'] == yr) & upgrades['abn_21d'].notna()]
        dn_yr = downgrades[(downgrades['year'] == yr) & downgrades['abn_21d'].notna()]
        up_r = up_yr['abn_21d']
        dn_r = dn_yr['abn_21d']

        up_str = f"{len(up_r):>5} {(up_r>0).mean()*100:>5.0f}% {up_r.mean()*100:>+7.2f}% {up_r.median()*100:>+7.2f}%" if len(up_r) > 0 else f"{'':>5} {'':>6} {'':>8} {'':>8}"
        dn_str = f"{len(dn_r):>5} {(dn_r>0).mean()*100:>5.0f}% {dn_r.mean()*100:>+7.2f}% {dn_r.median()*100:>+7.2f}%" if len(dn_r) > 0 else f"{'':>5} {'':>6} {'':>8} {'':>8}"
        print(f"  {int(yr):>6}  {up_str}  {dn_str}")

    # Pre vs post 2020
    print()
    for label, sub in [("Pre-2020 upgrades", upgrades[(upgrades['year'] < 2020) & upgrades['abn_21d'].notna()]),
                        ("Post-2020 upgrades", upgrades[(upgrades['year'] >= 2020) & upgrades['abn_21d'].notna()]),
                        ("Pre-2020 downgrades", downgrades[(downgrades['year'] < 2020) & downgrades['abn_21d'].notna()]),
                        ("Post-2020 downgrades", downgrades[(downgrades['year'] >= 2020) & downgrades['abn_21d'].notna()])]:
        r = sub['abn_21d']
        if len(r) == 0:
            continue
        print(f"  {label:<25} N={len(r):>4}  Win={(r>0).mean()*100:>5.0f}%  Avg={r.mean()*100:>+6.2f}%  Med={r.median()*100:>+6.2f}%")

    # ===== 4. CLUSTER UPGRADES (multiple upgrades in 30-day window) =====
    print(f"\n{'='*100}")
    print("4. CLUSTER UPGRADES (2+ upgrades within 30 days)")
    print(f"{'='*100}")

    # For each upgrade, count how many other upgrades happened in the prior 30 days for same ticker
    upgrades_sorted = upgrades.sort_values(['permaTicker', 'date']).copy()
    upgrade_counts = []
    for pt, grp in upgrades_sorted.groupby('permaTicker'):
        dates = grp['date'].sort_values().values
        for i, d in enumerate(dates):
            # Count upgrades in past 30 days (including this one)
            window = dates[(dates >= d - np.timedelta64(30, 'D')) & (dates <= d)]
            upgrade_counts.append((grp.iloc[i].name, len(window)))

    count_df = pd.DataFrame(upgrade_counts, columns=['orig_idx', 'cluster_count']).set_index('orig_idx')
    upgrades_clustered = upgrades_sorted.join(count_df)

    for cluster_n in [1, 2, 3]:
        sub = upgrades_clustered[(upgrades_clustered['cluster_count'] == cluster_n) & upgrades_clustered['abn_21d'].notna()]
        r = sub['abn_21d']
        if len(r) == 0:
            continue
        wr = (r > 0).mean() * 100
        from scipy import stats
        t_stat, p_val = stats.ttest_1samp(r, 0)
        sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''
        label = f"{'Single':>12}" if cluster_n == 1 else f"{cluster_n} in 30d"
        print(f"    {label:<20} N={len(r):>4}  Win={wr:>5.1f}%  Avg={r.mean()*100:>+6.2f}%  Med={r.median()*100:>+6.2f}%  t={t_stat:>+5.2f} p={p_val:.4f} {sig}")

    # ===== 5. MAGNITUDE OF ORDINAL CHANGE =====
    print(f"\n{'='*100}")
    print("5. UPGRADE MAGNITUDE (ordinal change 1 vs 2+)")
    print(f"{'='*100}")

    for mag in [1, 2]:
        sub = upgrades[(upgrades['ordinal_change'] == mag) & upgrades['abn_21d'].notna()]
        r = sub['abn_21d']
        if len(r) == 0:
            continue
        wr = (r > 0).mean() * 100
        from scipy import stats
        t_stat, p_val = stats.ttest_1samp(r, 0)
        sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''
        label = f"+{mag} ordinal"
        print(f"    {label:<20} N={len(r):>4}  Win={wr:>5.1f}%  Avg={r.mean()*100:>+6.2f}%  Med={r.median()*100:>+6.2f}%  t={t_stat:>+5.2f} p={p_val:.4f} {sig}")

    # ===== 6. TIMING COMPLEMENTARITY =====
    print(f"\n{'='*100}")
    print("6. TIMING: Which months do analyst revisions happen?")
    print(f"{'='*100}")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    print(f"\n  {'Month':<6} {'Upgrades':>9} {'Downgrades':>11}  PEAD overlap")
    print("  " + "-" * 60)
    for m in range(1, 13):
        n_up = int((revisions['date'].dt.month == m) & (revisions['revision_type'] == 'upgrade')).sum() if False else int(((revisions['date'].dt.month == m) & (revisions['revision_type'] == 'upgrade')).sum())
        n_dn = int(((revisions['date'].dt.month == m) & (revisions['revision_type'] == 'downgrade')).sum())
        if n_up + n_dn == 0:
            continue
        overlap = "Earnings season" if m in [1, 2, 4, 5, 7, 8, 10, 11] else "Shoulder (idle)"
        print(f"  {month_names[m-1]:<6} {n_up:>9} {n_dn:>11}  {overlap}")

    # ===== 7. LONG/SHORT COMBINED =====
    print(f"\n{'='*100}")
    print("7. LONG-SHORT STRATEGY (buy upgrades, short downgrades)")
    print(f"{'='*100}")

    for hold in HOLD_PERIODS:
        up_r = upgrades[f'abn_{hold}d'].dropna()
        dn_r = downgrades[f'abn_{hold}d'].dropna()
        if len(up_r) == 0 or len(dn_r) == 0:
            continue
        # Long-short = upgrade return + |downgrade return| (short profit)
        ls_avg = up_r.mean() + abs(dn_r.mean())
        ls_total = up_r.sum() + abs(dn_r.sum())
        n_total = len(up_r) + len(dn_r)
        print(f"    Hold {hold:>2}d ({hold/5:.0f}wk):  Long-short avg={ls_avg*100:>+5.2f}%  "
              f"total={ls_total*100:>+7.1f}%  N={n_total}")

    # ===== VERDICT =====
    print(f"\n{'='*100}")
    print("VERDICT")
    print(f"{'='*100}")

    up_21 = upgrades['abn_21d'].dropna()
    dn_21 = downgrades['abn_21d'].dropna()
    up_21_post = upgrades[(upgrades['year'] >= 2020)]['abn_21d'].dropna()
    dn_21_post = downgrades[(downgrades['year'] >= 2020)]['abn_21d'].dropna()

    print(f"\n  21-day (1 month) abnormal returns:")
    print(f"    All upgrades:     N={len(up_21):>4}, Avg={up_21.mean()*100:+.2f}%, Win={(up_21>0).mean()*100:.0f}%")
    print(f"    Post-2020 upgr:   N={len(up_21_post):>4}, Avg={up_21_post.mean()*100:+.2f}%, Win={(up_21_post>0).mean()*100:.0f}%")
    print(f"    All downgrades:   N={len(dn_21):>4}, Avg={dn_21.mean()*100:+.2f}%, Win={(dn_21>0).mean()*100:.0f}%")
    print(f"    Post-2020 downgr: N={len(dn_21_post):>4}, Avg={dn_21_post.mean()*100:+.2f}%, Win={(dn_21_post>0).mean()*100:.0f}%")

    if up_21_post.mean() > 0.01 and len(up_21_post) > 50:
        print(f"\n  => UPGRADE EDGE IS ALIVE post-2020. Consider building a strategy.")
    elif up_21_post.mean() > 0:
        print(f"\n  => UPGRADE EDGE IS WEAK post-2020. Small positive but may not cover costs.")
    else:
        print(f"\n  => UPGRADE EDGE IS DEAD post-2020.")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()
