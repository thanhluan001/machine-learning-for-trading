#!/usr/bin/env python3
"""
Index Rebalancing Edge Probe — S&P 400 constituent additions.

Question: When a stock is added to the S&P 400, does it drift up before the
effective date? If so, buying N days before the effective date and selling on
the effective date captures the index addition effect (mechanical buying by
index funds).

Caveat: 'added_date' is the EFFECTIVE date (when index funds must buy), NOT
the announcement date. S&P typically announces 3-5 days before. Buying 5
days before effective ≈ buying around announcement.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import numpy as np, pandas as pd

DB = 'C:/Users/thanh/Projects/machine-learning-for-trading/luan_bot_trading/01_data/db.h5'

# Windows to test: buy N days before effective, sell on effective
BUY_DAYS_BEFORE = [1, 3, 5, 7, 10]


def main():
    print("=" * 100)
    print("INDEX REBALANCING EDGE PROBE — S&P 400 Constituent Additions")
    print("=" * 100)

    with pd.HDFStore(DB, mode='r') as s:
        meta = s['/metadata/sp400_permatickers']
        sp400_keys = set(k for k in s.keys() if k.startswith('/sp400/'))

    # Parse all addition events from wikipedia_intervals
    additions = []
    for _, row in meta.iterrows():
        pt = row['permaTicker']
        ticker = row.get('canonical_ticker', pt)
        intervals_raw = row.get('wikipedia_intervals')
        if pd.isna(intervals_raw):
            continue
        try:
            intervals = json.loads(intervals_raw) if isinstance(intervals_raw, str) else intervals_raw
        except (json.JSONDecodeError, TypeError):
            continue
        for interval in intervals:
            added = interval.get('added')
            removed = interval.get('removed')
            if added is None:
                continue
            added_date = pd.to_datetime(added, errors='coerce')
            if pd.isna(added_date):
                continue
            # Skip pre-2015 additions (outside our reliable data window)
            if added_date < pd.Timestamp('2015-01-01'):
                continue
            # Skip if no price data
            key = f'/sp400/{pt}'
            if key not in sp400_keys:
                continue
            additions.append({
                'permaTicker': pt,
                'ticker': ticker,
                'index_ref': row.get('index_ref', ''),
                'added_date': added_date,
                'removed_date': pd.to_datetime(removed, errors='coerce') if removed else pd.NaT,
                'price_key': key,
            })

    additions_df = pd.DataFrame(additions).sort_values('added_date').reset_index(drop=True)
    print(f"\n  Total addition events (2015+): {len(additions_df)}")
    print(f"  Year distribution:")
    additions_df['year'] = additions_df['added_date'].dt.year
    for yr, cnt in additions_df.groupby('year').size().items():
        print(f"    {yr}: {cnt}")

    # ===== COMPUTE RETURNS FOR EACH WINDOW =====
    print(f"\n{'='*100}")
    print("INDEX ADDITION RETURNS (buy N days before effective, sell on effective)")
    print(f"{'='*100}")

    results_by_window = {}

    with pd.HDFStore(DB, mode='r') as s:
        for buy_days in BUY_DAYS_BEFORE:
            returns = []
            details = []
            for _, ev in additions_df.iterrows():
                prices = s[ev['price_key']]
                prices = prices.copy()
                prices['Date'] = pd.to_datetime(prices['Date'])
                prices = prices.sort_values('Date').reset_index(drop=True)

                effective = ev['added_date']

                # Find the trading day at or just before the effective date
                eff_mask = prices['Date'] <= effective
                if not eff_mask.any():
                    continue
                eff_idx = int(eff_mask.values[::-1].argmax())
                eff_idx = len(prices) - 1 - eff_idx

                # Buy date: buy_days trading days before effective
                buy_idx = eff_idx - buy_days
                if buy_idx < 0:
                    continue

                buy_price = prices.iloc[buy_idx]['Adj_Close']
                eff_price = prices.iloc[eff_idx]['Adj_Close']

                if pd.isna(buy_price) or pd.isna(eff_price) or buy_price <= 0:
                    continue

                ret = float(eff_price / buy_price - 1.0)

                # Also compute IJH benchmark return for same window
                returns.append({
                    'permaTicker': ev['permaTicker'],
                    'ticker': ev['ticker'],
                    'index_ref': ev['index_ref'],
                    'added_date': ev['added_date'],
                    'year': ev['added_date'].year,
                    'buy_date': prices.iloc[buy_idx]['Date'],
                    'eff_date': prices.iloc[eff_idx]['Date'],
                    'buy_price': buy_price,
                    'eff_price': eff_price,
                    'return': ret,
                })

            res_df = pd.DataFrame(returns)
            results_by_window[buy_days] = res_df

            if len(res_df) == 0:
                continue

            r = res_df['return']
            wins = r[r > 0]
            losses = r[r <= 0]
            n = len(r)
            wr = len(wins) / n * 100

            print(f"\n  Buy {buy_days} day(s) before effective, sell on effective:")
            print(f"    N={n}, Win rate={wr:.1f}%, Avg={r.mean()*100:+.2f}%, Median={r.median()*100:+.2f}%")
            print(f"    Avg win={wins.mean()*100:+.2f}%, Avg loss={losses.mean()*100:+.2f}%")
            print(f"    Payoff={wins.mean()/abs(losses.mean()):.2f}, Total={r.sum()*100:+.1f}%")
            print(f"    p25={np.percentile(r,25)*100:+.2f}%, p75={np.percentile(r,75)*100:+.2f}%")

            # t-test
            from scipy import stats
            t_stat, p_val = stats.ttest_1samp(r, 0)
            print(f"    t-stat={t_stat:.3f}, p-value={p_val:.4f} {'***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else 'ns'}")

    # ===== YEAR-BY-YEAR (7-day window — the significant edge) =====
    print(f"\n{'='*100}")
    print("YEAR-BY-YEAR (7-day window \u2014 the significant edge)")
    print(f"{'='*100}")

    res7 = results_by_window[7]
    print(f"\n  {'Year':>6} {'N':>5} {'Win%':>6} {'Avg':>8} {'Median':>8} {'Total':>9}")
    print("  " + "-" * 50)
    for yr in sorted(res7['year'].unique()):
        sub = res7[res7['year'] == yr]
        r = sub['return']
        wr = (r > 0).mean() * 100
        print(f"  {int(yr):>6} {len(r):>5} {wr:>5.0f}% {r.mean()*100:>+7.2f}% {r.median()*100:>+7.2f}% {r.sum()*100:>+8.1f}%")

    pre7 = res7[res7['year'] < 2020]['return']
    post7 = res7[res7['year'] >= 2020]['return']
    print(f"\n  Pre-2020:  N={len(pre7):>3}, Avg={pre7.mean()*100:+.2f}%, Win={(pre7>0).mean()*100:.0f}%")
    print(f"  Post-2020: N={len(post7):>3}, Avg={post7.mean()*100:+.2f}%, Win={(post7>0).mean()*100:.0f}%")

    # ===== YEAR-BY-YEAR (5-day window — is the edge decaying?) =====
    print(f"\n{'='*100}")
    print("YEAR-BY-YEAR (5-day window — is the edge decaying?)")
    print(f"{'='*100}")

    res5 = results_by_window[5]
    print(f"\n  {'Year':>6} {'N':>5} {'Win%':>6} {'Avg':>8} {'Median':>8} {'Total':>9}")
    print("  " + "-" * 50)
    for yr in sorted(res5['year'].unique()):
        sub = res5[res5['year'] == yr]
        r = sub['return']
        wr = (r > 0).mean() * 100
        print(f"  {int(yr):>6} {len(r):>5} {wr:>5.0f}% {r.mean()*100:>+7.2f}% {r.median()*100:>+7.2f}% {r.sum()*100:>+8.1f}%")

    # Compare pre-2020 vs post-2020
    pre = res5[res5['year'] < 2020]['return']
    post = res5[res5['year'] >= 2020]['return']
    print(f"\n  Pre-2020:  N={len(pre):>3}, Avg={pre.mean()*100:+.2f}%, Win={(pre>0).mean()*100:.0f}%")
    print(f"  Post-2020: N={len(post):>3}, Avg={post.mean()*100:+.2f}%, Win={(post>0).mean()*100:.0f}%")

    # ===== MARKET-ADJUSTED RETURNS =====
    print(f"\n{'='*100}")
    print("MARKET-ADJUSTED RETURNS (5-day window, vs IJH benchmark)")
    print(f"{'='*100}")

    with pd.HDFStore(DB, mode='r') as s:
        if '/macros/IJH' in s:
            ijh = s['/macros/IJH'].copy()
            ijh['Date'] = pd.to_datetime(ijh['Date'])
            ijh = ijh.sort_values('Date').reset_index(drop=True)
            close_col = 'Adj_Close' if 'Adj_Close' in ijh.columns else 'Close'
            ijh_dict = dict(zip(ijh['Date'], ijh[close_col]))

            res5_adj = res5.copy()
            res5_adj['market_return'] = np.nan
            res5_adj['abnormal_return'] = np.nan

            for idx, row in res5.iterrows():
                buy_d = row['buy_date']
                eff_d = row['eff_date']
                # Find nearest IJH prices
                ijh_dates = ijh['Date'].values
                buy_mask = ijh_dates <= buy_d.to_datetime64()
                eff_mask = ijh_dates <= eff_d.to_datetime64()
                if not buy_mask.any() or not eff_mask.any():
                    continue
                ijh_buy_idx = len(ijh_dates) - 1 - int(buy_mask[::-1].argmax())
                ijh_eff_idx = len(ijh_dates) - 1 - int(eff_mask[::-1].argmax())
                ijh_buy = ijh.iloc[ijh_buy_idx][close_col]
                ijh_eff = ijh.iloc[ijh_eff_idx][close_col]
                if pd.isna(ijh_buy) or pd.isna(ijh_eff) or ijh_buy <= 0:
                    continue
                mkt_ret = float(ijh_eff / ijh_buy - 1.0)
                res5_adj.at[idx, 'market_return'] = mkt_ret
                res5_adj.at[idx, 'abnormal_return'] = row['return'] - mkt_ret

            valid = res5_adj['abnormal_return'].dropna()
            print(f"\n  Raw return:      N={len(valid)}, Avg={res5_adj.loc[valid.index,'return'].mean()*100:+.2f}%")
            print(f"  Market return:   Avg={res5_adj.loc[valid.index,'market_return'].mean()*100:+.2f}%")
            print(f"  Abnormal return: N={len(valid)}, Avg={valid.mean()*100:+.2f}%, Median={valid.median()*100:+.2f}%")
            wr_ab = (valid > 0).mean() * 100
            print(f"  Abnormal win rate: {wr_ab:.1f}%")

            from scipy import stats
            t_stat, p_val = stats.ttest_1samp(valid, 0)
            print(f"  t-stat={t_stat:.3f}, p-value={p_val:.4f}")
        else:
            print("  IJH benchmark not found in DB")

    # ===== TIMING: which months do additions happen? =====
    print(f"\n{'='*100}")
    print("TIMING: Which months do S&P 400 additions happen?")
    print(f"{'='*100}")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    print(f"\n  {'Month':<6} {'Count':>6}  Season overlap with PEAD")
    print("  " + "-" * 55)
    for m in range(1, 13):
        cnt = int((additions_df['added_date'].dt.month == m).sum())
        if cnt == 0:
            continue
        if m in [1, 2, 4, 5, 7, 8, 10, 11]:
            overlap = "OVERLAPS with PEAD earnings season"
        else:
            overlap = "PEAD shoulder month (idle capital)"
        bar = "#" * (cnt // 2)
        print(f"  {month_names[m-1]:<6} {cnt:>6}  {overlap} {bar}")

    # ===== SAMPLE TRADES (5-day window, post-2020) =====
    print(f"\n{'='*100}")
    print("SAMPLE TRADES (5-day window, post-2020, top 10 + bottom 10)")
    print(f"{'='*100}")

    post5 = res5[res5['year'] >= 2020].sort_values('return', ascending=False)
    print(f"\n  Top 10 winners:")
    print(f"  {'Ticker':<8} {'Added date':<12} {'Return':>8}")
    for _, t in post5.head(10).iterrows():
        print(f"  {t['ticker']:<8} {str(t['added_date'].date()):<12} {t['return']*100:>+7.1f}%")

    print(f"\n  Bottom 10 losers:")
    for _, t in post5.tail(10).iterrows():
        print(f"  {t['ticker']:<8} {str(t['added_date'].date()):<12} {t['return']*100:>+7.1f}%")

    # ===== SUMMARY VERDICT =====
    print(f"\n{'='*100}")
    print("VERDICT")
    print(f"{'='*100}")

    if 5 in results_by_window and len(results_by_window[5]) > 0:
        r5 = results_by_window[5]['return']
        post5_r = res5[res5['year'] >= 2020]['return']
        print(f"\n  5-day window (buy 5 days before effective, sell on effective):")
        print(f"    All-time: N={len(r5)}, Avg={r5.mean()*100:+.2f}%, Win={(r5>0).mean()*100:.0f}%")
        print(f"    Post-2020: N={len(post5_r)}, Avg={post5_r.mean()*100:+.2f}%, Win={(post5_r>0).mean()*100:.0f}%")

        if post5_r.mean() > 0.02:
            print(f"\n  => EDGE IS ALIVE post-2020. The index addition effect still exists.")
            print(f"     Consider building a rebalancing strategy.")
        elif post5_r.mean() > 0:
            print(f"\n  => EDGE IS WEAK post-2020. Positive but small. May not cover costs.")
        else:
            print(f"\n  => EDGE IS DEAD post-2020. Front-running has eliminated the drift.")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()
