#!/usr/bin/env python3
"""
Daily signal generator — consumes the validated 26-year methodology.
Outputs signals.json consumed by index.html.

Enriched output:
    tickers[t].recent_signals  : last 12 trades for this ticker (date, side, Z,
                                  price_at_signal, MA50_at_signal, sigma60_at_signal)
    global past_signals         : flat list of last 80 trades across the universe
    near_signals                : tickers with 0.70 <= |Z| < 1.00 (close but not triggered)

IMPORTANT : the `price` field is the OHLC close on the trigger day, NOT today's price.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timezone
import warnings, json

warnings.filterwarnings("ignore")

UNIVERSE = {
    "ETF": ["SPY","QQQ","IWM","DIA","VTI","EFA","EEM","XLF","XLK","XLE","XLV","XLY",
            "XLP","XLU","XLB","XLI","TLT","IEF","GLD","SLV","USO"],
    "S&P": ["NVDA","AAPL","MSFT","AMZN","GOOGL","GOOG","AVGO","TSLA","META","WMT",
            "BRK-B","LLY","MU","JPM","AMD","INTC","V","XOM","ORCL","JNJ",
            "COST","MA","CAT","CSCO","NFLX","LRCX","BAC","CVX","ABBV","AMAT",
            "UNH","PG","KO","PLTR","HD"],
    "FUT": ["CL=F","NG=F","GC=F","SI=F","HG=F","ZW=F","ZC=F","ZS=F",
            "RB=F","HO=F","KC=F","CC=F","ES=F"],
}
MA_WIN, SIG_WIN    = 50, 60
Z_THRESH           = 1.00
NEAR_LO, NEAR_HI   = 0.70, 1.00   # proximity band
HORIZON = 270
DELTA   = 0.10
RATE    = 0.045
ANTI_OVERLAP = 60
MAX_PAST = 80
MAX_RECENT_PER_TICKER = 12

def bs_strike(S, T, sigma, side):
    if sigma <= 0 or np.isnan(sigma): return None
    z = norm.ppf(1 - DELTA) if side == "call" else norm.ppf(DELTA)
    return float(S * np.exp(-0.5 * sigma**2 * T + z * sigma * np.sqrt(T)))

def signal_side(z):
    if z >  Z_THRESH: return "CALL"
    if z < -Z_THRESH: return "PUT"
    return None

def load_history(tkr):
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df = yf.download(tkr, start="2000-01-01", end=end,
                     auto_adjust=True, progress=False)
    if df is None or len(df) < 600:
        return None, 0
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    if "high" not in df.columns: df["high"] = df["close"]
    if "low"  not in df.columns: df["low"]  = df["close"]
    rets = df["close"].pct_change()
    df = df.loc[rets.abs() <= 0.30].copy()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["sigma60"] = df["log_ret"].rolling(SIG_WIN).std() * np.sqrt(252)
    df["MA50"]    = df["close"].rolling(MA_WIN).mean()
    df = df.iloc[MA_WIN + SIG_WIN:].copy()
    df["sigma_n"] = df["sigma60"] * np.sqrt(MA_WIN / 252)
    df["Z"]       = np.log(df["close"] / df["MA50"]) / df["sigma_n"]
    df["signal"]  = df["Z"].apply(signal_side)
    return df, round(len(df) / 252.0, 1)

def collect_signals(df):
    """Return [(date, side, Z, close_at_signal, MA50_at_signal, sigma60_at_signal)]
    with 60-day anti-overlap per side. `close_at_signal` is the close on the trigger
    day (NOT today's price)."""
    last_c = last_p = None
    out = []
    for idx, row in df.iterrows():
        s = row["signal"]
        if s == "CALL":
            if last_c is None or (idx - last_c).days >= ANTI_OVERLAP:
                last_c = idx
                out.append((idx, "CALL", float(row["Z"]),
                            float(row["close"]), float(row["MA50"]), float(row["sigma60"])))
        elif s == "PUT":
            if last_p is None or (idx - last_p).days >= ANTI_OVERLAP:
                last_p = idx
                out.append((idx, "PUT", float(row["Z"]),
                            float(row["close"]), float(row["MA50"]), float(row["sigma60"])))
    return out

def build_ticker_payload(tkr, group):
    df, years = load_history(tkr)
    if df is None or len(df) == 0:
        return None
    last = df.iloc[-1]
    S0   = float(last["close"])
    Z    = float(last["Z"])
    sig_ = float(last["sigma60"])
    MA   = float(last["MA50"])
    side = signal_side(Z)
    K    = bs_strike(S0, HORIZON/365.0, sig_, side) if side else None

    spark = df["Z"].tail(180).round(3)
    spark_z   = spark.tolist()
    spark_idx = [d.strftime("%Y-%m-%d") for d in spark.index]

    trades = collect_signals(df)
    last_trade = trades[-1] if trades else (None, None, None, None, None, None)
    recent = trades[-MAX_RECENT_PER_TICKER:] if trades else []

    near = (side is None) and (NEAR_LO <= abs(Z) < NEAR_HI)

    asof = df.index[-1]
    return {
        "ticker": tkr, "group": group,
        "asof":  asof.strftime("%Y-%m-%d"),
        "last_close":     round(S0, 2),
        "MA50":           round(MA, 2),
        "sigma60_annual": round(sig_, 4),
        "Z":              round(Z, 3),
        "signal":         side,
        "strike":         round(K, 2) if K else None,
        "distance_pct":   round((K / S0 - 1) * 100, 1) if K else None,
        "near_signal":    bool(near),
        "last_signal_date":       last_trade[0].strftime("%Y-%m-%d") if last_trade[0] is not None else None,
        "last_signal_side":       last_trade[1] if last_trade[1] else None,
        "last_signal_price":      round(last_trade[3], 4) if last_trade[3] is not None else None,
        "n_trades_total":  len(trades),
        "recent_signals": [
            {"date": d.strftime("%Y-%m-%d"),
             "side": s,
             "Z": round(z, 3),
             "price": round(p, 4),                # close au jour du signal
             "MA50": round(ma, 4),
             "sigma60_annual": round(sig, 4)}
            for (d, s, z, p, ma, sig) in recent
        ],
        "spark_z":   spark_z,
        "spark_dates": spark_idx,
        "years_covered": years,
    }

def main():
    out = {
        "generated_at":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema_version": 3,
        "config": {
            "MA": MA_WIN, "sigma_window": SIG_WIN,
            "Z_threshold": Z_THRESH,
            "near_band":   [NEAR_LO, NEAR_HI],
            "horizon_days": HORIZON, "delta": DELTA,
            "rate": RATE, "anti_overlap_days": ANTI_OVERLAP,
            "universe_size": sum(len(v) for v in UNIVERSE.values()),
        },
        "backtest_summary": {
            "period":                    "2000-01-01 → 2026-07-08",
            "tickers_tested":            69,
            "win_rate_put_only_pct":     91.4,
            "win_rate_mixed_pct":        75.8,
            "trades_per_year_put_only":  49.3,
            "trades_per_year_mixed":    115.4,
            "source":            "yfinance OHLCV",
            "method":            "MA=50 + |Z_log|>1σ, T=270d, Δ=0.10, r=4.5%, anti-overlap 60j",
            "near_band_explained": f"Tickers with {NEAR_LO}σ ≤ |Z| < {NEAR_HI}σ today (close but not triggered)",
        },
        "tickers": {},
        "past_signals": [],
        "near_signals": [],
    }
    failed = []
    for group, lst in UNIVERSE.items():
        for tkr in lst:
            try:
                p = build_ticker_payload(tkr, group)
                if p is not None:
                    out["tickers"][tkr] = p
            except Exception as e:
                failed.append({"ticker": tkr, "reason": str(e)})
    out["failed_tickers"] = failed

    # Flatten the latest MAX_PAST trades across the universe, with each
    # entry enriched by the OHLC close ON THE TRIGGER DATE (not today).
    flat = []
    for t, v in out["tickers"].items():
        for tr in v["recent_signals"]:
            flat.append({"ticker": t, "group": v["group"],
                         "date": tr["date"], "side": tr["side"],
                         "Z": tr["Z"],
                         "price": tr["price"],                # << historical price at signal
                         "MA50": tr["MA50"],
                         "sigma60_annual": tr["sigma60_annual"]})
    flat.sort(key=lambda r: r["date"], reverse=True)
    out["past_signals"] = flat[:MAX_PAST]

    near = []
    for t, v in out["tickers"].items():
        if v["near_signal"]:
            near.append({"ticker": t, "group": v["group"],
                         "asof": v["asof"], "last_close": v["last_close"],
                         "MA50": v["MA50"], "Z": v["Z"], "sigma60_annual": v["sigma60_annual"]})
    near.sort(key=lambda r: -abs(r["Z"]))
    out["near_signals"] = near

    with open("/home/user/dashboard/signals.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))

    print(f"OK  {len(out['tickers'])} tickers ({len(failed)} failed)")
    sigs = sum(1 for v in out["tickers"].values() if v["signal"])
    print(f"Active signals today     : {sigs}")
    print(f"Near-signal markets      : {len(out['near_signals'])}  (|Z| in [{NEAR_LO}, {NEAR_HI}))")
    print(f"Past signals stored      : {len(out['past_signals'])}")
    print(f"Past with 'price' field  : {sum(1 for r in out['past_signals'] if r.get('price') is not None)}")
    return out

if __name__ == "__main__":
    main()
