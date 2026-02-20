import os
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ccxt


# =========================
# Model (same as training)
# =========================
class CNNCls(nn.Module):
    def __init__(self, n_features, hidden=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x.transpose(1, 2)           # (B,F,T)
        z = self.net(x)
        z = self.pool(z).squeeze(-1)    # (B,H)
        return self.head(z).squeeze(-1) # logits


def load_classifier(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_features = int(ckpt["n_features"])
    hidden = int(ckpt.get("cfg", {}).get("hidden", 192))

    model = CNNCls(n_features=n_features, hidden=hidden).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    mean = torch.tensor(ckpt["mean"], dtype=torch.float32, device=device)
    std  = torch.tensor(ckpt["std"], dtype=torch.float32, device=device)
    return model, mean, std


# =========================
# Feature engineering (ctx + volume)
# =========================
BASE_FEATURES = ["open","high","low","close","ret1","range_n","body_n","upper_wick_n","lower_wick_n"]

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def slope(series: pd.Series, k: int = 10) -> pd.Series:
    return (series - series.shift(k)) / (k + 1e-12)

def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    o = out["open"].to_numpy(np.float64)
    h = out["high"].to_numpy(np.float64)
    l = out["low"].to_numpy(np.float64)
    c = out["close"].to_numpy(np.float64)
    eps = 1e-12

    ret1 = np.zeros(len(out), dtype=np.float64)
    ret1[1:] = (c[1:] - c[:-1]) / (np.abs(c[:-1]) + eps)

    out["ret1"] = ret1
    out["range_n"] = (h - l) / (np.abs(c) + eps)
    out["body_n"] = (c - o) / (np.abs(c) + eps)
    out["upper_wick_n"] = (h - np.maximum(o, c)) / (np.abs(c) + eps)
    out["lower_wick_n"] = (np.minimum(o, c) - l) / (np.abs(c) + eps)
    return out

def build_context_features(df_1m: pd.DataFrame):
    d = add_base_features(df_1m)

    d["ema20"] = ema(d["close"], 20)
    d["ema50"] = ema(d["close"], 50)
    d["atr14"] = atr(d, 14)
    d["atrp"] = d["atr14"] / (d["close"].abs() + 1e-12)
    d["rsi14"] = rsi(d["close"], 14)

    d["dist_ema20"] = (d["close"] - d["ema20"]) / (d["close"].abs() + 1e-12)
    d["dist_ema50"] = (d["close"] - d["ema50"]) / (d["close"].abs() + 1e-12)
    d["ema20_slope"] = slope(d["ema20"], 10) / (d["close"].abs() + 1e-12)
    d["disp_body_atr"] = (d["close"] - d["open"]).abs() / (d["atr14"] + 1e-12)

    # volume barrier helper
    if "volume" in d.columns:
        d["vol_sma50"] = d["volume"].rolling(50).mean()
        d["vol_mult50"] = d["volume"] / (d["vol_sma50"] + 1e-12)
    else:
        d["vol_mult50"] = 1.0

    # HTF context (5m / 15m)
    dd = d.set_index("time")[["open","high","low","close"]].copy()

    def htf(tf: str, prefix: str):
        agg = dd.resample(tf).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
        agg[f"{prefix}_ema20"] = ema(agg["close"], 20)
        agg[f"{prefix}_ema50"] = ema(agg["close"], 50)
        agg[f"{prefix}_ema20_slope"] = slope(agg[f"{prefix}_ema20"], 5) / (agg["close"].abs() + 1e-12)
        agg[f"{prefix}_dist_ema20"] = (agg["close"] - agg[f"{prefix}_ema20"]) / (agg["close"].abs() + 1e-12)
        agg[f"{prefix}_dist_ema50"] = (agg["close"] - agg[f"{prefix}_ema50"]) / (agg["close"].abs() + 1e-12)
        return agg[[f"{prefix}_ema20_slope", f"{prefix}_dist_ema20", f"{prefix}_dist_ema50"]].reindex(dd.index, method="ffill")

    h5 = htf("5min", "m5")
    h15 = htf("15min", "m15")

    d = d.set_index("time").join(h5, how="left").join(h15, how="left").reset_index()
    for c in ["m5_ema20_slope","m5_dist_ema20","m5_dist_ema50","m15_ema20_slope","m15_dist_ema20","m15_dist_ema50"]:
        d[c] = d[c].fillna(0.0)

    ctx_cols = [
        "ema20","ema50","atr14","atrp","rsi14",
        "dist_ema20","dist_ema50","ema20_slope","disp_body_atr",
        "m5_ema20_slope","m5_dist_ema20","m5_dist_ema50",
        "m15_ema20_slope","m15_dist_ema20","m15_dist_ema50",
        "vol_mult50",
    ]
    feat_cols = BASE_FEATURES + ctx_cols
    return d, feat_cols


# =========================
# Imbalance scanning
# =========================
@dataclass
class Zone:
    zone_id: str
    side: str            # LONG / SHORT
    z_low: float
    z_high: float
    formed_i: int
    formed_time: str

    gap: float
    gap_atr: float
    gap_pct: float
    vol_mult: float

    entry: float
    sl: float
    tp: float
    rr: float

    prob_tp: float = 0.0
    status: str = "ARMED"           # ARMED -> FILLED -> TP/SL/EXPIRED
    fill_i: Optional[int] = None
    fill_time: Optional[str] = None
    exit_i: Optional[int] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None


def make_zone_id(side: str, formed_ts: pd.Timestamp, z_low: float, z_high: float) -> str:
    return f"{side}:{formed_ts.value}:{round(z_low,2)}:{round(z_high,2)}"


def is_touched(df: pd.DataFrame, z_low: float, z_high: float, start_i: int, end_i: int) -> bool:
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    start_i = max(0, start_i)
    end_i = min(len(df)-1, end_i)
    for k in range(start_i, end_i + 1):
        # пересечение диапазона зоны
        if l[k] <= z_high and h[k] >= z_low:
            return True
    return False


def pct_move(a: float, b: float) -> float:
    if abs(a) < 1e-12:
        return 0.0
    return abs(b - a) / abs(a)


def build_levels(side: str, z_low: float, z_high: float, atr_i: float, rr: float, sl_pad_atr: float):
    entry = (z_low + z_high) / 2.0
    pad = sl_pad_atr * atr_i
    if side == "LONG":
        sl = z_low - pad
        tp = entry + rr * (entry - sl)
    else:
        sl = z_high + pad
        tp = entry - rr * (sl - entry)
    return float(entry), float(sl), float(tp)


def find_unworked_zone_stepwise(
    df: pd.DataFrame,
    last_i: int,
    lookback_bars: int,
    max_scan_bars: int,
    min_gap_atr: float,
    max_gap_atr: float,
    min_gap_pct: float,
    max_gap_pct: float,
    min_vol_mult: float,
    log_every: int = 25,
) -> Tuple[Optional[dict], dict]:
    """
    Посвечной поиск (от новой к старой), пока не найдём "неотработанную" зону.
    """
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    atr14 = df["atr14"].to_numpy(np.float64)
    volm = df["vol_mult50"].to_numpy(np.float64)

    start_i = max(2, last_i - lookback_bars)
    scanned = 0

    for i in range(last_i, start_i - 1, -1):
        scanned += 1
        if scanned > max_scan_bars:
            break

        if log_every > 0 and scanned % log_every == 0:
            age = last_i - i
            print(f"[SCAN] {scanned}/{max_scan_bars} | i={i} | age={age} bars назад")

        atr_i = float(atr14[i]) if np.isfinite(atr14[i]) else 0.0
        if atr_i <= 0:
            continue

        vol_mult = float(volm[i]) if np.isfinite(volm[i]) else 1.0
        if vol_mult < min_vol_mult:
            continue

        # -------- LONG FVG (bullish) --------
        if l[i] > h[i-2]:
            z_low = float(h[i-2])
            z_high = float(l[i])
            gap = z_high - z_low
            gap_atr = gap / atr_i
            mid = (z_low + z_high) / 2.0
            gap_pct = gap / (abs(mid) + 1e-12)

            if not (min_gap_atr <= gap_atr <= max_gap_atr):
                continue
            if not (min_gap_pct <= gap_pct <= max_gap_pct):
                continue

            # unworked: NOT touched after forming
            if not is_touched(df, z_low, z_high, start_i=i+1, end_i=last_i):
                age = last_i - i
                print(f"[FOUND] LONG FVG i={i} age={age} | gap_atr={gap_atr:.3f} | gap_pct={gap_pct*100:.3f}% | vol_mult={vol_mult:.2f}")
                return {
                    "side": "LONG",
                    "z_low": z_low,
                    "z_high": z_high,
                    "i": i,
                    "atr": atr_i,
                    "gap": gap,
                    "gap_atr": gap_atr,
                    "gap_pct": gap_pct,
                    "vol_mult": vol_mult,
                }, {"scanned": scanned, "start_i": start_i}

        # -------- SHORT FVG (bearish) --------
        if h[i] < l[i-2]:
            z_low = float(h[i])
            z_high = float(l[i-2])
            gap = z_high - z_low
            gap_atr = gap / atr_i
            mid = (z_low + z_high) / 2.0
            gap_pct = gap / (abs(mid) + 1e-12)

            if not (min_gap_atr <= gap_atr <= max_gap_atr):
                continue
            if not (min_gap_pct <= gap_pct <= max_gap_pct):
                continue

            if not is_touched(df, z_low, z_high, start_i=i+1, end_i=last_i):
                age = last_i - i
                print(f"[FOUND] SHORT FVG i={i} age={age} | gap_atr={gap_atr:.3f} | gap_pct={gap_pct*100:.3f}% | vol_mult={vol_mult:.2f}")
                return {
                    "side": "SHORT",
                    "z_low": z_low,
                    "z_high": z_high,
                    "i": i,
                    "atr": atr_i,
                    "gap": gap,
                    "gap_atr": gap_atr,
                    "gap_pct": gap_pct,
                    "vol_mult": vol_mult,
                }, {"scanned": scanned, "start_i": start_i}

    return None, {"scanned": scanned, "start_i": start_i}


# =========================
# Exchange
# =========================
def fetch_ohlcv_bybit(exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)


# =========================
# State
# =========================
def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"zones": [], "active_zone_id": None}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"zones": [], "active_zone_id": None}

def save_state(path: str, state: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# =========================
# Live loop
# =========================
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="imbalance_classifier_ctx.pt")
    ap.add_argument("--symbol", default="ETH/USDT:USDT")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--window", type=int, default=240)

    # scan behavior
    ap.add_argument("--lookback_bars", type=int, default=800)
    ap.add_argument("--max_scan_bars", type=int, default=150)
    ap.add_argument("--log_every", type=int, default=25, help="печать прогресса каждые N свечей (0=off)")

    # imbalance size barriers
    ap.add_argument("--min_gap_atr", type=float, default=0.15)
    ap.add_argument("--max_gap_atr", type=float, default=2.00)
    ap.add_argument("--min_gap_pct", type=float, default=0.0017, help="0.001=0.1%")
    ap.add_argument("--max_gap_pct", type=float, default=0.0200, help="0.02=2%")

    # volume barrier
    ap.add_argument("--min_vol_mult", type=float, default=1.0, help="объём свечи формирования >= SMA50*mult")

    # levels
    ap.add_argument("--rr", type=float, default=1.5)
    ap.add_argument("--sl_pad_atr", type=float, default=0.10)
    ap.add_argument("--min_move_pct", type=float, default=0.002, help="минимум 0.2% до TP/SL")

    # model filter
    ap.add_argument("--threshold", type=float, default=0.70)

    # demo management
    ap.add_argument("--fill_timeout_bars", type=int, default=240)
    ap.add_argument("--trade_timeout_bars", type=int, default=600)

    ap.add_argument("--poll_sec", type=int, default=15)
    ap.add_argument("--state_file", default="live_state_stepwise.json")
    ap.add_argument("--reset_state", action="store_true", help="сбросить state и начать заново")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, mean, std = load_classifier(args.model, device)

    exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})

    state = load_state(args.state_file)
    if args.reset_state:
        state = {"zones": [], "active_zone_id": None}
        save_state(args.state_file, state)
        print("State reset:", args.state_file)

    known = set(z["zone_id"] for z in state.get("zones", []))

    print("Live scanner started.")
    print("symbol:", args.symbol, "| tf:", args.timeframe, "| poll_sec:", args.poll_sec)
    print("scan:", "lookback_bars=", args.lookback_bars, "max_scan_bars=", args.max_scan_bars, "log_every=", args.log_every)
    print("gap:", f"atr {args.min_gap_atr}..{args.max_gap_atr} | pct {args.min_gap_pct}..{args.max_gap_pct}")
    print("vol:", "min_vol_mult=", args.min_vol_mult, "| threshold:", args.threshold)
    print("state:", args.state_file)

    while True:
        try:
            raw = fetch_ohlcv_bybit(exchange, args.symbol, args.timeframe, args.limit)
            if len(raw) < args.window + 100:
                print("Not enough candles:", len(raw))
                time.sleep(args.poll_sec)
                continue

            df, feat_cols = build_context_features(raw)
            last_i = len(df) - 1
            last_time = df.loc[last_i, "time"]

            # resolve active zone
            active = None
            active_id = state.get("active_zone_id")
            if active_id:
                for z in state.get("zones", []):
                    if z.get("zone_id") == active_id and z.get("status") in ("ARMED", "FILLED"):
                        active = Zone(**z)
                        break

            # =========================
            # 1) No active => find unworked imbalance (stepwise scan)
            # =========================
            if active is None:
                found, info = find_unworked_zone_stepwise(
                    df=df,
                    last_i=last_i,
                    lookback_bars=args.lookback_bars,
                    max_scan_bars=args.max_scan_bars,
                    min_gap_atr=args.min_gap_atr,
                    max_gap_atr=args.max_gap_atr,
                    min_gap_pct=args.min_gap_pct,
                    max_gap_pct=args.max_gap_pct,
                    min_vol_mult=args.min_vol_mult,
                    log_every=args.log_every,
                )

                print(f"{last_time.isoformat()} | scan_depth: scanned={info['scanned']} (max {args.max_scan_bars}), "
                      f"available_back={last_i - info['start_i']} bars")

                if found is None:
                    time.sleep(args.poll_sec)
                    continue

                formed_i = int(found["i"])
                formed_ts = df.loc[formed_i, "time"]
                zid = make_zone_id(found["side"], formed_ts, found["z_low"], found["z_high"])
                age = last_i - formed_i

                if zid in known:
                    print("Zone already seen:", zid, f"(age={age} bars)")
                    time.sleep(args.poll_sec)
                    continue

                entry, sl, tp = build_levels(found["side"], found["z_low"], found["z_high"], found["atr"], args.rr, args.sl_pad_atr)

                # enforce min 0.2% move to SL/TP
                if pct_move(entry, tp) < args.min_move_pct or pct_move(entry, sl) < args.min_move_pct:
                    known.add(zid)
                    z = Zone(
                        zone_id=zid, side=found["side"], z_low=found["z_low"], z_high=found["z_high"],
                        formed_i=formed_i, formed_time=formed_ts.isoformat(),
                        gap=found["gap"], gap_atr=found["gap_atr"], gap_pct=found["gap_pct"], vol_mult=found["vol_mult"],
                        entry=entry, sl=sl, tp=tp, rr=args.rr,
                        prob_tp=0.0, status="EXPIRED", exit_reason="TOO_SMALL"
                    )
                    state["zones"].append(asdict(z))
                    save_state(args.state_file, state)
                    print("Zone rejected: TOO_SMALL")
                    time.sleep(args.poll_sec)
                    continue

                # =========================
                # MODEL USED ONLY AFTER ZONE FOUND
                # =========================
                X = df.iloc[last_i - args.window:last_i][feat_cols].to_numpy(dtype=np.float32)
                Xt = torch.tensor(X, dtype=torch.float32, device=device)
                Xt = (Xt - mean) / std
                prob = torch.sigmoid(model(Xt.unsqueeze(0))).item()

                status = "ARMED" if prob >= args.threshold else "EXPIRED"
                exit_reason = None if status == "ARMED" else "LOW_PROB"

                zone = Zone(
                    zone_id=zid, side=found["side"], z_low=float(found["z_low"]), z_high=float(found["z_high"]),
                    formed_i=formed_i, formed_time=formed_ts.isoformat(),
                    gap=float(found["gap"]), gap_atr=float(found["gap_atr"]), gap_pct=float(found["gap_pct"]),
                    vol_mult=float(found["vol_mult"]),
                    entry=entry, sl=sl, tp=tp, rr=args.rr,
                    prob_tp=float(prob),
                    status=status,
                    exit_reason=exit_reason
                )

                state["zones"].append(asdict(zone))
                known.add(zid)
                if zone.status == "ARMED":
                    state["active_zone_id"] = zid
                save_state(args.state_file, state)

                print("\n=== UNWORKED IMBALANCE FOUND ===")
                print("now:", last_time.isoformat(), "| formed:", zone.formed_time, f"| age={age} bars")
                print("side:", zone.side, f"| prob_tp={zone.prob_tp:.3f} | vol_mult={zone.vol_mult:.2f}")
                print(f"gap={zone.gap:.2f} | gap_atr={zone.gap_atr:.3f} | gap_pct={zone.gap_pct*100:.3f}%")
                print(f"zone: [{zone.z_low:.2f} .. {zone.z_high:.2f}]")
                print(f"entry={zone.entry:.2f}  sl={zone.sl:.2f}  tp={zone.tp:.2f}  RR={zone.rr}")
                print("status:", zone.status)

                time.sleep(args.poll_sec)
                continue

            # =========================
            # 2) Manage active zone candle-by-candle
            # =========================
            zones = state.get("zones", [])
            idx_active = None
            for idx, z in enumerate(zones):
                if z.get("zone_id") == active.zone_id:
                    idx_active = idx
                    break
            if idx_active is None:
                state["active_zone_id"] = None
                save_state(args.state_file, state)
                time.sleep(args.poll_sec)
                continue

            hi = float(df.loc[last_i, "high"])
            lo = float(df.loc[last_i, "low"])

            if active.status == "ARMED":
                if last_i - active.formed_i > args.fill_timeout_bars:
                    zones[idx_active]["status"] = "EXPIRED"
                    zones[idx_active]["exit_reason"] = "NO_FILL_TIMEOUT"
                    zones[idx_active]["exit_i"] = int(last_i)
                    zones[idx_active]["exit_time"] = last_time.isoformat()
                    state["active_zone_id"] = None
                    save_state(args.state_file, state)
                    print("\n[EXPIRED] no fill timeout:", active.zone_id)
                else:
                    if lo <= active.entry <= hi:
                        zones[idx_active]["status"] = "FILLED"
                        zones[idx_active]["fill_i"] = int(last_i)
                        zones[idx_active]["fill_time"] = last_time.isoformat()
                        save_state(args.state_file, state)
                        print("\n[FILLED] entry touched:", f"{active.entry:.2f}", "| time:", last_time.isoformat())

            elif active.status == "FILLED":
                filled_i = int(active.fill_i) if active.fill_i is not None else last_i
                if last_i - filled_i > args.trade_timeout_bars:
                    zones[idx_active]["status"] = "EXPIRED"
                    zones[idx_active]["exit_reason"] = "TRADE_TIMEOUT"
                    zones[idx_active]["exit_i"] = int(last_i)
                    zones[idx_active]["exit_time"] = last_time.isoformat()
                    state["active_zone_id"] = None
                    save_state(args.state_file, state)
                    print("\n[EXPIRED] trade timeout:", active.zone_id)
                else:
                    if active.side == "LONG":
                        if lo <= active.sl:
                            zones[idx_active]["status"] = "SL"
                            zones[idx_active]["exit_reason"] = "SL"
                            zones[idx_active]["exit_i"] = int(last_i)
                            zones[idx_active]["exit_time"] = last_time.isoformat()
                            state["active_zone_id"] = None
                            save_state(args.state_file, state)
                            print("\n[SL HIT] time:", last_time.isoformat(), "sl:", f"{active.sl:.2f}")
                        elif hi >= active.tp:
                            zones[idx_active]["status"] = "TP"
                            zones[idx_active]["exit_reason"] = "TP"
                            zones[idx_active]["exit_i"] = int(last_i)
                            zones[idx_active]["exit_time"] = last_time.isoformat()
                            state["active_zone_id"] = None
                            save_state(args.state_file, state)
                            print("\n[TP HIT] time:", last_time.isoformat(), "tp:", f"{active.tp:.2f}")
                    else:
                        if hi >= active.sl:
                            zones[idx_active]["status"] = "SL"
                            zones[idx_active]["exit_reason"] = "SL"
                            zones[idx_active]["exit_i"] = int(last_i)
                            zones[idx_active]["exit_time"] = last_time.isoformat()
                            state["active_zone_id"] = None
                            save_state(args.state_file, state)
                            print("\n[SL HIT] time:", last_time.isoformat(), "sl:", f"{active.sl:.2f}")
                        elif lo <= active.tp:
                            zones[idx_active]["status"] = "TP"
                            zones[idx_active]["exit_reason"] = "TP"
                            zones[idx_active]["exit_i"] = int(last_i)
                            zones[idx_active]["exit_time"] = last_time.isoformat()
                            state["active_zone_id"] = None
                            save_state(args.state_file, state)
                            print("\n[TP HIT] time:", last_time.isoformat(), "tp:", f"{active.tp:.2f}")

            time.sleep(args.poll_sec)

        except Exception as e:
            print("Error:", repr(e))
            time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
