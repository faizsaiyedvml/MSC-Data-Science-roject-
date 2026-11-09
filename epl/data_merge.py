"""
Stage 02: data cleansing
"""
import json
import re

import numpy as np
import pandas as pd
from .elo import compute_elo

UNDERSTAT_NAMES = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
    "West Bromwich Albion": "West Brom",
    "Queens Park Rangers": "QPR",
}

FD_COLS = {
    "HomeTeam": "home", "AwayTeam": "away", "FTHG": "fthg", "FTAG": "ftag",
    "FTR": "ftr", "HS": "hs", "AS": "as", "HST": "hst", "AST": "ast",
    "HC": "hc", "AC": "ac", "HF": "hf", "AF": "af",
    "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar",
}
STAT_COLS = ["hs", "as", "hst", "ast", "hc", "ac", "hf", "af", "hy", "ay", "hr", "ar"]


def _norm(name):
    return re.sub(r"[^a-z]", "", str(name).lower())


def map_names(names, known, explicit):
    lookup = {_norm(k): k for k in known}
    out, missing = {}, []
    for n in names:
        if n in explicit:
            out[n] = explicit[n]
        elif n in known:
            out[n] = n
        elif _norm(n) in lookup:
            out[n] = lookup[_norm(n)]
        else:
            missing.append(n)
    return out, missing


def clean_football_data(fd):
    df = fd.rename(columns=FD_COLS)
    df = df[df["home"].notna() & df["away"].notna()].copy()
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.normalize()
    keep = ["season", "date", "home", "away", "fthg", "ftag", "ftr"] + STAT_COLS
    return df[keep]


def merge_sources(fd, us, fd_burnin=None, date_tol=3, elo_k=20.0, elo_home_adv=60.0,
                  elo_start=1500.0):
    qa = {"rows_football_data": int(len(fd))}
    df = clean_football_data(fd)
    # Elo runs over every result, burn-in included, before any rows are screened out
    res_cols = ["season", "date", "home", "away", "fthg", "ftag"]
    hist = df[res_cols]
    if fd_burnin is not None and len(fd_burnin):
        hist = pd.concat([clean_football_data(fd_burnin)[res_cols], hist], ignore_index=True)
    hist = hist.dropna(subset=["fthg", "ftag"])
    elo = compute_elo(hist, k=elo_k, home_adv=elo_home_adv, start=elo_start)
    qa["elo_seasons"] = sorted(int(x) for x in elo["season"].unique())
    teams = set(df["home"]) | set(df["away"])

    us_names = set(us["home_us"]) | set(us["away_us"])
    us_map, us_missing = map_names(us_names, teams, UNDERSTAT_NAMES)
    us = us.copy()
    us["home"] = us["home_us"].map(us_map)
    us["away"] = us["away_us"].map(us_map)
    qa["understat_unmapped"] = sorted(us_missing)


    m = df.merge(us[["season", "home", "away", "date", "h_xg", "a_xg", "us_hg", "us_ag"]]
                 .rename(columns={"date": "us_date"}),
                 on=["season", "home", "away"], how="left")
    gap = (m["date"] - m["us_date"]).abs().dt.days
    qa["date_mismatch_over_tol"] = m.loc[gap > date_tol, ["season", "home", "away"]] \
        .astype(str).values.tolist()
    score_diff = m["us_hg"].notna() & ((m["us_hg"] != m["fthg"]) | (m["us_ag"] != m["ftag"]))
    qa["score_mismatch_understat"] = int(score_diff.sum())
    m = m.drop(columns=["us_date", "us_hg", "us_ag"])

    m = m.merge(elo[["season", "date", "home", "away", "h_elo", "a_elo"]],
                on=["season", "date", "home", "away"], how="left")
    m, drops = quality_screen(m)
    qa["dropped"] = drops
    qa["rows_final"] = int(len(m))
    return m, qa


def quality_screen(m):
    """Drop unusable rows and record why each one went."""
    drops = []
    reasons = {
        "missing_result": m[["fthg", "ftag", "ftr"]].isna().any(axis=1),
        "missing_stats": m[STAT_COLS].isna().any(axis=1),
        "missing_xg": m[["h_xg", "a_xg"]].isna().any(axis=1),
        "missing_elo": m[["h_elo", "a_elo"]].isna().any(axis=1),
        "impossible_values": (m["hst"] > m["hs"]) | (m["ast"] > m["as"])
                             | (m[STAT_COLS + ["fthg", "ftag"]] < 0).any(axis=1),
    }
    bad = pd.Series(False, index=m.index)
    for why, mask in reasons.items():
        new = mask & ~bad
        for _, r in m[new].iterrows():
            drops.append({"season": int(r["season"]), "home": r["home"], "away": r["away"],
                          "reason": why})
        bad |= mask
    m = m[~bad].copy()
    m = m.sort_values(["date", "home"]).reset_index(drop=True)
    m["match_id"] = np.arange(len(m))
    return m, drops


def save_qa(qa, path):
    with open(path, "w") as f:
        json.dump(qa, f, indent=2, default=str)
