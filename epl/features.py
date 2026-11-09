"""
stage 03: pre-match features.
"""
import numpy as np
import pandas as pd

# base name -> (column in team view, description)
CONVENTIONAL = {
    "ppg":  ("pts",   "Points per game"),
    "gf":   ("gf",    "Goals scored per game"),
    "ga":   ("ga",    "Goals conceded per game"),
    "sf":   ("sf",    "Shots per game"),
    "sa":   ("sa",    "Shots conceded per game"),
    "sotf": ("sotf",  "Shots on target per game"),
    "sota": ("sota",  "Shots on target conceded per game"),
    "cf":   ("cf",    "Corners won per game"),
    "ca":   ("ca",    "Corners conceded per game"),
    "fouls": ("fouls", "Fouls committed per game"),
    "yc":   ("yel",   "Yellow cards per game"),
    "rc":   ("red",   "Red cards per game"),
}
FORM = {
    "form_pts": ("pts", "Mean points, last 5"),
    "form_gd":  ("gd",  "Mean goal difference, last 5"),
    "form_gf":  ("gf",  "Mean goals scored, last 5"),
    "form_ga":  ("ga",  "Mean goals conceded, last 5"),
    "form_win": ("win", "Win rate, last 5"),
}
XG_STD = {
    "xg_for":     ("xgf", "xG for per game, season to date"),
    "xg_against": ("xga", "xG against per game, season to date"),
}
XG_LAST5 = {
    "xg_for5":     ("xgf", "xG for per game, last 5"),
    "xg_against5": ("xga", "xG against per game, last 5"),
}

CATEGORY = {**{k: "conventional" for k in CONVENTIONAL}, **{k: "form" for k in FORM},
            **{k: "xg" for k in list(XG_STD) + list(XG_LAST5)}, "elo": "elo"}
BASE_FEATURES = list(CONVENTIONAL) + list(FORM) + list(XG_STD) + list(XG_LAST5) + ["elo"]
CATEGORIES = ["conventional", "elo", "xg", "form"]


def feature_category(col):
    return CATEGORY[col[2:]]


def team_view(m):
    """two rows per match, one from each team's side."""
    def side(home):
        p, o = ("h", "a") if home else ("a", "h")
        gf = m["fthg"] if home else m["ftag"]
        ga = m["ftag"] if home else m["fthg"]
        t = pd.DataFrame({
            "match_id": m["match_id"], "season": m["season"], "date": m["date"],
            "team": m["home" if home else "away"], "is_home": int(home),
            "gf": gf, "ga": ga,
            "sf": m[f"{p}s"], "sa": m[f"{o}s"],
            "sotf": m[f"{p}st"], "sota": m[f"{o}st"],
            "cf": m[f"{p}c"], "ca": m[f"{o}c"],
            "fouls": m[f"{p}f"], "yel": m[f"{p}y"], "red": m[f"{p}r"],
            "xgf": m[f"{p}_xg"], "xga": m[f"{o}_xg"], "elo": m[f"{p}_elo"],
        })
        t["pts"] = np.select([gf > ga, gf == ga], [3, 1], 0)
        t["win"] = (gf > ga).astype(int)
        t["gd"] = gf - ga
        return t

    tv = pd.concat([side(True), side(False)], ignore_index=True)
    return tv.sort_values(["date", "match_id", "is_home"], ascending=[True, True, False]) \
             .reset_index(drop=True)


def relegated_teams(tv):
    """bottom three of each season's table (points, then goal difference)."""
    tab = tv.groupby(["season", "team"]).agg(pts=("pts", "sum"), gd=("gd", "sum")).reset_index()
    out = {}
    for s, g in tab.groupby("season"):
        out[s] = g.sort_values(["pts", "gd"]).head(3)["team"].tolist()
    return out


def _spells(tv):
    seasons = tv[["team", "season"]].drop_duplicates().sort_values(["team", "season"])
    seasons["spell"] = seasons.groupby("team")["season"].diff().ne(1).groupby(seasons["team"]).cumsum()
    return tv.merge(seasons, on=["team", "season"], how="left")


def build_team_features(tv):
    tv = _spells(tv)
    out = tv[["match_id", "season", "date", "team", "is_home", "spell"]].copy()
    out["elo"] = tv["elo"]

    # season-to-date means of earlier matches this season
    g = tv.groupby(["team", "season"])
    n_prev = g.cumcount()
    for name, (col, _) in {**CONVENTIONAL, **XG_STD}.items():
        prev_sum = g[col].cumsum() - tv[col]
        out[name] = (prev_sum / n_prev).where(n_prev > 0)

    # last-5 means within the current EPL spell
    gs = tv.groupby(["team", "spell"])
    for name, (col, _) in {**FORM, **XG_LAST5}.items():
        out[name] = gs[col].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    out = _cold_start(out, tv)
    return out


def _cold_start(out, tv):
    """fill the first match of a season (season-to-date features) and the
    first match of a spell (last-5 features)

    established team: its own previous-season mean
    Promoted team: mean of the previous season's relegated teams
    """
    rel = relegated_teams(tv)
    std_cols = list(CONVENTIONAL) + list(XG_STD)
    roll_cols = list(FORM) + list(XG_LAST5)

    std_src = {**CONVENTIONAL, **XG_STD}
    season_mean = tv.groupby(["season", "team"])[[c for c, _ in std_src.values()]].mean()

    # last-5 values at the end of each team's season (for promoted-team fills)
    roll_src = {**FORM, **XG_LAST5}
    end_form = {}
    for (team, season), g in tv.groupby(["team", "season"]):
        tail = g.tail(5)
        end_form[(season, team)] = {n: tail[c].mean() for n, (c, _) in roll_src.items()}
    end_form = pd.DataFrame.from_dict(end_form, orient="index")

    for (team, season), idx in out.groupby(["team", "season"]).groups.items():
        first = idx[0]
        prev = season - 1
        if prev not in rel:
            continue
        if (prev, team) in season_mean.index:
            vals = season_mean.loc[(prev, team)]
        else:
            vals = season_mean.loc[[(prev, t) for t in rel[prev]]].mean()
        for name in std_cols:
            if pd.isna(out.at[first, name]):
                out.at[first, name] = vals[std_src[name][0]]
        if out.loc[first, roll_cols].isna().any():
            fill = end_form.loc[[(prev, t) for t in rel[prev]]].mean()
            for name in roll_cols:
                if pd.isna(out.at[first, name]):
                    out.at[first, name] = fill[name]
    return out


def build_dataset(merged, study_seasons):
    """match-level dataset, team-level feature table, data dictionary"""
    tv = team_view(merged)
    tf = build_team_features(tv)

    h = tf[tf["is_home"] == 1].set_index("match_id")[BASE_FEATURES].add_prefix("h_")
    a = tf[tf["is_home"] == 0].set_index("match_id")[BASE_FEATURES].add_prefix("a_")
    base = merged.set_index("match_id")[["season", "date", "home", "away", "fthg", "ftag", "ftr"]]
    ds = base.join(h).join(a).reset_index()
    ds["y"] = ds["ftr"].map({"H": 0, "D": 1, "A": 2})
    ds = ds[ds["season"].isin(study_seasons)]
    feats = feature_columns()
    ds = ds.dropna(subset=feats).sort_values(["date", "match_id"]).reset_index(drop=True)
    return ds, tf, data_dictionary()


def feature_columns():
    return [f"{p}_{b}" for p in ("h", "a") for b in BASE_FEATURES]


def data_dictionary():
    desc = {k: v[1] for d in (CONVENTIONAL, FORM, XG_STD, XG_LAST5) for k, v in d.items()}
    desc["elo"] = "Elo rating before kick-off, calculated from EPL results"
    window = {**{k: "season to date" for k in list(CONVENTIONAL) + list(XG_STD)},
              **{k: "last 5 matches" for k in list(FORM) + list(XG_LAST5)},
              "elo": "before match"}
    rows = []
    for p, side in (("h", "home"), ("a", "away")):
        for b in BASE_FEATURES:
            rows.append({"feature": f"{p}_{b}", "base": b, "side": side,
                         "category": CATEGORY[b], "window": window[b],
                         "description": f"{desc[b]} ({side} team)"})
    return pd.DataFrame(rows)
