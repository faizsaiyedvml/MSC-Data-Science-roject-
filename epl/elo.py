"""elo ratings built from match results.

api.clubelo.com returned 502 on every request when tested in Sept 2026, so the
ratings are now calculated here from football-data.co.uk results instead.

- each match stores both teams' ratings *before* kick-off, so a match never
  sees its own result
- home advantage is added to the home rating when working out the expected
  score. K is scaled by goal difference (goal-based Elo, Hvattum & Arntzen 2010),
  same as the synthetic league
- ratings start at ELO_START in the first burn-in season. A promoted team
  starts its season on the mean end-of-season rating of the teams that went
  down the year before (same idea as the cold-start rule in features.py).
- the burn-in seasons are only there to let the ratings settle. They are not
  used for anything else
- rows without a full-time score (blank lines, unplayed fixtures) are dropped,
  otherwise one NaN would spread through every later rating
"""
import numpy as np
import pandas as pd


def expected_home(h, a, home_adv):
    return 1.0 / (1.0 + 10 ** (-(h + home_adv - a) / 400.0))


def compute_elo(results, k=20.0, home_adv=60.0, start=1500.0):
    """results: one row per match with season, date, home, away, fthg, ftag.
    returns the played rows with h_elo / a_elo (pre-match) added."""
    m = results.dropna(subset=["fthg", "ftag"]).copy()
    if not pd.api.types.is_datetime64_any_dtype(m["date"]):
        raise TypeError("date must be parsed to datetime before compute_elo "
                        "(football-data dates are dd/mm/yy strings)")

    m = m.sort_values(["date", "home"]).reset_index(drop=True)
    season_order = m.groupby("season")["date"].min().sort_values().index
    if list(season_order) != sorted(season_order):
        raise ValueError("season labels don't sort in date order: "
                         f"{list(season_order)}")

    ratings = {}
    prev_teams = set()
    h_elo = np.empty(len(m))
    a_elo = np.empty(len(m))

    for season, g in m.groupby("season", sort=True):
        teams = set(g["home"]) | set(g["away"])
        if prev_teams:
            down = prev_teams - teams
            entry = np.mean([ratings[t] for t in down]) if down else start
            for t in teams - prev_teams:
                ratings[t] = entry
        else:
            for t in teams:
                ratings[t] = start

        for i, r in g.iterrows():
            h, a = ratings[r["home"]], ratings[r["away"]]
            h_elo[i], a_elo[i] = h, a
            gh, ga = r["fthg"], r["ftag"]
            res = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
            kk = k * (1 + abs(gh - ga)) ** 0.5
            shift = kk * (res - expected_home(h, a, home_adv))
            ratings[r["home"]] = h + shift
            ratings[r["away"]] = a - shift
        prev_teams = teams

    m["h_elo"] = h_elo.round(1)
    m["a_elo"] = a_elo.round(1)
    return m