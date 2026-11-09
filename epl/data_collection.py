"""
data collection

Sources
football-data.co.uk and Understat data available
Elo ratings had to be calculated from the results in elo.py (api.clubelo.com
returned 502 on every request when tested in Oct 2026), so extra football-data seasons are downloaded as an Elo burn-in.
"""
import json
import time

import pandas as pd
import requests

FD_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
US_URL = "https://understat.com/getLeagueData/EPL/{season}"
US_REFERER = "https://understat.com/league/EPL/{season}"

HEADERS = {
    "User-Agent": ("mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/javascript, */*; q=0.01",
}


def _get(url, headers=None, tries=3, pause=2.0):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=headers or HEADERS, timeout=30)
            if r.status_code == 200 and r.content:
                return r
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(pause * (i + 1))
    raise RuntimeError(f"could not download {url}: {last}")


def season_code(season):
    """2015 -> '1516'"""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


# ---------------------------------------------------------------- football-data
def get_football_data(season, raw_dir):
    f = raw_dir / f"fd_E0_{season}.csv"
    if not f.exists():
        r = _get(FD_URL.format(code=season_code(season)))
        f.write_bytes(r.content)
    return pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")


# ---------------------------------------------------------------- Understat
def parse_understat(payload):
    """turn the getLeagueData JSON into one row per played match."""
    if isinstance(payload, (str, bytes)):
        payload = json.loads(payload)
    games = payload.get("dates", payload) if isinstance(payload, dict) else payload
    rows = []
    for g in games:
        if str(g.get("isResult", "")).lower() not in ("true", "1"):
            continue
        rows.append({
            "us_id": int(g["id"]),
            "date": pd.to_datetime(g["datetime"]).normalize(),
            "home_us": g["h"]["title"],
            "away_us": g["a"]["title"],
            "us_hg": int(g["goals"]["h"]),
            "us_ag": int(g["goals"]["a"]),
            "h_xg": float(g["xG"]["h"]),
            "a_xg": float(g["xG"]["a"]),
        })
    return pd.DataFrame(rows)


def get_understat(season, raw_dir):
    f = raw_dir / f"understat_{season}.json"
    if not f.exists():
        h = dict(HEADERS)
        h["X-Requested-With"] = "XMLHttpRequest"
        h["Referer"] = US_REFERER.format(season=season)
        r = _get(US_URL.format(season=season), headers=h)
        json.loads(r.text)   # fail here rather than caching an HTML error page
        f.write_text(r.text, encoding="utf-8")
    return parse_understat(f.read_text(encoding="utf-8"))


def collect_all(seasons, burnin_seasons, raw_dir, log=print):
    """seasons: warm-up + study seasons (all three sources).
    burnin_seasons: earlier seasons, football-data only, used to settle Elo."""
    fd, us, burn = [], [], []
    for s in seasons:
        log(f"  season {s}/{(s + 1) % 100:02d}")
        d = get_football_data(s, raw_dir)
        d["season"] = s
        fd.append(d)
        u = get_understat(s, raw_dir)
        u["season"] = s
        us.append(u)
    for s in burnin_seasons:
        log(f"  Elo burn-in {s}/{(s + 1) % 100:02d} (results only)")
        d = get_football_data(s, raw_dir)
        d["season"] = s
        burn.append(d)
    fd = pd.concat(fd, ignore_index=True)
    us = pd.concat(us, ignore_index=True)
    burn = pd.concat(burn, ignore_index=True) if burn else fd.iloc[0:0]
    fd.to_csv(raw_dir / "fd_all.csv", index=False)
    us.to_csv(raw_dir / "understat_all.csv", index=False)
    burn.to_csv(raw_dir / "fd_elo_burnin.csv", index=False)
    return fd, us, burn
