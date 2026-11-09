
import numpy as np
import pandas as pd


def round_robin(teams, rng):
    t = list(teams)
    rng.shuffle(t)
    n = len(t)
    rounds = []
    for r in range(n - 1):
        pairs = [(t[i], t[n - 1 - i]) for i in range(n // 2)]
        rounds.append([(a, b) if (r + i) % 2 == 0 else (b, a) for i, (a, b) in enumerate(pairs)])
        t = [t[0]] + [t[-1]] + t[1:-1]
    second = [[(b, a) for a, b in rnd] for rnd in rounds]
    return rounds + second


def make_league(first=2014, last=2025, n_teams=20, pool=32, seed=42):
    rng = np.random.default_rng(seed)
    names = [f"Team {chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(pool)]
    elo = {t: rng.normal(1650, 90) for t in names}
    league = names[:n_teams]
    others = names[n_teams:]
    rows = []
    for s in range(first, last + 1):
        start = pd.Timestamp(f"{s}-08-10")
        for r, rnd in enumerate(round_robin(league, rng)):
            day = start + pd.Timedelta(days=7 * r)
            for h, a in rnd:
                dh, da = elo[h], elo[a]
                diff = (dh + 60 - da) / 400
                lam_h, lam_a = np.exp(0.3 + 0.55 * diff), np.exp(0.1 - 0.55 * diff)
                xh = lam_h * rng.gamma(8, 1 / 8)
                xa = lam_a * rng.gamma(8, 1 / 8)
                gh, ga = rng.poisson(xh), rng.poisson(xa)
                hs = rng.poisson(9 + 5 * lam_h)
                as_ = rng.poisson(9 + 5 * lam_a)
                rows.append({
                    "season": s, "date": day, "home": h, "away": a,
                    "fthg": gh, "ftag": ga, "ftr": "H" if gh > ga else ("A" if ga > gh else "D"),
                    "hs": hs, "as": as_,
                    "hst": rng.binomial(hs, 0.35), "ast": rng.binomial(as_, 0.35),
                    "hc": rng.poisson(5 + lam_h), "ac": rng.poisson(5 + lam_a),
                    "hf": rng.poisson(11), "af": rng.poisson(11.5),
                    "hy": rng.poisson(1.6), "ay": rng.poisson(1.8),
                    "hr": rng.binomial(1, 0.04), "ar": rng.binomial(1, 0.05),
                    "h_xg": round(xh, 2), "a_xg": round(xa, 2),
                    "h_elo": round(dh, 1), "a_elo": round(da, 1),
                })
                exp_h = 1 / (1 + 10 ** (-(dh + 60 - da) / 400))
                res = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
                k = 20 * (1 + abs(gh - ga)) ** 0.5
                elo[h] += k * (res - exp_h)
                elo[a] -= k * (res - exp_h)
        df = pd.DataFrame([x for x in rows if x["season"] == s])
        pts = {}
        for _, x in df.iterrows():
            ph = 3 if x.fthg > x.ftag else (1 if x.fthg == x.ftag else 0)
            pa = 3 if x.ftag > x.fthg else (1 if x.fthg == x.ftag else 0)
            pts[x.home] = pts.get(x.home, 0) + ph
            pts[x.away] = pts.get(x.away, 0) + pa
        down = sorted(league, key=lambda t: pts[t])[:3]
        up = sorted(others, key=lambda t: -elo[t])[:3]
        league = [t for t in league if t not in down] + up
        others = [t for t in others if t not in up] + down
        for t in up:
            elo[t] -= 40
    m = pd.DataFrame(rows).sort_values(["date", "home"]).reset_index(drop=True)
    m["match_id"] = np.arange(len(m))
    return m
