"""S1 spike: fetch Eurostat env_wasgen CDW (W121) panel and report coverage/missingness.

Slices fetched:
  - waste=W121 (Mineral waste from construction and demolition)
  - nace_r2=TOTAL_HH (all activities incl. households) and F (construction sector)
  - hazard=HAZ_NHAZ (total)
  - unit=T (tonnes)
All geo (incl. EU aggregates), all available time (biennial 2004-2022).
Saves long-format CSV per slice + a coverage report to results/.
"""
import io
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/env_wasgen"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eurostat"
RES = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

HDR = {"User-Agent": "Mozilla/5.0 (research; plan4 spike)"}


def fetch(params: str, retries: int = 4) -> dict:
    url = f"{BASE}?format=JSON&lang=EN&{params}"
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 + 2 * k)
    raise RuntimeError(f"fetch failed after {retries} tries: {last}")


def js_to_df(d: dict) -> pd.DataFrame:
    ids, sizes = d["id"], d["size"]
    vals = d.get("value", {})
    # build category code lists in flat-index order
    cats = {}
    for dim in ids:
        idx = d["dimension"][dim]["category"]["index"]
        # index may be dict code->pos
        cats[dim] = [c for c, _ in sorted(idx.items(), key=lambda kv: kv[1])]
    rows = []
    for flat, v in vals.items():
        n = int(flat)
        coords = []
        for sz in reversed(sizes):
            coords.append(n % sz)
            n //= sz
        coords = coords[::-1]
        row = {dim: cats[dim][i] for dim, i in zip(ids, coords)}
        row["value"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def main():
    frames = []
    for nace in ["TOTAL_HH", "F"]:
        d = fetch(f"waste=W121&nace_r2={nace}&hazard=HAZ_NHAZ&unit=T")
        raw = OUT / f"env_wasgen_W121_{nace}.json"
        raw.write_text(json.dumps(d), encoding="utf-8")
        df = js_to_df(d)
        df.to_csv(OUT / f"env_wasgen_W121_{nace}.csv", index=False)
        frames.append((nace, df))
        print(f"[{nace}] rows={len(df)} geos={df['geo'].nunique()} "
              f"times={sorted(df['time'].unique())}")

    # coverage report on TOTAL_HH
    df = dict(frames)["TOTAL_HH"]
    pv = df.assign(flag=1).pivot_table(index="geo", columns="time",
                                       values="flag", aggfunc="max").fillna(0).astype(int)
    pv.to_csv(RES / "s1_coverage_total_hh.csv")
    rep = []
    rep.append(f"geos total: {pv.shape[0]}, times: {list(pv.columns)}")
    full = pv[(pv.sum(axis=1) == pv.shape[1])]
    rep.append(f"geos with FULL coverage (all {pv.shape[1]} waves): {len(full)}")
    rep.append("full-coverage geos: " + ", ".join(full.index.tolist()))
    per_time = pv.sum(axis=0)
    rep.append("obs per wave:\n" + per_time.to_string())
    # value sanity
    rep.append("\nvalue stats (tonnes):\n" +
               df.groupby("time")["value"].describe().round(0).to_string())
    (RES / "s1_coverage_report.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
