"""W2c: digitize Gao et al. (2025) Fig.3 left panel — China national CW generation 2000-2022.

Approach: render page 14 at 400 dpi, isolate the left bar-chart panel, detect the
y-axis spine and its tick marks (0 .. 2.5e8 in units of 10^4 t), detect lavender bar
columns, map bar-top pixels to data units by linear calibration from tick pixels.
Outputs data/china/gao2025_national_series.csv with per-year estimates (10^4 t).

Digitization error is documented in the file header (we estimate +-5%); in the
observation model this source carries a wide sd_m anyway.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "source_pdfs" / "Gao2025_grey_china.pdf"
OUT = ROOT / "data" / "china"

doc = pymupdf.open(str(PDF))
page = doc[13]  # page 14 (0-based)
DPI = 400
pix = page.get_pixmap(dpi=DPI)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
if pix.n == 4:
    img = img[:, :, :3]
H, W, _ = img.shape
print("page px:", W, H)

# left panel generous crop (fractions of page, from 200-dpi inspection)
x0, x1 = int(0.085 * W), int(0.35 * W)
y0, y1 = int(0.20 * H), int(0.375 * H)
crop = img[y0:y1, x0:x1]
ch, cw, _ = crop.shape
print("crop px:", cw, ch)

# --- detect bars: lavender ~ R>G, B>R, moderately saturated
R, G, B = crop[:, :, 0].astype(int), crop[:, :, 1].astype(int), crop[:, :, 2].astype(int)
bar_mask = (B > R + 20) & (np.abs(R - G) < 25) & (B > 140) & (R > 100)
# exclude legend box (upper-left inside plot area): measured extent x[95,465] y[80,200]
bar_mask[80:200, 95:465] = False
# restrict to band that excludes legend (top of plot) but keeps shortest bars
BASELINE_Y = 691
band = bar_mask[150:BASELINE_Y + 1, :]
colsig = band.sum(axis=0).astype(float)
# bars are 23 evenly spaced purple columns; use peak detection with pitch prior
from scipy.signal import find_peaks
peaks, props = find_peaks(colsig, height=20, distance=22, prominence=10)
print(f"peaks detected: {len(peaks)} at {peaks.tolist()}")
if len(peaks) != 23:
    # fall back: enforce regular grid between first and last strong column
    strong = np.where(colsig > 20)[0]
    lo, hi = strong.min(), strong.max()
    peaks = np.round(np.linspace(lo, hi, 23)).astype(int)
    print(f"fallback regular grid: {peaks.tolist()}")
groups = [(int(p), int(p)) for p in peaks]
print(f"bars detected: {len(groups)}")

tops = []
for (a, b) in groups:
    mid = (a + b) // 2
    col = bar_mask[:, mid]
    ys = np.where(col)[0]
    tops.append(ys.min() if len(ys) else np.nan)

# --- y calibration: find left spine and ticks (dark pixels)
dark = (R < 90) & (G < 90) & (B < 90)
# spine = leftmost column with many dark pixels in lower 2/3 of crop
colsum = dark[int(ch * 0.1):, :].sum(axis=0)
spine_x = int(np.argmax(colsum[: int(cw * 0.35)]))
print("spine_x:", spine_x, "dark count:", colsum[spine_x])
# ticks: dark pixels immediately LEFT of spine
tick_zone = dark[:, max(0, spine_x - 12): spine_x - 2]
rowsum = tick_zone.sum(axis=1)
tick_rows = np.where(rowsum >= 4)[0]
# cluster tick rows
ticks = []
if len(tick_rows):
    s = tick_rows[0]; p = tick_rows[0]
    for r in tick_rows[1:]:
        if r - p > 3:
            ticks.append((s + p) / 2); s = r
        p = r
    ticks.append((s + p) / 2)
print("tick rows:", [round(t, 1) for t in ticks])

# y-axis tick labels read "2.5x10^5 ... 5.0x10^4, 0.0" in units of 10^4 t
# (verified against text: 2023 forecast 2.39e9 t = 239000 x10^4t)
if len(ticks) >= 5:
    tick_vals = np.array([2.5e5, 2.0e5, 1.5e5, 1.0e5, 0.5e5][:len(ticks)]
                         if len(ticks) <= 5
                         else np.linspace(2.5e5, 0.0, len(ticks)))[:len(ticks)]
    # fit value(10^4 t) = a + b * y_pixel  (y grows downward)
    A = np.vstack([np.ones(len(ticks)), np.array(ticks)]).T
    a_, b_ = np.linalg.lstsq(A, tick_vals, rcond=None)[0]
    y_at_0 = -a_ / b_
    print(f"calibration: value(1e4 t) = {a_:.1f} + {b_:.2f}*y ; baseline y0={y_at_0:.1f}")
    vals = a_ + b_ * np.array(tops)  # already in 10^4 t
else:
    raise RuntimeError("tick detection failed; need manual calibration")

years = np.arange(2000, 2000 + len(vals))
df = pd.DataFrame({"year": years, "cw_generation_1e4t": np.round(vals, 0)})
df.to_csv(OUT / "gao2025_national_series_digitized.csv", index=False, encoding="utf-8-sig")
print(df.to_string(index=False))
# sanity anchors: 2022 ~ 22.8e8 t = 228000 (1e4t); text forecast 2023 = 239000
v2022 = df.loc[df["year"] == 2022, "cw_generation_1e4t"]
print("\nsanity: 2022 =", v2022.tolist(), "(expect ~225000-235000)")
