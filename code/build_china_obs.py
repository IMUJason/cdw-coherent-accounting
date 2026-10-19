"""W2a: build the China CDW multi-source observation database (v0.1, spike-grade).

Transcribes the verified numbers from notes/s3_china_data_inventory.md and the two
MEE-affiliated PDFs (data/china/meescc_2024_solidwaste_report.pdf,
meescc_2025_solidwaste_report.pdf, mee_2025_cdw_standard_draft_notes.pdf)
into a machine-readable long table with caliber annotations.

Every row answers: who says how much CDW, where, when, under which caliber,
from which method family -- this is the observation-process registry the
state-space model's measurement equations are parameterized from.

Guardrails (per patch-safety discipline): values are asserted against known
consistency relations where available; the script fails loudly on violation.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "china"
OUT.mkdir(parents=True, exist_ok=True)

# columns: level, geo_cn, year, value_wt (万吨), caliber, substream,
#          includes_spoil, method_family, source_id, credibility, url, notes
R = [
    # ---------------- national: MEE-affiliated annual aggregate (same caliber)
    dict(level="national", geo_cn="全国", year=2023, value_wt=241000.0,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="yes(部分市)",
         method_family="official_aggregate", source_id="meescc_2024",
         credibility="官方统计汇总",
         url="https://www.meescc.cn/zhxx/zxyw/202505/W020250530336582088864.pdf",
         notes="基于265城统计；315城发布信息"),
    dict(level="national", geo_cn="全国", year=2023, value_wt=86000.0,
         caliber="资源化利用量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="official_aggregate", source_id="meescc_2024",
         credibility="官方统计汇总",
         url="https://www.meescc.cn/zhxx/zxyw/202505/W020250530336582088864.pdf",
         notes="基于225城"),
    dict(level="national", geo_cn="全国", year=2023, value_wt=76000.0,
         caliber="填埋量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="official_aggregate", source_id="meescc_2024",
         credibility="官方统计汇总",
         url="https://www.meescc.cn/zhxx/zxyw/202505/W020250530336582088864.pdf",
         notes="基于188城"),
    dict(level="national", geo_cn="全国", year=2023, value_wt=162000.0,
         caliber="产生量", substream="工程渣土", includes_spoil="yes",
         method_family="official_aggregate", source_id="meescc_2024",
         credibility="官方统计汇总",
         url="https://www.meescc.cn/zhxx/zxyw/202505/W020250530336582088864.pdf",
         notes="基于183城；渣土占总量的~67%（口径内）"),
    dict(level="national", geo_cn="全国", year=2023, value_wt=5001.9,
         caliber="产生量", substream="工程泥浆", includes_spoil="yes",
         method_family="official_aggregate", source_id="meescc_2024",
         credibility="官方统计汇总",
         url="https://www.meescc.cn/zhxx/zxyw/202505/W020250530336582088864.pdf",
         notes="基于109城"),
    dict(level="national", geo_cn="全国", year=2024, value_wt=231000.0,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="yes(部分市)",
         method_family="official_aggregate", source_id="meescc_2025",
         credibility="官方统计汇总",
         url="https://www.meescc.cn/zhxx/zxyw/202603/W020260317610986472151.pdf",
         notes="基于287城；317城发布"),
    dict(level="national", geo_cn="全国", year=2024, value_wt=69000.0,
         caliber="资源化利用量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="official_aggregate", source_id="meescc_2025",
         credibility="官方统计汇总",
         url="https://www.meescc.cn/zhxx/zxyw/202603/W020260317610986472151.pdf",
         notes="基于259城"),
    dict(level="national", geo_cn="全国", year=2024, value_wt=44000.0,
         caliber="填埋量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="official_aggregate", source_id="meescc_2025",
         credibility="官方统计汇总",
         url="https://www.meescc.cn/zhxx/zxyw/202603/W020260317610986472151.pdf",
         notes="基于207城"),
    dict(level="national", geo_cn="全国", year=2024, value_wt=133000.0,
         caliber="处理能力", substream="处理设施", includes_spoil="n/a",
         method_family="disposal_capacity", source_id="meescc_2025",
         credibility="官方统计汇总",
         url="https://www.meescc.cn/zhxx/zxyw/202603/W020260317610986472151.pdf",
         notes="4262座设施，基于216城——处置端观测过程"),
    # ---------------- national: official wide caliber
    dict(level="national", geo_cn="全国", year=2023, value_wt=300000.0,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="yes",
         method_family="official_wide", source_id="mee_draft_2025",
         credibility="官方文件",
         url="https://www.mee.gov.cn/xxgk2018/xxgk/xxgk06/202504/W020250415370988316064.pdf",
         notes="'超过30亿吨'取下限；占城市垃圾总量40%+；法定5类：渣土/泥浆/工程/拆除/装修"),
    # ---------------- national: industry associations
    dict(level="national", geo_cn="全国", year=2014, value_wt=150000.0,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="industry_assoc", source_id="cdw_alliance",
         credibility="行业协会",
         url="http://www.fqwzyhlm.org/",
         notes="'超15亿吨'取下限"),
    dict(level="national", geo_cn="全国", year=2017, value_wt=159300.0,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="industry_assoc", source_id="envirunion_2017",
         credibility="行业协会估算",
         url="http://www.envirunion.com/newsinfo-14398.html",
         notes="中国环联；另有19.3亿吨的同年并列估计（口径待核）"),
    # ---------------- national: academic series
    dict(level="national", geo_cn="全国", year=2008, value_wt=236000.0,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="academic_WGR_MFA", source_id="zheng_2017",
         credibility="学术估算",
         url="https://www.sciencedirect.com/science/article/abs/pii/S0950061817300703",
         notes="2003-2013年均约23.6亿吨（每年），weight-per-construction-area法，拆除占97%(2013)；已核实 DOI 10.1016/j.conbuildmat.2017.01.055；与gao_2025同期差~2.8倍属口径/系数差异，由模型偏差因子吸收"),
    dict(level="national", geo_cn="全国", year=2022, value_wt=240000.0,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="academic_WGR", source_id="wang_2022",
         credibility="学术估算",
         url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10108335/",
         notes="政策综述转引'约24亿吨/年'(引Duan 2019等)；注意16.75/3.7为武汉市数字非全国；已核实 DOI 10.1177/0734242X221126393 (WM&R 41(4):914-923)"),
    # ---------------- provincial
    dict(level="provincial", geo_cn="北京", year=2024, value_wt=12413.81,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="yes",
         method_family="official_aggregate", source_id="beijing_2024",
         credibility="官方统计",
         url="https://sthjj.beijing.gov.cn/bjhrb/index/xxgk69/sthjlyzwg/1718880/1718881/1718883/743637818/index.html",
         notes="含工程渣土；资源化3745.06万吨，直接利用8668.75万吨"),
    dict(level="provincial", geo_cn="上海", year=2025, value_wt=13643.0,
         caliber="申报处置量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="official_record", source_id="shanghai_2025",
         credibility="官方统计",
         url="https://www.jfdaily.com/sgh/detail?id=1760425",
         notes="申报口径≠产生量；主要利用为场地平整"),
    dict(level="provincial", geo_cn="广东", year=2018, value_wt=None,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="official_aggregate", source_id="guangdong_plan",
         credibility="官方文件",
         url="http://www.jzlj.org.cn/html/10382/10382.html",
         notes="21552.65万m³——体积口径，换算需密度假设（1.1-1.5 t/m³, 宽区间）"),
    dict(level="provincial", geo_cn="广东", year=2019, value_wt=None,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="official_aggregate", source_id="guangdong_plan",
         credibility="官方文件",
         url="http://www.jzlj.org.cn/html/10382/10382.html",
         notes="26955.48万m³（体积）"),
    dict(level="provincial", geo_cn="广东", year=2020, value_wt=None,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="official_aggregate", source_id="guangdong_plan",
         credibility="官方文件",
         url="http://www.jzlj.org.cn/html/10382/10382.html",
         notes="34320.66万m³（体积）"),
    dict(level="provincial", geo_cn="广东", year=2021, value_wt=None,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="official_aggregate", source_id="guangdong_plan",
         credibility="官方文件",
         url="http://www.jzlj.org.cn/html/10382/10382.html",
         notes="33965.90万m³（体积）"),
    dict(level="provincial", geo_cn="江苏", year=2024, value_wt=30000.0,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="industry_assoc", source_id="cace_jiangsu",
         credibility="协会/媒体转引",
         url="https://chinacace.org/news/fieldsview?id=16018",
         notes="'近3亿吨/年'"),
    # ---------------- city exemplars (multi-year where available)
    dict(level="city", geo_cn="广州", year=2024, value_wt=9959.36,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="yes(部分)",
         method_family="official_aggregate", source_id="gz_2024",
         credibility="官方统计",
         url="https://www.gz.gov.cn/zwgk/zdly/hjbh/fzcs/qt/content/post_10356975.html",
         notes="填埋7387.21万吨；另有窄口径：工程垃圾162.94万+拆除/装修222.47万"),
    dict(level="city", geo_cn="东莞", year=2023, value_wt=3600.9,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="official_aggregate", source_id="dg_rd_2025",
         credibility="官方文件",
         url="https://dgrd.dg.gov.cn/dgrd/c100394/202607/64c9b4dbdcb24375aae8a57bdaf825d9.shtml",
         notes="人大执法检查报告"),
    dict(level="city", geo_cn="东莞", year=2024, value_wt=2781.4,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="official_aggregate", source_id="dg_rd_2025",
         credibility="官方文件",
         url="https://dgrd.dg.gov.cn/dgrd/c100394/202607/64c9b4dbdcb24375aae8a57bdaf825d9.shtml",
         notes="同上"),
    dict(level="city", geo_cn="呼和浩特", year=2024, value_wt=491.85,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="official_aggregate", source_id="hhht",
         credibility="官方统计",
         url="http://sthjj.huhhot.gov.cn/ywgz/wrfz/gfzs/202506/t20250604_1897732.html",
         notes="资源化356.24万吨，填埋135.61万吨"),
    dict(level="city", geo_cn="呼和浩特", year=2025, value_wt=256.03,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="official_aggregate", source_id="hhht",
         credibility="官方统计",
         url="http://sthjj.huhhot.gov.cn/ywgz/wrfz/gfzs/202606/t20260609_2012169.html",
         notes="资源化194.21万吨，填埋61.82万吨"),
    dict(level="city", geo_cn="乌海", year=2024, value_wt=229.26,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="yes",
         method_family="official_aggregate", source_id="wuhai_plan",
         credibility="官方文件",
         url="http://zjw.wuhai.gov.cn/zjj/879643/879664/2381030/2025121217042756993.pdf",
         notes="工程渣土178.04万吨等，规划原文"),
    dict(level="city", geo_cn="许昌", year=2020, value_wt=291.2,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="yes",
         method_family="official_aggregate", source_id="xuchang_2020",
         credibility="官方宣传",
         url="https://www.mee.gov.cn/home/ztbd/2020/wfcsjssdgz/wfcsxwbd/wfcsmtbd/202101/t20210112_817110.shtml",
         notes="城区口径：工程弃土259.3+砖渣31.9"),
    dict(level="city", geo_cn="泰州", year=2023, value_wt=762.9,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="official_aggregate", source_id="taizhou_2023",
         credibility="官方统计",
         url="http://hbj.taizhou.gov.cn/hjzl/hjgb/art/2024/art_b148e95cf17f48c4904420f6b3aea5f4.html",
         notes=""),
    dict(level="city", geo_cn="中山", year=2025, value_wt=1493.6,
         caliber="产生量", substream="建筑垃圾总量", includes_spoil="unknown",
         method_family="official_aggregate", source_id="zhongshan_2025",
         credibility="官方统计",
         url="http://zsepb.zs.gov.cn/xxml/ztzl/wryjgxx/wrfz/dzcsndgtfwwrfzgb/content/post_2617054.html",
         notes=""),
    # ---------------- national 2024 sub-streams (meescc_2025)
    dict(level="national", geo_cn="全国", year=2024, value_wt=176000.0,
         caliber="产生量", substream="工程渣土", includes_spoil="yes",
         method_family="official_aggregate", source_id="meescc_2025",
         credibility="官方统计汇总",
         url="https://www.meescc.cn/zhxx/zxyw/202603/W020260317610986472151.pdf",
         notes="基于222城"),
    dict(level="national", geo_cn="全国", year=2024, value_wt=4699.4,
         caliber="产生量", substream="工程泥浆", includes_spoil="yes",
         method_family="official_aggregate", source_id="meescc_2025",
         credibility="官方统计汇总",
         url="https://www.meescc.cn/zhxx/zxyw/202603/W020260317610986472151.pdf",
         notes="基于166城"),
    dict(level="national", geo_cn="全国", year=2024, value_wt=7345.3,
         caliber="产生量", substream="工程垃圾", includes_spoil="no",
         method_family="official_aggregate", source_id="meescc_2025",
         credibility="官方统计汇总",
         url="https://www.meescc.cn/zhxx/zxyw/202603/W020260317610986472151.pdf",
         notes="基于197城"),
    dict(level="national", geo_cn="全国", year=2024, value_wt=10473.8,
         caliber="产生量", substream="拆除垃圾", includes_spoil="no",
         method_family="official_aggregate", source_id="meescc_2025",
         credibility="官方统计汇总",
         url="https://www.meescc.cn/zhxx/zxyw/202603/W020260317610986472151.pdf",
         notes="基于200城"),
    dict(level="national", geo_cn="全国", year=2024, value_wt=5396.7,
         caliber="产生量", substream="装修垃圾", includes_spoil="no",
         method_family="official_aggregate", source_id="meescc_2025",
         credibility="官方统计汇总",
         url="https://www.meescc.cn/zhxx/zxyw/202603/W020260317610986472151.pdf",
         notes="基于201城"),
]

df = pd.DataFrame(R)
df.insert(0, "obs_id", [f"CN-{i+1:03d}" for i in range(len(df))])

# ---- append Gao et al. (2025) digitized national series (Fig 3, 2000-2022) ----
gao_path = OUT / "gao2025_national_series_digitized.csv"
if gao_path.exists():
    gao = pd.read_csv(gao_path)
    gao_rows = []
    for _, r in gao.iterrows():
        gao_rows.append(dict(
            level="national", geo_cn="全国", year=int(r["year"]),
            value_wt=float(r["cw_generation_1e4t"]),  # 10^4 t == 万吨, no conversion
            caliber="产生量", substream="建筑垃圾总量", includes_spoil="no(施工+拆除+装修)",
            method_family="academic_WGR", source_id="gao_2025",
            credibility="学术估算",
            url="https://www.mdpi.com/2071-1050/17/4/1711",
            notes="图3数字化(±5%读数误差);WGR系数:施工0.055t/m²,拆除1.3t/m²(拆除面积=10%新建),装修0.1t/m²"))
    gdf = pd.DataFrame(gao_rows)
    gdf.insert(0, "obs_id", [f"GAO-{i+1:03d}" for i in range(len(gdf))])
    df = pd.concat([df, gdf], ignore_index=True)
    print(f"appended Gao2025 series: {len(gdf)} rows "
          f"({gao['year'].min()}-{gao['year'].max()})")
else:
    print("WARNING: gao2025_national_series_digitized.csv not found, skipped")


# ---- guards ----------------------------------------------------------------
nat23 = df.query("geo_cn=='全国' and year==2023 and caliber=='产生量' "
                 "and substream=='建筑垃圾总量'")
assert len(nat23) == 2, nat23  # meescc24 同口径 + mee_draft 宽口径
spoil = df.query("obs_id=='CN-004'")["value_wt"].iloc[0]
total = df.query("obs_id=='CN-001'")["value_wt"].iloc[0]
assert spoil < total, "渣土不得超过总量"
assert abs(df.query("obs_id=='CN-002'")["value_wt"].iloc[0]
           + df.query("obs_id=='CN-003'")["value_wt"].iloc[0]
           - 162000) == 0  # 利用8.6亿+填埋7.6亿=16.2亿吨 <= 产生24.1亿
util_land = (df.query("obs_id=='CN-002'")["value_wt"].iloc[0]
             + df.query("obs_id=='CN-003'")["value_wt"].iloc[0])
assert util_land < total, "利用+填埋不得超过产生"
# 2024 sub-stream sum must not exceed the 2024 total (partial city coverage)
sub24 = df.query("geo_cn=='全国' and year==2024 and caliber=='产生量' "
                 "and substream!='建筑垃圾总量'")["value_wt"].sum()
tot24 = df.query("obs_id=='CN-006'")["value_wt"].iloc[0]
assert 0 < sub24 < tot24, (sub24, tot24)
df.to_csv(OUT / "china_observations_v01.csv", index=False, encoding="utf-8-sig")

src = df.groupby(["source_id", "method_family", "credibility"]).size().reset_index(name="n_obs")
src.to_csv(OUT / "china_sources_v01.csv", index=False, encoding="utf-8-sig")
print(df[["obs_id", "level", "geo_cn", "year", "value_wt", "caliber",
          "method_family"]].to_string(index=False))
print(f"\nrows={len(df)}, sources={df['source_id'].nunique()}, guards passed")
