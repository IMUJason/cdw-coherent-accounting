"""Post-R5 manuscript audit: abstract length, style invariants, key-value consistency."""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
tex = (root / "paper" / "main.tex").read_text(encoding="utf-8")

m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
abstract = m.group(1)
clean = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", abstract)
clean = clean.replace("$", " ").replace("\\", " ")
print("abstract words:", len(clean.split()))

checks = {
    "question marks": tex.count("?"),
    "standalone we/our": len(re.findall(r"\b(We|Our|we|our)\b", tex)),
    "Section refs": len(re.findall(r"Section~?\\ref", tex)),
    "old 2.76": tex.count("2.76"),
    "old 5.6%": tex.count("5.6\\%"),
    "old 36.2": tex.count("36.2"),
    "old 38%": tex.count("38\\%"),
    "old 2.84": tex.count("2.84"),
    "thousand tonnes": tex.count("thousand tonnes"),
    "PI token": len(re.findall(r"\bPI\b", tex)),
    "CI token": len(re.findall(r"\bCI\b", tex)),
    "ground truth": tex.count("ground truth"),
}
for k, v in checks.items():
    print(f"{k}: {v}")

bib = (root / "paper" / "refs.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set()
for mm in re.finditer(r"\\cite[pt]?\{([^}]*)\}", tex):
    for k in mm.group(1).split(","):
        cited.add(k.strip())
print("cited keys:", len(cited), "missing in bib:", sorted(cited - keys))
print("uncited bib entries:", len(keys - cited))

for fig in re.findall(r"includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex):
    p = root / "paper" / fig
    print("figure", fig, "exists" if p.exists() else "MISSING")
