"""Count citation usage per key in main.tex."""
import re
from collections import Counter

s = open("paper/main.tex", encoding="utf-8").read()
c = Counter()
for m in re.finditer(r"\\cite[pt]?\{([^}]*)\}", s):
    for k in m.group(1).split(","):
        c[k.strip()] += 1
print(len(c), "keys cited")
for k, v in sorted(c.items(), key=lambda x: (x[1], x[0])):
    print(v, k)
