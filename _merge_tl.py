import csv

path = 'results/dim_reduction/tinyllama_compression_shootout.csv'
sources = ['results/dim_reduction/_tl_b64.csv',
            'results/dim_reduction/_tmp_tl2/tinyllama_compression_shootout.csv']
base = {}
for src in sources:
    try:
        for r in csv.DictReader(open(src, encoding='utf-8')):
            base[(r['method'], r['budget_bytes'])] = r
    except FileNotFoundError:
        print('missing:', src)
rows = sorted(base.values(),
               key=lambda r: (r['budget_bytes'].zfill(4), r['method']))
with open(path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
import collections
c = collections.Counter(r['method'] for r in rows)
print('merged rows:', len(rows), dict(c))
