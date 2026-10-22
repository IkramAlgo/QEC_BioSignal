import os
import sys

path = sys.argv[1]
size = os.path.getsize(path)

with open(path, "rb") as f:
    h = f.read(256)
    ns = int(h[252:256])
    n_rec = int(h[236:244])
    hdr_bytes = int(h[184:192])
    sig = f.read(ns * 256)

sizes = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
names = ["label", "transducer", "dim", "pmin", "pmax", "dmin", "dmax", "prefilter", "nsamp", "reserved"]
fields = {}
pos = 0
for name, sz in zip(names, sizes):
    fields[name] = [sig[pos + i * sz: pos + (i + 1) * sz] for i in range(ns)]
    pos += sz * ns

problems = 0

# 1. non printable bytes in any text field
for name in ["label", "transducer", "dim", "prefilter", "reserved"]:
    for i, raw in enumerate(fields[name]):
        if any(b < 32 or b > 126 for b in raw):
            print(f"signal {i}: non printable bytes in {name}: {raw!r}")
            problems += 1

# 2. numeric fields parse, and ranges are sane
nsamp = []
for i in range(ns):
    try:
        pmin = float(fields["pmin"][i]); pmax = float(fields["pmax"][i])
        dmin = float(fields["dmin"][i]); dmax = float(fields["dmax"][i])
        n = int(fields["nsamp"][i])
    except ValueError:
        print(f"signal {i} ({fields['label'][i].decode(errors='replace').strip()}): numeric field will not parse")
        problems += 1
        nsamp.append(0)
        continue
    nsamp.append(n)
    if pmin == pmax or dmin >= dmax:
        print(f"signal {i} ({fields['label'][i].decode(errors='replace').strip()}): bad range pmin={pmin} pmax={pmax} dmin={dmin} dmax={dmax}")
        problems += 1

# 3. expected file size versus real file size
expected = hdr_bytes + n_rec * sum(nsamp) * 2
print(f"expected size {expected}, actual size {size}, difference {size - expected}")
if size != expected:
    problems += 1

# 4. annotation channel, check the time keeping entry at the start of every record
ann = [i for i, l in enumerate(fields["label"]) if l.decode(errors="replace").startswith("EDF Annotations")]
print(f"annotation signals at index: {ann}")
if ann:
    a = ann[0]
    offset_in_rec = sum(nsamp[:a]) * 2
    ann_len = nsamp[a] * 2
    rec_bytes = sum(nsamp) * 2
    bad = []
    with open(path, "rb") as f:
        for r in range(n_rec):
            f.seek(hdr_bytes + r * rec_bytes + offset_in_rec)
            chunk = f.read(ann_len)
            ok = len(chunk) == ann_len and chunk[:1] in (b"+", b"-") and b"\x14\x14\x00" in chunk[:40]
            if not ok:
                bad.append((r, chunk[:30]))
    print(f"records with a bad time keeping annotation: {len(bad)} of {n_rec}")
    for r, c in bad[:10]:
        print(f"  record {r}: {c!r}")
    problems += len(bad)

print(f"\nTotal problems found: {problems}")