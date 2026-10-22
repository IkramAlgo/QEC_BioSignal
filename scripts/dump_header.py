import sys

with open(sys.argv[1], "rb") as f:
    h = f.read(256)

fields = [("version", 8), ("patient", 80), ("recording", 80), ("startdate", 8),
          ("starttime", 8), ("header_bytes", 8), ("reserved", 44),
          ("n_records", 8), ("record_duration", 8), ("n_signals", 4)]
pos = 0
for name, n in fields:
    print(f"{name:16} {h[pos:pos+n]!r}")
    pos += n

n_sig = int(h[252:256].decode("ascii", errors="replace").strip() or 0)
print("expected header_bytes:", 256 * (n_sig + 1))