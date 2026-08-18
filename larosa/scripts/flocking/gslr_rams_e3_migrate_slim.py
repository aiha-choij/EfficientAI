"""One-off migration: old-format E3 candidate layer_i.pt files stored
wg/wu/wd/weight (wg/wu identical to the original model's weights, never
refit -- pure redundancy, ~541MB/file instead of ~90MB). Rewrites every
existing <out>/<tag>/layer_i.pt in-place to the slim {wd (bf16), weight}
format gslr_rams_e3.py now writes/reads. Pure CPU tensor load/resave, no
model/GPU needed -- run via qsub -g 0 per this infra's convention (see
report 20260819-041144-rams-e3: hit a shared /raid at 98% full mid-calib,
this reclaims the redundant space without repeating any GPU compute)."""
import argparse
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
args = ap.parse_args()

n_slim = 0
n_already = 0
bytes_before = 0
bytes_after = 0
for tag in sorted(os.listdir(args.out)):
    tagdir = os.path.join(args.out, tag)
    if not os.path.isdir(tagdir) or tag == "xcache":
        continue
    for fn in sorted(os.listdir(tagdir)):
        if not fn.endswith(".pt"):
            continue
        path = os.path.join(tagdir, fn)
        before = os.path.getsize(path)
        sd = torch.load(path, map_location="cpu")
        if "wg" not in sd and "wu" not in sd:
            n_already += 1
            bytes_before += before
            bytes_after += before
            continue
        slim = {"wd": sd["wd"].bfloat16() if sd["wd"].dtype != torch.bfloat16 else sd["wd"],
                "weight": sd["weight"]}
        torch.save(slim, path)
        after = os.path.getsize(path)
        bytes_before += before
        bytes_after += after
        n_slim += 1
        print(f"{tag}/{fn}: {before/1e6:.1f}MB -> {after/1e6:.1f}MB", flush=True)

print(f"done. slimmed={n_slim} already_slim={n_already} "
      f"total {bytes_before/1e9:.2f}GB -> {bytes_after/1e9:.2f}GB "
      f"(reclaimed {(bytes_before-bytes_after)/1e9:.2f}GB)")
