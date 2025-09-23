# one-off script: make_bg_transparent.py
# pip install pillow numpy
from PIL import Image
import numpy as np
import sys

def make_bg_transparent(in_path, out_path, fuzz=30):
    im = Image.open(in_path).convert("RGBA")
    arr = np.array(im)
    bg = arr[0, 0, :3]                       # sample top-left color (R,G,B)
    diff = np.abs(arr[:, :, :3] - bg)        # per-channel difference
    mask = (diff <= fuzz).all(axis=2)        # within tolerance?
    arr[mask, 3] = 0                         # set alpha=0 for background
    out = Image.fromarray(arr, "RGBA")

    # optional: auto-trim transparent border
    alpha = out.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        out = out.crop(bbox)

    out.save(out_path)
    print(f"Saved isolated PNG → {out_path}")

if __name__ == "__main__":
    in_path = sys.argv[1]          # e.g. /path/to/razz.png
    out_path = sys.argv[2]         # e.g. /path/to/razz_iso.png
    make_bg_transparent(in_path, out_path, fuzz=30)