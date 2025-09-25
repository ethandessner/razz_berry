import os, sys, asyncio, certifi
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# TLS bootstrap for macOS etc.
import ssl, urllib.request
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: _SSL_CTX
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=_SSL_CTX))
)

import cv2
from PIL import Image
import numpy as np
import imagehash
from dotenv import load_dotenv
from supabase import create_client, Client

# ---------- CONFIG ----------
CUTOFF = 18                           # same as your original project
TABLE  = os.getenv("SUPABASE_TABLE","destined")
PAGE_SIZE = 1000                      # fetch rows in pages
OPTIONAL_SET_FILTER = None            # e.g. "CrystalGuardians" to limit scope
# -----------------------------

@dataclass
class CardHashes:
    card_id: str
    name: str
    set_name: str
    ext_number: str
    subtype_name: str
    image_path: str
    # each list: [normal, mir, ud, udmir] as ImageHash objects
    avg:  List[imagehash.ImageHash]
    wh:   List[imagehash.ImageHash]
    ph:   List[imagehash.ImageHash]
    dh:   List[imagehash.ImageHash]

def hx(s: str) -> imagehash.ImageHash:
    # DB stores hex strings; convert to ImageHash
    return imagehash.hex_to_hash(str(s))

def build_card(row: dict) -> CardHashes:
    return CardHashes(
        card_id=row["card_id"],
        name=row["name"],
        set_name=row["set_name"],
        ext_number=row["ext_number"],
        subtype_name=row["subtype_name"],
        image_path=row["image_path"],
        avg=[hx(row["avghashes"]), hx(row["avghashesmir"]), hx(row["avghashesud"]), hx(row["avghashesudmir"])],
        wh =[hx(row["whashes"]),  hx(row["whashesmir"]),  hx(row["whashesud"]),  hx(row["whashesudmir"])],
        ph =[hx(row["phashes"]),  hx(row["phashesmir"]),  hx(row["phashesud"]),  hx(row["phashesudmir"])],
        dh =[hx(row["dhashes"]),  hx(row["dhashesmir"]),  hx(row["dhashesud"]),  hx(row["dhashesudmir"])],
    )

def compute_hashes_pil(img_pil: Image.Image) -> Tuple[imagehash.ImageHash, imagehash.ImageHash, imagehash.ImageHash, imagehash.ImageHash]:
    img = img_pil.convert("RGB")
    return (
        imagehash.average_hash(img),
        imagehash.whash(img),
        imagehash.phash(img),
        imagehash.dhash(img),
    )

def score_card(captured: Tuple[imagehash.ImageHash, imagehash.ImageHash, imagehash.ImageHash, imagehash.ImageHash],
               card: CardHashes) -> int:
    cap_avg, cap_wh, cap_ph, cap_dh = captured
    # MIN across 4 orientations for each hash family
    d_avg = min(cap_avg - h for h in card.avg)
    d_wh  = min(cap_wh  - h for h in card.wh)
    d_ph  = min(cap_ph  - h for h in card.ph)
    d_dh  = min(cap_dh  - h for h in card.dh)
    # MAX of those mins (your "max of mins" rule)
    return max(d_avg, d_wh, d_ph, d_dh)

def fetch_all_cards(sb: Client, set_filter: Optional[str] = OPTIONAL_SET_FILTER) -> List[CardHashes]:
    print("Fetching cards from Supabase…")
    offset = 0
    cards: List[CardHashes] = []
    while True:
        q = sb.table(TABLE).select(
            "card_id,name,set_name,ext_number,subtype_name,image_path,"
            "avghashes,avghashesmir,avghashesud,avghashesudmir,"
            "whashes,whashesmir,whashesud,whashesudmir,"
            "phashes,phashesmir,phashesud,phashesudmir,"
            "dhashes,dhashesmir,dhashesud,dhashesudmir"
        ).range(offset, offset + PAGE_SIZE - 1)
        if set_filter:
            q = q.eq("set_name", set_filter)
        res = q.execute()
        data = res.data or []
        if not data:
            break
        for row in data:
            try:
                cards.append(build_card(row))
            except Exception as e:
                # skip malformed rows
                print("Row parse error, skipping:", e)
        offset += PAGE_SIZE
        if len(data) < PAGE_SIZE:
            break
    print(f"Loaded {len(cards)} rows.")
    return cards

def pil_from_bgr(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def _put_instructions(img):
    txt1 = "Press [S]/[SPACE]/[ENTER] or CLICK to capture • [Q] to quit"
    (w, h), _ = cv2.getTextSize(txt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    pad = 10
    overlay = img.copy()
    cv2.rectangle(overlay, (8, img.shape[0]-h-20), (8+w+16, img.shape[0]-8), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, txt1, (16, img.shape[0]-14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

def capture_frame() -> Optional[np.ndarray]:
    # Try default camera; change 0→1/2 if needed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available.")
        return None

    win = "Scanner"
    cv2.namedWindow(win)
    captured = {"do": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            captured["do"] = True
    cv2.setMouseCallback(win, on_mouse)

    frame = None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        disp = img.copy()
        _put_instructions(disp)
        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            frame = None
            break
        if key in (ord('s'), ord('S'), 32, 13) or captured["do"]:  # space=32, enter=13
            frame = img.copy()
            break

    cap.release()
    cv2.destroyWindow(win)
    if frame is None:
        return None

    # Immediately open ROI selector
    roi = select_roi(frame)
    return roi

def select_roi(img: np.ndarray) -> np.ndarray:
    win = "Select Card ROI (ENTER to accept)"
    r = cv2.selectROI(win, img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win)
    x, y, w, h = r
    if w > 0 and h > 0:
        return img[y:y+h, x:x+w]
    return img

def main():
    load_dotenv()
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        print("Set SUPABASE_URL and SUPABASE_KEY in .env")
        sys.exit(1)

    sb = create_client(url, key)

    # 1) Load DB into memory (fast matching)
    cards = fetch_all_cards(sb, set_filter=OPTIONAL_SET_FILTER)
    if not cards:
        print("No cards found.")
        sys.exit(1)

    # 2) Capture an image
    frame = capture_frame()
    if frame is None:
        print("No frame captured.")
        sys.exit(0)

    # optional ROI
    print("Press [r] in the next window to draw a crop; or close it to keep full frame.")
    roi = select_roi(frame)

    # basic preproc: resize tall side to ~800px for stability (imagehash is robust; this is optional)
    h, w = roi.shape[:2]
    scale = 800.0 / max(h, w)
    if scale < 1.0:
        roi = cv2.resize(roi, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # 3) Compute hashes on the capture
    cap_hashes = compute_hashes_pil(pil_from_bgr(roi))

    # 4) Compare against DB
    best_score = 10**9
    best_card: Optional[CardHashes] = None

    for c in cards:
        s = score_card(cap_hashes, c)
        if s < best_score:
            best_score, best_card = s, c

    # 5) Report
    if best_card and best_score < CUTOFF:
        print("\n=== MATCH ===")
        print(f" Score:         {best_score}  (cutoff {CUTOFF})")
        print(f" Card ID:       {best_card.card_id}")
        print(f" Name:          {best_card.name}")
        print(f" Set:           {best_card.set_name}")
        print(f" Ext number:    {best_card.ext_number}")
        print(f" Variant:       {best_card.subtype_name}")
        print(f" Image path:    {best_card.image_path}")
    else:
        print("\nNo confident match found.")
        if best_card:
            print(f" Best candidate: {best_card.card_id} ({best_card.name}) with score {best_score}")

if __name__ == "__main__":
    main()
