import os, sys, ssl, certifi, urllib.request, re, asyncio, aiohttp, aiofiles
import threading, queue
from typing import Dict, List, Optional, Callable

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import tempfile

from supabase import create_client, Client
from dotenv import load_dotenv

import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont

# --- TLS bootstrap (for macOS cert issues etc.) ---
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: _SSL_CTX
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=_SSL_CTX))
)

# ================== CONFIG ==================
CSV_FOLDER = "/Users/ethandessner/Desktop/PokemonCSVs"
GCS_BUCKET_NAME = "razz_berry"
TABLE_NAME = "destinedrivals"
UPSERT_ON_CONFLICT = "card_id,subtype_name"
VERIFY_GCS = True          # HEAD/GET check the GCS URL before hashing
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30, connect=10)
IMG_HEADERS = {"User-Agent": "RazzBerryOnboarder/1.0", "Accept": "image/png,image/*;q=0.8,*/*;q=0.5"}
# ============================================

# Map set folder name → api_set_id prefix (used to build card_id)
csv_to_api_set_id = {
    "SV10DestinedRivals" : "sv10"
}

# ---- helpers ----

def prepare_card_for_hash(img_pil: Image.Image, max_side: int = 900) -> Image.Image:
    """Same normalization for DB and live: trim border, CLAHE on luma, mild denoise, area resample."""
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # trim 3.5% uniform border (sleeves/edge lighting)
    h, w = bgr.shape[:2]
    dx = int(w * 0.035); dy = int(h * 0.035)
    bgr = bgr[dy:h-dy, dx:w-dx].copy()

    # CLAHE on luminance only
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(y)
    bgr = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

    # mild edge-preserving denoise
    bgr = cv2.bilateralFilter(bgr, 5, 60, 60)

    # downscale (if needed) with INTER_AREA (best for shrink)
    h, w = bgr.shape[:2]
    scale = float(max_side) / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def load_backup_image_lookup(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    lookup = {}
    # expects columns: name, imageUrl
    for row in df.itertuples(index=False):
        name = str(row.name).strip()
        url = str(row.imageUrl).strip()
        if url and url != "nan" and name not in lookup:
            lookup[name] = url
    return lookup

def gcs_object_path(set_name: str, card_id: str, variant_flag: str) -> str:
    safe_variant = variant_flag.replace(' ', '_').lower()
    filename = f"{card_id}_{safe_variant}.png"
    return f"{set_name}/{filename}"

def build_gcs_view_url(obj_path: str) -> str:
    # human viewer (your required DB format) — may require auth in browsers
    return f"https://storage.cloud.google.com/{GCS_BUCKET_NAME}/{obj_path}"

def build_gcs_download_url(obj_path: str) -> str:
    # programmatic/public fetch; works with curl/aiohttp if object is public
    return f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{obj_path}"


def compute_model_hashes_from_file(path: str) -> Dict[str, str]:
    img = Image.open(path).convert("RGB")
    img = prepare_card_for_hash(img, max_side=900)   # <<< same as matcher
    def hset(i):
        return {
            "avghashes": str(imagehash.average_hash(i, hash_size=8)),
            "whashes":   str(imagehash.whash(i, hash_size=8)),
            "phashes":   str(imagehash.phash(i, hash_size=8)),
            "dhashes":   str(imagehash.dhash(i, hash_size=8)),
        }
    out = {}
    out.update(hset(img))
    m  = img.transpose(Image.FLIP_LEFT_RIGHT)
    ud = img.transpose(Image.ROTATE_180)
    um = ud.transpose(Image.FLIP_LEFT_RIGHT)
    out.update({k+"mir":   v for k, v in hset(m ).items()})
    out.update({k+"ud":    v for k, v in hset(ud).items()})
    out.update({k+"udmir": v for k, v in hset(um).items()})
    return out

async def head_exists(session: aiohttp.ClientSession, url: str) -> bool:
    try:
        async with session.get(url, headers=IMG_HEADERS, allow_redirects=True) as resp:
            ctype = resp.headers.get("Content-Type", "")
            return resp.status == 200 and ctype.startswith("image/")
    except Exception:
        return False

async def fetch_bytes(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    try:
        async with session.get(url, headers=IMG_HEADERS, allow_redirects=True) as resp:
            if resp.status == 200 and resp.headers.get("Content-Type","").startswith("image/"):
                return await resp.read()
    except Exception:
        pass
    return None


def parse_prefix(num_str: str):
    """Return (letters, digits) for an extNumber numerator like '004' or 'BW4'."""
    s = num_str.strip()
    m = re.match(r"^([A-Za-z]+)?(\d+)$", s)
    if not m:
        return "", s
    letters, digits = m.groups()
    return letters or "", digits

async def infer_card_formatter_from_gcs(
    session: aiohttp.ClientSession, set_name: str, api_set_id: str, df: pd.DataFrame
) -> Callable[[str], str]:
    """
    Decide zero-padding from actual objects in your GCS.
    Try a few sample numerators with zfill 1/2/3 and pick the first that exists.
    If nothing matches, fall back to CSV-based heuristic: letters → zfill(2), numeric → zfill(3).
    """
    samples = []
    for ext in df["extNumber"].dropna().astype(str).tolist():
        prefix = ext.split("/")[0]
        letters, digits = parse_prefix(prefix)
        samples.append((letters, digits))
        if len(samples) >= 10:
            break

    for pad in (3, 2, 1):
        for letters, digits in samples:
            num = f"{letters}{str(int(digits)).zfill(pad)}" if digits.isdigit() else f"{letters}{digits}"
            card_id = f"{api_set_id}-{num}"

            obj_path = gcs_object_path(set_name, card_id, "normal")
            test_url = build_gcs_download_url(obj_path)

            if await head_exists(session, test_url):
                return lambda d: f"{api_set_id}-{ (f'{int(d)}'.zfill(pad) if d.isdigit() else d) }"

    # heuristic fallback
    has_letters = any(l for l, _ in samples)
    pad = 2 if has_letters else 3
    return lambda d: f"{api_set_id}-{ (f'{int(d)}'.zfill(pad) if d.isdigit() else d) }"

async def process_set_to_supabase_noapi(
    supabase: Client, set_name: str, api_set_id: str, csv_path: str
):
    print(f"\n==== Onboarding (no API): {set_name} ({api_set_id}) ====")
    df = pd.read_csv(csv_path)

    # canonicalize variants from CSV subtype
    subtype_to_variant_flag = {
        'Normal': 'normal',
        'Reverse Holofoil': 'reverse',
        'Holofoil': 'holo',
        '1st Edition': 'firstEdition'
    }

    # Build variant map directly from CSV (and dedupe)
    variant_map: Dict[str, set] = {}
    card_id_to_extNumber: Dict[str, str] = {}
    card_id_to_name: Dict[str, str] = {}

    connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
    async with aiohttp.ClientSession(connector=connector, timeout=HTTP_TIMEOUT) as session:
        # choose a card_id formatter that matches what you uploaded to GCS
        format_fn = await infer_card_formatter_from_gcs(session, set_name, api_set_id, df)

        for row in df.itertuples(index=False):
            if pd.isna(row.extNumber):
                continue
            ext_number = str(row.extNumber).strip()
            raw_prefix = ext_number.split("/")[0].strip()
            letters, digits = parse_prefix(raw_prefix)
            card_number = digits if digits else raw_prefix
            card_id = format_fn(card_number)

            subtype = str(row.subTypeName).strip()
            variant_flag = subtype_to_variant_flag.get(subtype)
            if not variant_flag:
                # unrecognized subtype → skip
                continue

            # dedupe by (card_id, variant)
            variant_set = variant_map.setdefault(card_id, set())
            variant_set.add(variant_flag)

            card_id_to_extNumber[card_id] = ext_number
            card_id_to_name[card_id] = str(row.name).strip()

        print(f"Found {len(variant_map)} cards with variants in CSV.")

        # fallback image lookup from CSV TCGplayer column, for hashing only
        backup_image_lookup = load_backup_image_lookup(csv_path)

        batch: List[dict] = []
        BATCH = 80

        for card_id in sorted(variant_map.keys()):
            display_name = card_id_to_name.get(card_id, "Unknown")
            ext_number = card_id_to_extNumber.get(card_id, "Unknown")

            for variant in sorted(variant_map[card_id]):
                # 1) persistent image_path we store in DB (always your GCS URL)
                obj_path   = gcs_object_path(set_name, card_id, variant)
                # image_path = build_gcs_view_url(obj_path)          # what you store in Supabase
                image_path = build_gcs_download_url(obj_path)

                url_for_fetch = build_gcs_download_url(obj_path)   # what we GET for hashing



                chosen_for_hash = url_for_fetch
                if VERIFY_GCS:
                    ok = await head_exists(session, url_for_fetch)
                    if not ok:
                        fallback = backup_image_lookup.get(display_name)
                        if fallback:
                            chosen_for_hash = fallback

                img_bytes = await fetch_bytes(session, chosen_for_hash)
                if not img_bytes:
                    print(f"Skip (no image available to hash): {card_id} {variant}")
                    continue

                tmp_path = None
                try:
                    # create a secure temp file and write bytes
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
                        tmp_path = tf.name
                        tf.write(img_bytes)
                    hashes = compute_model_hashes_from_file(tmp_path)
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try: os.remove(tmp_path)
                        except Exception: pass


                rec = {
                    "card_id": card_id,
                    "name": display_name,
                    "set_name": set_name,
                    "ext_number": ext_number,
                    "subtype_name": variant,
                    "image_path": image_path,

                    "avghashes":     hashes["avghashes"],
                    "avghashesmir":  hashes["avghashesmir"],
                    "avghashesud":   hashes["avghashesud"],
                    "avghashesudmir":hashes["avghashesudmir"],

                    "whashes":       hashes["whashes"],
                    "whashesmir":    hashes["whashesmir"],
                    "whashesud":     hashes["whashesud"],
                    "whashesudmir":  hashes["whashesudmir"],

                    "phashes":       hashes["phashes"],
                    "phashesmir":    hashes["phashesmir"],
                    "phashesud":     hashes["phashesud"],
                    "phashesudmir":  hashes["phashesudmir"],

                    "dhashes":       hashes["dhashes"],
                    "dhashesmir":    hashes["dhashesmir"],
                    "dhashesud":     hashes["dhashesud"],
                    "dhashesudmir":  hashes["dhashesudmir"],
                }
                batch.append(rec)

                if len(batch) >= BATCH:
                    try:
                        supabase.table(TABLE_NAME).upsert(batch, on_conflict=UPSERT_ON_CONFLICT).execute()
                        print(f"Upserted {len(batch)} rows.")
                    except Exception as e:
                        print("Supabase upsert error:", e)
                    batch = []

        if batch:
            try:
                supabase.table(TABLE_NAME).upsert(batch, on_conflict=UPSERT_ON_CONFLICT).execute()
                print(f"Upserted {len(batch)} rows.")
            except Exception as e:
                print("Supabase upsert error:", e)

# --------- Razz theme ---------
RZ_DARK  = "#56362d"; RZ_MED = "#285838"; RZ_LIGHT = "#f08828"; RZ_PALE = "#efeecc"
def apply_razz_theme(root):
    style = ttk.Style(root); style.theme_use("clam"); root.configure(bg=RZ_DARK)
    fams = set(tkfont.families()); PIXEL = "Press Start 2P" if "Press Start 2P" in fams else ("VT323" if "VT323" in fams else "Courier")
    FONT_UI = (PIXEL, 10); FONT_HEADER = (PIXEL, 12, "bold")
    style.configure("GB.TFrame", background=RZ_PALE, borderwidth=4, relief="ridge")
    style.configure("GB.Header.TLabel", background=RZ_MED, foreground=RZ_PALE, font=FONT_HEADER, padding=(10,6))
    style.configure("GB.TLabel", background=RZ_PALE, foreground=RZ_DARK, font=FONT_UI)
    style.configure("GB.TButton", background=RZ_LIGHT, foreground=RZ_PALE, font=FONT_UI, padding=(8,4), borderwidth=3, relief="ridge")
    style.map("GB.TButton", background=[("active", RZ_LIGHT), ("pressed", RZ_MED)], relief=[("pressed", "sunken")])
    return FONT_UI, FONT_HEADER
def skin_text(txt, font): txt.configure(bg=RZ_PALE, fg=RZ_DARK, insertbackground=RZ_DARK, highlightbackground=RZ_DARK, highlightthickness=2, font=font)
def skin_listbox(lb, font): lb.configure(bg=RZ_PALE, fg=RZ_DARK, selectbackground=RZ_LIGHT, selectforeground=RZ_PALE, highlightbackground=RZ_DARK, highlightthickness=2, relief="flat", font=font)

# --------- GUI ----------
class QueueWriter:
    def __init__(self, q: queue.Queue): self.q = q
    def write(self, s): 
        if s: self.q.put(s)
    def flush(self): pass

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAZZ BERRY • Supabase Onboarder (No API)")
        self.geometry("860x640"); self.resizable(False, False)

        self.logo_img = None  # keep a ref on the instance so Tk doesn't GC it
        RAZZ_ICON_PATH = "/Users/ethandessner/dev/Pokemon-Card-Scanner/images/razz_iso.png"  # transparent PNG/GIF
        try:
            self.logo_img = tk.PhotoImage(master=self, file=RAZZ_ICON_PATH)
            # Optional (Windows/Linux): show in titlebar/taskbar too
            if sys.platform != "darwin":
                self.iconphoto(True, self.logo_img)
        except Exception as e:
            print("logo load failed:", e)

        self.log_queue = queue.Queue(); self.stdout_backup = sys.stdout; sys.stdout = QueueWriter(self.log_queue)
        ui_font, header_font = apply_razz_theme(self)

        self.screen = ttk.Frame(self, style="GB.TFrame", padding=10)
        self.screen.place(relx=0.5, rely=0.5, anchor="center", width=820, height=600)

        hdr = ttk.Frame(self.screen, style="GB.TFrame")
        hdr.grid(row=0, column=0, columnspan=3, sticky="we", padx=4, pady=(0,8))

        if self.logo_img:
            ttk.Label(hdr, image=self.logo_img, style="GB.TLabel").pack(side="left", padx=(2,8))
        ttk.Label(hdr, text="RAZZ BERRY ▸ SUPABASE ONBOARDER", style="GB.Header.TLabel").pack(side="left")


        ttk.Label(self.screen, text=f"CSV: {CSV_FOLDER}", style="GB.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(self.screen, text=f"GCS Bucket: {GCS_BUCKET_NAME}", style="GB.TLabel").grid(row=1, column=1, columnspan=2, sticky="w")

        load_dotenv()
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_KEY", "").strip()
        if not url or not key:
            messagebox.showwarning("Supabase", "Set SUPABASE_URL and SUPABASE_KEY in your environment/.env")
        try:
            self.supabase: Optional[Client] = create_client(url, key) if (url and key) else None
        except Exception as e:
            self.supabase = None; messagebox.showerror("Supabase", f"Failed to init: {e}")

        ttk.Label(self.screen, text="Sets:", style="GB.TLabel").grid(row=2, column=0, sticky="ne", padx=(0,8), pady=(6,6))
        self.listbox = tk.Listbox(self.screen, selectmode=tk.EXTENDED, width=44, height=14, exportselection=False)
        for name in csv_to_api_set_id.keys(): self.listbox.insert(tk.END, name)
        self.listbox.grid(row=2, column=1, sticky="w", pady=(6,6))
        skin_listbox(self.listbox, ui_font)

        side = ttk.Frame(self.screen, style="GB.TFrame"); side.grid(row=2, column=2, sticky="n", padx=(12,0), pady=(6,6))
        ttk.Button(side, text="SELECT ALL", style="GB.TButton", command=lambda: self.listbox.select_set(0, tk.END)).pack(fill="x", pady=6)
        ttk.Button(side, text="CLEAR", style="GB.TButton", command=lambda: self.listbox.selection_clear(0, tk.END)).pack(fill="x", pady=2)
        self.run_btn = ttk.Button(side, text="UPSERT → SUPABASE", style="GB.TButton", command=self._run_clicked); self.run_btn.pack(fill="x", pady=10)
        ttk.Button(side, text="QUIT", style="GB.TButton", command=self._exit).pack(fill="x", pady=2)

        ttk.Label(self.screen, text="LOG:", style="GB.TLabel").grid(row=3, column=0, sticky="w", pady=(10,2))
        self.log_box = tk.Text(self.screen, height=14, width=100); self.log_box.grid(row=4, column=0, columnspan=3, sticky="we"); skin_text(self.log_box, ui_font)

        self.screen.grid_columnconfigure(1, weight=1)
        self.after(100, self._drain_log)

    def _append_log(self, s: str):
        self.log_box.configure(state="normal"); self.log_box.insert("end", s); self.log_box.see("end"); self.log_box.configure(state="disabled")
    def _drain_log(self):
        try:
            while True: self._append_log(self.log_queue.get_nowait())
        except queue.Empty: pass
        self.after(100, self._drain_log)

    def _run_clicked(self):
        if not os.path.isdir(CSV_FOLDER):
            messagebox.showerror("CSV", f"CSV folder missing:\n{CSV_FOLDER}"); return
        if self.supabase is None:
            messagebox.showerror("Supabase", "Client not initialized"); return
        sel = self.listbox.curselection()
        if not sel: messagebox.showwarning("Pick sets", "Select at least one set."); return
        sets = [self.listbox.get(i) for i in sel]
        self.run_btn.state(["disabled"])

        def worker():
            async def runner():
                try:
                    for idx, set_name in enumerate(sets, 1):
                        api_set_id = csv_to_api_set_id.get(set_name)
                        if not api_set_id or api_set_id == "Not Supported":
                            print(f"Skipping unsupported/missing mapping: {set_name}"); continue
                        csv_path = os.path.join(CSV_FOLDER, f"{set_name}ProductsAndPrices.csv")
                        if not os.path.isfile(csv_path):
                            print(f"CSV not found: {csv_path} — skipping {set_name}"); continue
                        print(f"\n=== [{idx}/{len(sets)}] BEGIN {set_name} ===")
                        await process_set_to_supabase_noapi(self.supabase, set_name, api_set_id, csv_path)
                        print(f"=== [{idx}/{len(sets)}] END {set_name} ===")
                    self.after(0, lambda: messagebox.showinfo("Done", "Upsert complete."))
                except Exception as e:
                    self.after(0, lambda exc=e: messagebox.showerror("Error", str(exc)))
                finally:
                    self.after(0, lambda: self.run_btn.state(["!disabled"]))
            asyncio.run(runner())

        threading.Thread(target=worker, daemon=True).start()

    def _exit(self):
        try: sys.stdout = self.stdout_backup
        except Exception: pass
        self.destroy()

if __name__ == "__main__":
    ttk.Style().theme_use("clam")
    App().mainloop()
