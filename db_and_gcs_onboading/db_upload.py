# gui_supabase_onboarder_min_schema.py
# Razz-styled GUI: Onboard Pokémon cards into Supabase with EXACT schema + full hash set.

import os, sys, ssl, certifi, urllib.request
import asyncio, aiohttp, aiofiles
import threading, queue
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image
import imagehash

from supabase import create_client, Client
from dotenv import load_dotenv
from tcgdexsdk import TCGdex

import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont

# ---------- TLS bootstrap ----------
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: _SSL_CTX
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=_SSL_CTX))
)
# -----------------------------------

# ======= HARD-CODED CONFIG =======
CSV_FOLDER = "/Users/ethandessner/Desktop/PokemonCSVs"
GCS_BUCKET_NAME = "razz_berry"          # used to build image_path
TABLE_NAME = "destined"
UPSERT_ON_CONFLICT = "card_id,subtype_name"
VERIFY_GCS = True                       # if True, we test GCS URL and hash from fallback if needed
# TCGDEX_THROTTLE_SECONDS = 0.5
# =================================

csv_to_api_set_id = {
    "SV10DestinedRivals": "sv10",
}

def load_backup_image_lookup(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    backup_image_lookup = {}
    # expects columns: name, imageUrl
    for row in df.itertuples(index=False):
        name = str(row.name).strip()
        image_url = str(row.imageUrl).strip()
        if image_url and image_url != "nan":
            backup_image_lookup.setdefault(name, image_url)
    print(f"Loaded {len(backup_image_lookup)} fallback images.")
    return backup_image_lookup

async def determine_card_id_format(set_id: str, test_card_number: int):
    formats_to_try = [
        f"{set_id}-{test_card_number}",
        f"{set_id}-{str(test_card_number).zfill(2)}",
        f"{set_id}-{str(test_card_number).zfill(3)}"
    ]
    connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
    async with aiohttp.ClientSession(connector=connector) as session:
        for card_id in formats_to_try:
            url = f"https://api.tcgdex.net/v2/en/cards/{card_id}"
            async with session.get(url) as response:
                if response.status == 200:
                    print(f"Format working for this set: {card_id}")
                    pad_len = len(card_id.split('-')[1])
                    return lambda num: f"{set_id}-{str(num).zfill(pad_len)}"
            # await asyncio.sleep(TCGDEX_THROTTLE_SECONDS)
    print(f"Could not determine working card_id format for set {set_id}. Using raw number.")
    return lambda num: f"{set_id}-{num}"

def build_gcs_url(set_name: str, card_id: str, variant_flag: str) -> str:
    safe_variant = variant_flag.replace(' ', '_').lower()
    filename = f"{card_id}_{safe_variant}.png"
    return f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{set_name}/{filename}"

def compute_model_hashes_from_file(path: str) -> Dict[str, str]:
    """Return 16 fields named like the original model project."""
    img = Image.open(path).convert("RGB")
    def hset(i):
        return {
            "avghashes":     str(imagehash.average_hash(i)),
            "whashes":       str(imagehash.whash(i)),
            "phashes":       str(imagehash.phash(i)),
            "dhashes":       str(imagehash.dhash(i)),
        }
    out = {}

    # normal
    out.update(hset(img))
    # mirrored
    m = img.transpose(Image.FLIP_LEFT_RIGHT)
    out.update({k+"mir": v for k, v in hset(m).items()})
    # upside down
    ud = img.transpose(Image.ROTATE_180)
    out.update({k+"ud": v for k, v in hset(ud).items()})
    # upside down mirrored
    udm = ud.transpose(Image.FLIP_LEFT_RIGHT)
    out.update({k+"udmir": v for k, v in hset(udm).items()})
    return out

async def fetch_bytes(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.read()
    except Exception as e:
        print("download failed:", e)
    return None

async def process_set_to_supabase(
    sdk: TCGdex,
    supabase: Client,
    set_name: str,
    api_set_id: str,
    csv_path: str
):
    print(f"\n==== Onboarding to Supabase: {set_name} ({api_set_id}) ====")
    df = pd.read_csv(csv_path)
    format_fn = await determine_card_id_format(api_set_id, 1)

    # variant mapping from CSV subtype → our canonical lowercase flags
    subtype_to_variant_flag = {
        'Normal': 'normal',
        'Reverse Holofoil': 'reverse',
        'Holofoil': 'holo',
        '1st Edition': 'firstEdition'
    }

    variant_map: Dict[str, set] = {}
    card_id_to_extNumber: Dict[str, str] = {}
    card_id_to_cardName: Dict[str, str] = {}

    for row in df.itertuples(index=False):
        if pd.isna(row.extNumber):
            continue
        ext_number = str(row.extNumber).strip()
        card_number = ext_number.split("/")[0].lstrip("0")
        card_id = format_fn(card_number)

        subtype = str(row.subTypeName).strip()
        variant_flag = subtype_to_variant_flag.get(subtype)
        if not variant_flag:
            print(f"Unrecognized subtype in CSV: {subtype!r} → skipping variant")
            continue

        variant_map.setdefault(card_id, set()).add(variant_flag)
        card_id_to_extNumber[card_id] = ext_number
        card_id_to_cardName[card_id] = str(row.name).strip()

    print(f"Found {len(variant_map)} cards with variants in CSV.")

    # also load fallback image URLs from CSV (TCGplayer)
    backup_image_lookup = load_backup_image_lookup(csv_path)

    connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
    async with aiohttp.ClientSession(connector=connector) as session:
        batch: List[dict] = []
        BATCH_SIZE = 50

        for card_id in sorted(variant_map.keys()):
            # get base card info from TCGdex
            try:
                card = await sdk.card.get(card_id)
            except Exception as e:
                print(f"TCGdex fetch failed for {card_id}: {e}")
                continue

            name_api = card.name
            tcgdex_img = card.get_image_url(quality="high", extension="png")
            ext_number = card_id_to_extNumber.get(card_id, "Unknown")
            display_name = card_id_to_cardName.get(card_id, name_api)
            if ' - ' in display_name:
                display_name = display_name.split(' - ')[0]

            for variant in sorted(variant_map[card_id]):
                # 1) Build canonical image_path (GCS) to store in DB
                image_path = build_gcs_url(set_name, card_id, variant)

                # 2) Choose a URL to DOWNLOAD for hashing (prefer GCS if available)
                chosen_for_hash = image_path
                if VERIFY_GCS:
                    exists = False
                    try:
                        async with session.get(image_path) as resp:
                            exists = (resp.status == 200)
                    except Exception:
                        exists = False
                    if not exists:
                        chosen_for_hash = tcgdex_img or backup_image_lookup.get(display_name) or image_path

                # 3) Download and compute all 16 hashes; if we can’t hash, skip (to honor NOT NULL hash cols)
                img_bytes = await fetch_bytes(session, chosen_for_hash)
                if not img_bytes:
                    print(f"Skip (no image for hashing): {card_id} {variant}")
                    continue

                tmp_path = os.path.join("/tmp", f"{card_id}_{variant}.png")
                async with aiofiles.open(tmp_path, "wb") as f:
                    await f.write(img_bytes)
                try:
                    hashes = compute_model_hashes_from_file(tmp_path)
                finally:
                    try: os.remove(tmp_path)
                    except Exception: pass

                # 4) Build record exactly to your schema
                rec = {
                    "card_id": card_id,
                    "name": display_name,
                    "set_name": set_name,
                    "ext_number": ext_number,
                    "subtype_name": variant,     # already lowercase
                    "image_path": image_path,    # always GCS path

                    # 16 hash fields:
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
                if len(batch) >= BATCH_SIZE:
                    try:
                        supabase.table(TABLE_NAME).upsert(batch, on_conflict=UPSERT_ON_CONFLICT).execute()
                        print(f"Upserted {len(batch)} rows.")
                    except Exception as e:
                        print("Supabase upsert error:", e)
                    batch = []

        # flush
        if batch:
            try:
                supabase.table(TABLE_NAME).upsert(batch, on_conflict=UPSERT_ON_CONFLICT).execute()
                print(f"Upserted {len(batch)} rows.")
            except Exception as e:
                print("Supabase upsert error:", e)

# ---------- Razz theme ----------
RZ_DARK  = "#56362d"
RZ_MED   = "#285838"
RZ_LIGHT = "#f08828"
RZ_PALE  = "#efeecc"

def apply_razz_theme(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(bg=RZ_DARK)
    fams = set(tkfont.families())
    PIXEL = "Press Start 2P" if "Press Start 2P" in fams else ("VT323" if "VT323" in fams else "Courier")
    FONT_UI = (PIXEL, 10)
    FONT_HEADER = (PIXEL, 12, "bold")
    style.configure("GB.TFrame", background=RZ_PALE, borderwidth=4, relief="ridge")
    style.configure("GB.Header.TLabel", background=RZ_MED, foreground=RZ_PALE,
                    font=FONT_HEADER, padding=(10,6))
    style.configure("GB.TLabel", background=RZ_PALE, foreground=RZ_DARK, font=FONT_UI)
    style.configure("GB.TButton", background=RZ_LIGHT, foreground=RZ_PALE, font=FONT_UI,
                    padding=(8,4), borderwidth=3, relief="ridge")
    style.map("GB.TButton",
              background=[("active", RZ_LIGHT), ("pressed", RZ_MED)],
              relief=[("pressed", "sunken")])
    return FONT_UI, FONT_HEADER

def skin_text(txt, font):
    txt.configure(bg=RZ_PALE, fg=RZ_DARK, insertbackground=RZ_DARK,
                  highlightbackground=RZ_DARK, highlightthickness=2, font=font)

def skin_listbox(lb, font):
    lb.configure(bg=RZ_PALE, fg=RZ_DARK,
                 selectbackground=RZ_LIGHT, selectforeground=RZ_PALE,
                 highlightbackground=RZ_DARK, highlightthickness=2,
                 relief="flat", font=font)

# ---------- GUI ----------
class QueueWriter:
    def __init__(self, q: queue.Queue): self.q = q
    def write(self, s): 
        if s: self.q.put(s)
    def flush(self): pass

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAZZ BERRY • Supabase Onboarder")
        self.geometry("860x640")
        self.resizable(False, False)

        # logging
        self.log_queue = queue.Queue()
        self.stdout_backup = sys.stdout
        sys.stdout = QueueWriter(self.log_queue)

        ui_font, header_font = apply_razz_theme(self)

        self.screen = ttk.Frame(self, style="GB.TFrame", padding=10)
        self.screen.place(relx=0.5, rely=0.5, anchor="center", width=820, height=600)

        hdr = ttk.Frame(self.screen, style="GB.TFrame")
        hdr.grid(row=0, column=0, columnspan=3, sticky="we", padx=4, pady=(0, 8))
        ttk.Label(hdr, text="POKéMON CENTER ▸ SUPABASE ONBOARDER", style="GB.Header.TLabel").pack(side="left")

        ttk.Label(self.screen, text=f"CSV: {CSV_FOLDER}", style="GB.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(self.screen, text=f"GCS Bucket: {GCS_BUCKET_NAME}", style="GB.TLabel").grid(row=1, column=1, columnspan=2, sticky="w")

        # Supabase
        load_dotenv()
        supabase_url = os.getenv("SUPABASE_URL", "").strip()
        supabase_key = os.getenv("SUPABASE_KEY", "").strip()
        if not supabase_url or not supabase_key:
            messagebox.showwarning("Supabase", "Set SUPABASE_URL and SUPABASE_KEY in .env or environment.")
        try:
            self.supabase: Optional[Client] = create_client(supabase_url, supabase_key) if (supabase_url and supabase_key) else None
        except Exception as e:
            self.supabase = None
            messagebox.showerror("Supabase", f"Failed to init client: {e}")

        ttk.Label(self.screen, text="Sets:", style="GB.TLabel").grid(row=2, column=0, sticky="ne", padx=(0,8), pady=(6,6))
        self.listbox = tk.Listbox(self.screen, selectmode=tk.EXTENDED, width=44, height=14, exportselection=False)
        for name in csv_to_api_set_id.keys():
            self.listbox.insert(tk.END, name)
        self.listbox.grid(row=2, column=1, sticky="w", pady=(6,6))
        skin_listbox(self.listbox, ui_font)

        opts = ttk.Frame(self.screen, style="GB.TFrame")
        opts.grid(row=2, column=2, sticky="n", padx=(12,0), pady=(6,6))
        self.btn_all = ttk.Button(opts, text="SELECT ALL", style="GB.TButton", command=lambda: self.listbox.select_set(0, tk.END))
        self.btn_all.pack(fill="x", pady=6)
        self.btn_clear = ttk.Button(opts, text="CLEAR", style="GB.TButton", command=lambda: self.listbox.selection_clear(0, tk.END))
        self.btn_clear.pack(fill="x", pady=2)
        self.run_btn = ttk.Button(opts, text="UPSERT → SUPABASE", style="GB.TButton", command=self._run_clicked)
        self.run_btn.pack(fill="x", pady=10)
        ttk.Button(opts, text="QUIT", style="GB.TButton", command=self._exit).pack(fill="x", pady=2)

        ttk.Label(self.screen, text="LOG:", style="GB.TLabel").grid(row=3, column=0, sticky="w", pady=(10,2))
        self.log_box = tk.Text(self.screen, height=14, width=100)
        self.log_box.grid(row=4, column=0, columnspan=3, sticky="we")
        skin_text(self.log_box, ui_font)

        self.screen.grid_columnconfigure(1, weight=1)
        self.after(100, self._drain_log)

    def _append_log(self, s: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", s)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _drain_log(self):
        try:
            while True:
                s = self.log_queue.get_nowait()
                self._append_log(s)
        except queue.Empty:
            pass
        self.after(100, self._drain_log)

    def _run_clicked(self):
        if not os.path.isdir(CSV_FOLDER):
            messagebox.showerror("CSV", f"CSV folder missing:\n{CSV_FOLDER}")
            return
        if self.supabase is None:
            messagebox.showerror("Supabase", "Client not initialized (check SUPABASE_URL/KEY).")
            return

        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("Pick sets", "Select at least one set.")
            return
        sets = [self.listbox.get(i) for i in sel]

        self.run_btn.state(["disabled"])
        def worker():
            async def runner():
                try:
                    sdk = TCGdex()
                    for idx, set_name in enumerate(sets, 1):
                        api_set_id = csv_to_api_set_id.get(set_name)
                        if not api_set_id or api_set_id == "Not Supported":
                            print(f"Skipping unsupported/missing mapping: {set_name}")
                            continue
                        csv_path = os.path.join(CSV_FOLDER, f"{set_name}ProductsAndPrices.csv")
                        if not os.path.isfile(csv_path):
                            print(f"CSV not found: {csv_path} — skipping {set_name}")
                            continue
                        print(f"\n=== [{idx}/{len(sets)}] BEGIN {set_name} ===")
                        await process_set_to_supabase(
                            sdk=sdk,
                            supabase=self.supabase,
                            set_name=set_name,
                            api_set_id=api_set_id,
                            csv_path=csv_path
                        )
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