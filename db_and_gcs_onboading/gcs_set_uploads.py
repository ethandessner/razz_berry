# gui_upload_multiselect_gb_listbox.py
# Game Boy–styled uploader using the SAME multi-select LISTBOX as the earlier working GUI.
# - Hard-coded CSV folder + GCS bucket
# - Listbox (multi-select) with "Select All" / "Clear"
# - Your original pipeline for CSV->TCGdex image->GCS
# - TLS bootstrap via certifi to avoid SSL issues

# source ~/venvs/tk/bin/activate 


import os, sys, re, ssl, certifi, asyncio, threading, queue
from typing import Dict, List

# ---------------- TLS BOOTSTRAP (covers urllib/aiohttp/tcgdexsdk/requests) ----------------
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: _SSL_CTX
import urllib.request
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=_SSL_CTX))
)
# ------------------------------------------------------------------------------------------

import pandas as pd
import aiohttp
import aiofiles
from google.cloud import storage
from tcgdexsdk import TCGdex

import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont

# ========= HARD-CODED CONFIG (edit if needed) =========
CSV_FOLDER = "/Users/ethandessner/Desktop/PokemonCSVs"
GCS_BUCKET_NAME = "razz_berry"
# ======================================================

# ---- Your original mapping (unchanged) ----
csv_to_api_set_id = {
    "Aquapolis": "ecard2",
    "Arceus": "pl4",
    "BaseSet2": "base4",
    "BaseSet": "base1",
    "BlackandWhite": "bw1",
    "BlackandWhitePromos": "bwp",
    "BoundariesCrossed": "bw7",
    "CallofLegends": "col1",
    "Celebrations": "cel25",
    "Champion'sPath": "swsh3.5",
    "CrownZenith": "swsh12.5",
    "CrystalGuardians": "ex14",
    "DarkExplorers": "bw5",
    "DeltaSpecies": "ex11",
    "Deoxys": "ex8",
    "DetectivePikachu": "det1",
    "DiamondandPearl": "dp1",
    "DiamondandPearlPromos": "dpp",
    "DoubleCrisis": "dc1",
    "DragonFrontiers": "ex15",
    "DragonMajesty": "sm7.5",
    "Dragon": "ex3",
    "DragonVault": "dv1",
    "DragonsExalted": "bw6",
}

# --------- Original helpers (unchanged in spirit) ---------
def load_backup_image_lookup(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    backup_image_lookup = {}
    for row in df.itertuples(index=False):
        name = str(row.name).strip()
        image_url = str(row.imageUrl).strip()
        if image_url and image_url != "nan":
            if name not in backup_image_lookup:
                backup_image_lookup[name] = image_url
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
    print(f"Could not determine working card_id format for set {set_id}. Using raw number.")
    return lambda num: f"{set_id}-{num}"

async def upload_to_gcs(local_path: str, gcs_path: str, gcs_bucket):
    blob = gcs_bucket.blob(gcs_path)
    if not blob.exists():
        blob.upload_from_filename(local_path)
        print(f"✅ Uploaded to GCS: {gcs_path}")
    else:
        print(f"⚠️ GCS already contains: {gcs_path}")

async def process_csv_for_set(sdk, csv_path: str, api_set_id: str, set_name_for_output_folder: str,
                              backup_image_lookup: Dict[str, str], gcs_bucket):
    print(f"\n==== Processing set {api_set_id} from CSV {csv_path} ====")
    df = pd.read_csv(csv_path)
    format_fn = await determine_card_id_format(api_set_id, 1)

    subtype_to_variant_flag = {
        'Normal': 'normal',
        'Reverse Holofoil': 'reverse',
        'Holofoil': 'holo',
        '1st Edition': 'firstEdition'
    }

    variant_map = {}
    card_id_to_extNumber = {}
    card_id_to_cardName = {}

    for row in df.itertuples(index=False):
        if pd.isna(row.extNumber):
            continue

        ext_number = str(row.extNumber).strip()
        card_number = ext_number.split("/")[0].lstrip("0")
        card_id = format_fn(card_number)

        subtype = str(row.subTypeName).strip()
        variant_flag = subtype_to_variant_flag.get(subtype)
        if variant_flag is None:
            print(f"Unrecognized subtype: {subtype}")

        if card_id not in variant_map:
            variant_map[card_id] = set()
        if variant_flag:
            variant_map[card_id].add(variant_flag)

        card_id_to_extNumber[card_id] = ext_number
        card_id_to_cardName[card_id] = str(row.name).strip()

    print(f"\nLoaded {len(variant_map)} cards with variant data from CSV.")

    connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
    async with aiohttp.ClientSession(connector=connector) as session:
        for card_id in sorted(variant_map.keys()):
            variants = variant_map[card_id]
            try:
                card = await sdk.card.get(card_id)
                ext_number = card_id_to_extNumber.get(card_id, "Unknown")
                card_name = card_id_to_cardName.get(card_id, "Unknown")
                if ' - ' in card_name:
                    card_name = card_name.split(' - ')[0]

                image_url = card.get_image_url(quality="high", extension="png")
                print(f"\nProcessing {card_name} ({ext_number}) → {card_id}")
                print(f"Image URL: {image_url}")

                for variant in variants:
                    safe_variant_name = variant.replace(' ', '_').lower()
                    filename = f"{card_id}_{safe_variant_name}.png"
                    gcs_path = f"{set_name_for_output_folder}/{filename}"
                    local_save_path = os.path.join("/tmp", filename)

                    # skip if already uploaded
                    if gcs_bucket.blob(gcs_path).exists():
                        print(f"⚠️ Exists in GCS, skipping: {gcs_path}")
                        continue

                    if not os.path.exists(local_save_path):
                        chosen_url = image_url or backup_image_lookup.get(card_name)
                        if chosen_url:
                            async with session.get(chosen_url) as resp:
                                if resp.status == 200:
                                    img_bytes = await resp.read()
                                    async with aiofiles.open(local_save_path, "wb") as f:
                                        await f.write(img_bytes)
                                else:
                                    print(f"⚠️ Failed to download image from {chosen_url}")
                                    continue
                        else:
                            print(f"⚠️ No image URL for {card_name}")
                            continue

                    await upload_to_gcs(local_save_path, gcs_path, gcs_bucket)

                    try:
                        os.remove(local_save_path)
                    except Exception:
                        pass

            except Exception as e:
                print(f"Failed to process card ID: {card_id}. Error: {str(e)}")

# --------- Sequential batch over selected sets ---------
async def run_set_queue(set_names: List[str], csv_root: str, gcs_bucket):
    sdk = TCGdex()
    total = len(set_names)
    for idx, set_name in enumerate(set_names, start=1):
        api_set_id = csv_to_api_set_id.get(set_name)
        if not api_set_id or api_set_id == "Not Supported":
            print(f"Skipping unsupported or missing mapping: {set_name}")
            continue

        csv_path = os.path.join(csv_root, f"{set_name}ProductsAndPrices.csv")
        if not os.path.isfile(csv_path):
            print(f"CSV not found: {csv_path} — skipping {set_name}")
            continue

        try:
            backup_image_lookup = load_backup_image_lookup(csv_path)
        except Exception as e:
            print(f"Failed to load backup CSV for {set_name}: {e} — continuing")
            backup_image_lookup = {}

        print(f"\n=== [{idx}/{total}] BEGIN {set_name} ({api_set_id}) ===")
        await process_csv_for_set(
            sdk=sdk,
            csv_path=csv_path,
            api_set_id=api_set_id,
            set_name_for_output_folder=set_name,
            backup_image_lookup=backup_image_lookup,
            gcs_bucket=gcs_bucket
        )
        print(f"=== [{idx}/{total}] END {set_name} ===")

# ---------------- UI THEME: Game Boy ----------------
# GB_DARK  = "#0f380f"
# GB_MED   = "#306230"
# GB_LIGHT = "#8bac0f"
# GB_PALE  = "#9bbc0f"

# ---------------- UI THEME: Poké Ball ----------------
# GB_DARK  = "#1a1a1a"
# GB_MED   = "#8b1a1a"
# GB_LIGHT = "#e33b2f"
# GB_PALE  = "#f7f5ed"

# ---------------- UI THEME: Team Rocket ----------------
# GB_DARK  = "#0d0d0f"
# GB_MED   = "#33333b"
# GB_LIGHT = "#b10f2e"
# GB_PALE  = "#dcdde1"

# --- Razz Berry theme ---
RZ_DARK  = "#56362d"  # outlines / text
RZ_MED   = "#285838"  # header / leaf
RZ_LIGHT = "#f08828"  # buttons / accents
RZ_PALE  = "#efeecc"  # main surface

def apply_razz_theme(root):
    import tkinter.font as tkfont
    from tkinter import ttk

    root.configure(bg=RZ_DARK)
    style = ttk.Style(root)
    style.theme_use("clam")

    fams = set(tkfont.families())
    PIXEL = "Press Start 2P" if "Press Start 2P" in fams else ("VT323" if "VT323" in fams else "Courier")
    FONT_UI = (PIXEL, 10)
    FONT_HEADER = (PIXEL, 12, "bold")

    # Panels / frames
    style.configure("GB.TFrame", background=RZ_PALE, borderwidth=4, relief="ridge")

    # Header bar
    style.configure("GB.Header.TLabel", background=RZ_MED, foreground=RZ_PALE,
                    font=FONT_HEADER, padding=(10,6))

    # Labels
    style.configure("GB.TLabel", background=RZ_PALE, foreground=RZ_DARK, font=FONT_UI)

    # Buttons
    style.configure("GB.TButton", background=RZ_LIGHT, foreground=RZ_PALE, font=FONT_UI,
                    padding=(8,4), borderwidth=3, relief="ridge")
    style.map("GB.TButton",
              background=[("active", RZ_LIGHT), ("pressed", RZ_MED)],
              relief=[("pressed", "sunken")])

    return FONT_UI, FONT_HEADER

def skin_text_razz(txt, font):
    txt.configure(bg=RZ_PALE, fg=RZ_DARK, insertbackground=RZ_DARK,
                  highlightbackground=RZ_DARK, highlightthickness=2, font=font)

def skin_listbox_razz(lb, font):
    lb.configure(bg=RZ_PALE, fg=RZ_DARK,
                 selectbackground=RZ_LIGHT, selectforeground=RZ_PALE,
                 highlightbackground=RZ_DARK, highlightthickness=2,
                 relief="flat", font=font)

# ---------------- GUI (listbox multi-select) ----------------
class QueueWriter:
    def __init__(self, q: queue.Queue): self.q = q
    def write(self, s): 
        if s: self.q.put(s)
    def flush(self): pass

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        # self.title("RAZZ BERRY • Set Uploader")
        # self.geometry("840x600")
        # self.resizable(False, False)


        self.title("RAZZ BERRY • Set Uploader")
        self.geometry("840x600")
        self.resizable(False, False)

        # 0) DEFINE THE ATTRIBUTE BEFORE ANY USE
        self.logo_img = None

        # 1) Load the berry image (PNG/GIF) and keep a reference on the instance
        RAZZ_ICON_PATH = "/Users/ethandessner/dev/Pokemon-Card-Scanner/images/razz_iso.png"
        try:
            self.logo_img = tk.PhotoImage(master=self, file=RAZZ_ICON_PATH)
        except Exception as e:
            print("logo load failed:", e)       

        # 3) logging setup (your existing code) ...
        self.log_queue = queue.Queue()
        self.stdout_backup = sys.stdout
        sys.stdout = QueueWriter(self.log_queue)

        # 4) theme
        ui_font, header_font = apply_razz_theme(self)

        # 5) main "screen" frame
        self.screen = ttk.Frame(self, style="GB.TFrame", padding=10)
        self.screen.place(relx=0.5, rely=0.5, anchor="center", width=800, height=560)

        # 6) header WITH the berry image inside the UI
        hdr = ttk.Frame(self.screen, style="GB.TFrame")
        hdr.grid(row=0, column=0, columnspan=3, sticky="we", padx=4, pady=(0, 8))

# break
        if self.logo_img:
            ttk.Label(hdr, image=self.logo_img, style="GB.TLabel").pack(side="left", padx=(2, 8))
        ttk.Label(hdr, text="RAZZ BERRY ▸ SET UPLOADER", style="GB.Header.TLabel").pack(side="left")
        # info labels
        ttk.Label(self.screen, text=f"Bucket: {GCS_BUCKET_NAME}", style="GB.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(self.screen, text=f"CSV: {CSV_FOLDER}", style="GB.TLabel").grid(row=1, column=1, columnspan=2, sticky="w")

        # listbox multi-select (same behavior as earlier working GUI)
        ttk.Label(self.screen, text="Sets:", style="GB.TLabel").grid(row=2, column=0, sticky="ne", padx=(0,8), pady=(6,6))
        self.listbox = tk.Listbox(self.screen, selectmode=tk.EXTENDED, width=40, height=14, exportselection=False)
        for name in csv_to_api_set_id.keys():
            self.listbox.insert(tk.END, name)
        self.listbox.grid(row=2, column=1, sticky="w", pady=(6,6))
        skin_listbox_razz(self.listbox, ui_font)

        # side buttons (Select All / Clear / Start / Quit)
        side = ttk.Frame(self.screen, style="GB.TFrame")
        side.grid(row=2, column=2, sticky="n", padx=(12,0), pady=(6,6))
        self.btn_all = ttk.Button(side, text="SELECT ALL", style="GB.TButton", command=self._select_all)
        self.btn_all.pack(fill="x", pady=4)
        self.btn_clear = ttk.Button(side, text="CLEAR", style="GB.TButton", command=self._clear_sel)
        self.btn_clear.pack(fill="x", pady=4)
        self.run_btn = ttk.Button(side, text="START", style="GB.TButton", command=self._run_clicked)
        self.run_btn.pack(fill="x", pady=12)
        self.exit_btn = ttk.Button(side, text="QUIT", style="GB.TButton", command=self._exit)
        self.exit_btn.pack(fill="x", pady=4)

        # log
        ttk.Label(self.screen, text="LOG:", style="GB.TLabel").grid(row=3, column=0, sticky="w", pady=(10,2))
        self.log_box = tk.Text(self.screen, height=14, width=96)
        self.log_box.grid(row=4, column=0, columnspan=3, sticky="we")
        skin_text_razz(self.log_box, ui_font)

        # grid stretch
        self.screen.grid_columnconfigure(1, weight=1)

        # gcs bucket
        try:
            self.gcs_bucket = storage.Client().bucket(GCS_BUCKET_NAME)
        except Exception as e:
            messagebox.showerror("GCS Error", f"Failed to init GCS for '{GCS_BUCKET_NAME}': {e}")
            self.gcs_bucket = None

        self.after(100, self._drain_log)

    # listbox helpers
    def _select_all(self):
        self.listbox.select_set(0, tk.END)

    def _clear_sel(self):
        self.listbox.selection_clear(0, tk.END)

    # logging
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

    # run
    def _run_clicked(self):
        if self.gcs_bucket is None:
            messagebox.showerror("GCS Error", "Bucket not initialized.")
            return
        if not os.path.isdir(CSV_FOLDER):
            messagebox.showerror("CSV Error", f"CSV folder missing:\n{CSV_FOLDER}")
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
                    await run_set_queue(sets, CSV_FOLDER, self.gcs_bucket)
                    self.after(0, lambda: messagebox.showinfo("Done", "Finished uploading selected sets."))
                except Exception as e:
                    self.after(0, lambda exc=e: messagebox.showerror("Error", str(exc)))
                finally:
                    self.after(0, lambda: self.run_btn.state(["!disabled"]))
            asyncio.run(runner())

        threading.Thread(target=worker, daemon=True).start()

    def _exit(self):
        try:
            sys.stdout = self.stdout_backup
        except Exception:
            pass
        self.destroy()

if __name__ == "__main__":
    ttk.Style().theme_use("clam")
    App().mainloop()
