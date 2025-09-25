# gui_upload_to_gcs_multiset_min.py
# source ~/venvs/tk/bin/activate 
import os, sys, asyncio, threading, queue, re
from typing import Dict, List

import pandas as pd
import aiohttp
import aiofiles
from google.cloud import storage
from tcgdexsdk import TCGdex
import ssl, certifi
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# ====== HARD-CODED CONFIG (edit here if needed) ======
CSV_ROOT_DEFAULT = "/Users/ethandessner/Desktop/PokemonCSVs"
GCS_BUCKET_NAME = "razz_berry"   # hard-coded as you asked
# =====================================================

# Your mapping (unchanged)
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

# ---------- Original helpers (unchanged) ----------
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

import ssl, certifi
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

async def determine_card_id_format(set_id, test_card_number):
    formats_to_try = [
        f"{set_id}-{test_card_number}",
        f"{set_id}-{str(test_card_number).zfill(2)}",
        f"{set_id}-{str(test_card_number).zfill(3)}"
    ]
    connector = aiohttp.TCPConnector(ssl=SSL_CONTEXT)
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

async def process_csv_for_set(sdk, csv_path, api_set_id, set_name_for_output_folder, backup_image_lookup, gcs_bucket):
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
        if card_id not in variant_map:
            variant_map[card_id] = set()
        if variant_flag:
            variant_map[card_id].add(variant_flag)
        card_id_to_extNumber[card_id] = ext_number
        card_id_to_cardName[card_id] = str(row.name).strip()

    connector = aiohttp.TCPConnector(ssl=SSL_CONTEXT)
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

async def upload_to_gcs(local_path: str, gcs_path: str, gcs_bucket):
    blob = gcs_bucket.blob(gcs_path)
    if not blob.exists():
        blob.upload_from_filename(local_path)
        print(f"✅ Uploaded to GCS: {gcs_path}")
    else:
        print(f"⚠️ GCS already contains: {gcs_path}")


# ---------- Batch runner (sequential sets) ----------
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
            set_name_for_output_folder=set_name,  # folder name in GCS (unchanged)
            backup_image_lookup=backup_image_lookup,
            gcs_bucket=gcs_bucket
        )
        print(f"=== [{idx}/{total}] END {set_name} ===")

# ================== Tkinter GUI ==================
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class QueueWriter:
    def __init__(self, q: queue.Queue): self.q = q
    def write(self, s): 
        if s: self.q.put(s)
    def flush(self): pass

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Upload TCGdex Images → GCS (multi-set; no bucket/folder inputs)")
        self.geometry("780x520")
        self.resizable(True, True)

        # state
        self.csv_root_var = tk.StringVar(value=CSV_ROOT_DEFAULT)

        # logging
        self.log_queue = queue.Queue()
        self.stdout_backup = sys.stdout
        sys.stdout = QueueWriter(self.log_queue)

        # GCS init (hard-coded bucket)
        try:
            self.gcs_bucket = storage.Client().bucket(GCS_BUCKET_NAME)
        except Exception as e:
            messagebox.showerror("GCS Error", f"Failed to init GCS client for bucket '{GCS_BUCKET_NAME}': {e}")
            self.gcs_bucket = None

        self._build()
        self.after(100, self._drain_log)

    def _build(self):
        pad = {"padx": 10, "pady": 6}
        r = 0

        ttk.Label(self, text=f"GCS bucket: {GCS_BUCKET_NAME} (hard-coded)").grid(row=r, column=0, columnspan=2, sticky="w", **pad)

        r += 1
        ttk.Label(self, text="CSV root folder:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.csv_root_var, width=56).grid(row=r, column=1, sticky="w", **pad)
        ttk.Button(self, text="Browse…", command=self._pick_csv_root).grid(row=r, column=2, sticky="w", **pad)

        r += 1
        ttk.Label(self, text="Select one or more sets:").grid(row=r, column=0, sticky="w", **pad)
        self.listbox = tk.Listbox(self, selectmode=tk.EXTENDED, width=48, height=12)
        for name in csv_to_api_set_id.keys():
            self.listbox.insert(tk.END, name)
        self.listbox.grid(row=r, column=1, sticky="w", **pad)

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=r, column=2, sticky="nsw", **pad)
        ttk.Button(btn_frame, text="Select All", command=self._select_all).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Clear", command=self._clear_sel).pack(fill="x", pady=2)

        r += 1
        self.run_btn = ttk.Button(self, text="Upload Selected Sets → GCS", command=self._run_clicked)
        self.run_btn.grid(row=r, column=0, sticky="w", **pad)
        ttk.Button(self, text="Exit", command=self._exit).grid(row=r, column=2, sticky="e", **pad)

        r += 1
        ttk.Label(self, text="Log:").grid(row=r, column=0, sticky="nw", **pad)
        self.log_box = tk.Text(self, height=12, width=92, state="disabled")
        self.log_box.grid(row=r, column=0, columnspan=3, sticky="w", padx=10, pady=(0,10))

    def _pick_csv_root(self):
        path = filedialog.askdirectory(title="Select CSV folder")
        if path:
            self.csv_root_var.set(path)

    def _select_all(self):
        self.listbox.select_set(0, tk.END)

    def _clear_sel(self):
        self.listbox.selection_clear(0, tk.END)

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
        if self.gcs_bucket is None:
            messagebox.showerror("GCS Error", "Bucket not initialized.")
            return

        csv_root = self.csv_root_var.get().strip()
        if not os.path.isdir(csv_root):
            messagebox.showwarning("CSV folder", "CSV folder does not exist.")
            return

        sel_indices = self.listbox.curselection()
        if not sel_indices:
            messagebox.showwarning("Select sets", "Pick at least one set in the list.")
            return
        sets = [self.listbox.get(i) for i in sel_indices]

        self.run_btn.configure(state="disabled")

        def worker():
            async def runner():
                try:
                    await run_set_queue(sets, csv_root, self.gcs_bucket)
                    self.after(0, lambda: messagebox.showinfo("Done", "Finished uploading selected sets."))
                except Exception as e:
                    self.after(0, lambda exc=e: messagebox.showerror("Error", str(exc)))
                finally:
                    self.after(0, lambda: self.run_btn.configure(state="normal"))
            asyncio.run(runner())

        threading.Thread(target=worker, daemon=True).start()

    def _exit(self):
        try:
            sys.stdout = self.stdout_backup
        except Exception:
            pass
        self.destroy()

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import ttk
    ttk.Style().theme_use("clam")
    App().mainloop()
