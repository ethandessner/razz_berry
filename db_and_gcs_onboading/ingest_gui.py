# ingest_gui.py
import os
import io
import sys
import json
import asyncio
import threading
from datetime import datetime
from typing import List, Dict, Any

import tkinter as tk
print("ok")
from tkinter import ttk, messagebox

import aiohttp
from PIL import Image
from supabase import create_client, Client
from tcgdexsdk import TCGdex
import imagehash

# ========= CONFIG =========
# Set these as env vars before running:
#   export SUPABASE_URL=...
#   export SUPABASE_SERVICE_ROLE_KEY=...
# Optional:
#   export TCGDEX_LANG=en
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
DEFAULT_LANG = os.environ.get("TCGDEX_LANG", "en")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in your environment.")
    sys.exit(1)

# ========= SUPABASE CLIENT =========
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ========= TCGdex HELPERS =========
# def build_tcgdex(lang: str) -> TCGdex:
#     return TCGdex(lang or DEFAULT_LANG)

dex = TCGdex()

def tcgdex_get_sets(dex: TCGdex) -> List[Dict[str, Any]]:
    """
    Returns list of {id, name, ...} for sets in the given language.
    """
    items = []
    for s in dex.sets():
        # hydrate to get nice fields
        full = dex.set(s.id)
        items.append({
            "id": full.id,
            "name": full.name,
            "series": getattr(full, "serie", None) or getattr(full, "series", None),
            "printed_total": getattr(full, "printedTotal", None),
            "release_date": getattr(full, "releaseDate", None),
        })
    # sort by release date (if available), newest first
    def rd(x):
        try:
            return datetime.fromisoformat(x["release_date"])
        except Exception:
            return datetime.min
    items.sort(key=rd, reverse=True)
    return items

# ========= HASHING =========
def four_orientations(im: Image.Image):
    # normal, mirrored, upside_down, upside_down_mirrored
    im_norm = im
    im_mir = im.transpose(Image.FLIP_LEFT_RIGHT)
    im_ud  = im.transpose(Image.ROTATE_180)
    im_udm = im_ud.transpose(Image.FLIP_LEFT_RIGHT)
    return [im_norm, im_mir, im_ud, im_udm]

def compute_hashes(im: Image.Image):
    ims = four_orientations(im)
    avg_hashes = [str(imagehash.average_hash(x)) for x in ims]
    w_hashes   = [str(imagehash.whash(x))        for x in ims]
    p_hashes   = [str(imagehash.phash(x))        for x in ims]
    d_hashes   = [str(imagehash.dhash(x))        for x in ims]
    return avg_hashes, w_hashes, p_hashes, d_hashes

async def fetch_image_bytes(session: aiohttp.ClientSession, url: str) -> bytes:
    async with session.get(url, timeout=30) as r:
        r.raise_for_status()
        return await r.read()

# ========= INGESTION CORE =========
def upsert_sets_in_db(lang: str, sets_payload: List[Dict[str, Any]]):
    # add lang column
    for p in sets_payload:
        p["lang"] = lang
    supabase.table("sets").upsert(sets_payload, on_conflict="id").execute()

def upsert_cards_in_db(cards_payload: List[Dict[str, Any]]):
    supabase.table("cards").upsert(cards_payload, on_conflict="id").execute()

def upsert_hashes_in_db(hash_rows: List[Dict[str, Any]]):
    if hash_rows:
        supabase.table("card_hashes").upsert(hash_rows, on_conflict="card_id").execute()

async def ingest_set(lang: str, set_id: str, compute_hashes_flag: bool, log):
    """
    Pull cards for set_id in `lang`, write to Supabase, optionally download images & compute hashes.
    """
    log(f"Starting ingestion for set: {set_id} (lang={lang})")
    # dex = build_tcgdex(lang)

    # 1) Ensure set exists in DB (upsert)
    full_set = dex.set(set_id)
    upsert_sets_in_db(lang, [{
        "id": full_set.id,
        "name": full_set.name,
        "series": getattr(full_set, "serie", None) or getattr(full_set, "series", None),
        "printed_total": getattr(full_set, "printedTotal", None),
        "release_date": getattr(full_set, "releaseDate", None),
    }])

    # 2) Fetch cards for this set
    # Depending on tcgdexsdk version; try common patterns:
    try:
        cards_refs = dex.cards_in_set(set_id)
    except Exception:
        # fallback approach if API differs
        cards_refs = getattr(full_set, "cards", [])

    card_payload = []
    for ref in cards_refs:
        full_card = dex.card(ref.id if hasattr(ref, "id") else ref)
        card_payload.append({
            "id": full_card.id,
            "set_id": set_id,
            "number": full_card.number,
            "name": full_card.name,
            "supertype": getattr(full_card, "supertype", None),
            "subtypes": getattr(full_card, "subtypes", None) or [],
            "rarity": getattr(full_card, "rarity", None),
            "types": getattr(full_card, "types", None) or [],
            "images": getattr(full_card, "images", None) or {},
            "tcgplayer": getattr(full_card, "tcgplayer", None) or {},
        })

    log(f"Upserting {len(card_payload)} cards …")
    upsert_cards_in_db(card_payload)

    if not compute_hashes_flag:
        log("Skipping perceptual-hash computation (unchecked). Done.")
        return

    # 3) Download images & compute hashes
    rows = supabase.table("cards").select("id, images").eq("set_id", set_id).execute().data
    if not rows:
        log("No cards found post-insert; done.")
        return

    hash_rows = []
    total = len(rows)
    log(f"Computing perceptual hashes for {total} cards …")

    async with aiohttp.ClientSession() as session:
        for i, row in enumerate(rows, start=1):
            images = row["images"] or {}
            url = images.get("large") or images.get("small")
            if not url:
                log(f"[{i}/{total}] {row['id']}: no image URL")
                continue
            try:
                raw = await fetch_image_bytes(session, url)
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                avg, w, p, d = compute_hashes(im)
                hash_rows.append({
                    "card_id": row["id"],
                    "avg_hashes": avg,
                    "w_hashes":   w,
                    "p_hashes":   p,
                    "d_hashes":   d
                })
                if i % 25 == 0:
                    upsert_hashes_in_db(hash_rows)
                    log(f"  → wrote {len(hash_rows)} rows (batch)")
                    hash_rows.clear()
                else:
                    log(f"[{i}/{total}] {row['id']}: hashed")
            except Exception as e:
                log(f"[warn] {row['id']}: {e}")

    if hash_rows:
        upsert_hashes_in_db(hash_rows)
        log(f"Final batch wrote {len(hash_rows)} rows.")
    log("Ingestion complete.")

# ========= TKINTER GUI =========
class IngestGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TCGdex → Supabase Ingest")
        self.geometry("700x460")
        self.resizable(False, False)

        # State
        self.lang_var = tk.StringVar(value=DEFAULT_LANG)
        self.sets: List[Dict[str, Any]] = []
        self.selected_set_id = tk.StringVar(value="")
        self.compute_hashes_var = tk.BooleanVar(value=True)

        # Widgets
        self._build_widgets()

        # Load sets initially
        self._load_sets_async()

    def _build_widgets(self):
        pad = {"padx": 10, "pady": 6}

        row = 0
        ttk.Label(self, text="Language (e.g., en, fr, ja):").grid(row=row, column=0, sticky="w", **pad)
        self.lang_entry = ttk.Entry(self, textvariable=self.lang_var, width=8)
        self.lang_entry.grid(row=row, column=1, sticky="w", **pad)
        self.refresh_btn = ttk.Button(self, text="Refresh Sets", command=self._load_sets_async)
        self.refresh_btn.grid(row=row, column=2, sticky="w", **pad)

        row += 1
        ttk.Label(self, text="Select Set:").grid(row=row, column=0, sticky="w", **pad)
        self.set_combo = ttk.Combobox(self, state="readonly", width=60)
        self.set_combo.grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        self.set_combo.bind("<<ComboboxSelected>>", self._on_set_selected)

        row += 1
        self.hash_chk = ttk.Checkbutton(self, text="Compute perceptual hashes (avg/whash/phash/dhash)",
                                        variable=self.compute_hashes_var)
        self.hash_chk.grid(row=row, column=0, columnspan=3, sticky="w", **pad)

        row += 1
        self.ingest_btn = ttk.Button(self, text="Ingest Selected Set", command=self._ingest_selected)
        self.ingest_btn.grid(row=row, column=0, sticky="w", **pad)

        self.stop_btn = ttk.Button(self, text="Exit", command=self.destroy)
        self.stop_btn.grid(row=row, column=2, sticky="e", **pad)

        row += 1
        ttk.Label(self, text="Log:").grid(row=row, column=0, sticky="nw", **pad)
        self.log_text = tk.Text(self, height=16, width=85, state="disabled")
        self.log_text.grid(row=row, column=0, columnspan=3, sticky="w", padx=10, pady=(0,10))

    def log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.update_idletasks()

    def _on_set_selected(self, _evt=None):
        sel = self.set_combo.get()
        # we format entries as "name (id) – series"
        if "(" in sel and ")" in sel:
            self.selected_set_id.set(sel.split("(")[-1].split(")")[0])

    def _load_sets_async(self):
        self.ingest_btn.configure(state="disabled")
        self.refresh_btn.configure(state="disabled")
        self.log("Loading sets from TCGdex…")

        def worker():
            try:
                # dex = build_tcgdex(self.lang_var.get().strip())
                sets = tcgdex_get_sets(dex)
                self.sets = sets
                # build UI-friendly labels
                labels = [
                    f"{s['name']} ({s['id']}) – {s.get('series') or ''}".strip()
                    for s in sets
                ]
                def done():
                    self.set_combo["values"] = labels
                    if labels:
                        self.set_combo.current(0)
                        self._on_set_selected()
                    self.log(f"Loaded {len(labels)} sets.")
                    self.ingest_btn.configure(state="normal")
                    self.refresh_btn.configure(state="normal")
                self.after(0, done)
            except Exception as e:
                def err():
                    messagebox.showerror("Error", f"Failed to load sets: {e}")
                    self.log(f"[error] {e}")
                    self.ingest_btn.configure(state="normal")
                    self.refresh_btn.configure(state="normal")
                self.after(0, err)

        threading.Thread(target=worker, daemon=True).start()

    def _ingest_selected(self):
        set_id = self.selected_set_id.get().strip()
        if not set_id:
            messagebox.showwarning("Select a set", "Please pick a set from the dropdown.")
            return
        self.ingest_btn.configure(state="disabled")
        self.refresh_btn.configure(state="disabled")

        lang = self.lang_var.get().strip()
        do_hashes = self.compute_hashes_var.get()

        def run_async():
            async def runner():
                try:
                    await ingest_set(lang, set_id, do_hashes, self.log)
                    self.after(0, lambda: messagebox.showinfo("Done", f"Ingested set {set_id}"))
                except Exception as e:
                    self.after(0, lambda: messagebox.showerror("Error", str(e)))
                finally:
                    self.after(0, lambda: (self.ingest_btn.configure(state="normal"),
                                           self.refresh_btn.configure(state="normal")))
            asyncio.run(runner())

        threading.Thread(target=run_async, daemon=True).start()

if __name__ == "__main__":
    app = IngestGUI()
    app.mainloop()
