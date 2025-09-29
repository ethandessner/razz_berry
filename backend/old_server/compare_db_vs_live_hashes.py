import os
import sys
import requests
from dotenv import load_dotenv
from supabase import create_client
from PIL import Image
import imagehash
import io

# Import your hash normalization from backend/app.py
from app import prepare_card_for_hash, _hx

load_dotenv()
SUPABASE_URL   = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY   = os.getenv("SUPABASE_KEY", "").strip()
TABLE          = os.getenv("SUPABASE_TABLE", "destinedrivals")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Set SUPABASE_URL and SUPABASE_KEY in your .env file.")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_card_row(card_id):
    resp = supabase.table(TABLE).select("*").eq("card_id", card_id).execute()
    data = resp.data or []
    if not data:
        print(f"Card {card_id} not found in DB.")
        sys.exit(1)
    return data[0]

def download_image(url):
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to download image: {url}")
        sys.exit(1)
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def print_hashes(label, hashes):
    print(f"{label}:")
    for name, h in hashes.items():
        print(f"  {name}: {h}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_db_vs_live_hashes.py <card_id>")
        sys.exit(1)
    card_id = sys.argv[1]
    row = fetch_card_row(card_id)
    img_url = row["image_path"]
    print(f"Card: {row['name']} | {card_id}")
    print(f"Image URL: {img_url}")
    img = download_image(img_url)
    # Compute live hashes
    norm_img = prepare_card_for_hash(img, max_side=900)
    hashes_live = {
        "avghashes": str(imagehash.average_hash(norm_img)),
        "whashes":   str(imagehash.whash(norm_img)),
        "phashes":   str(imagehash.phash(norm_img)),
        "dhashes":   str(imagehash.dhash(norm_img)),
    }
    # DB hashes
    hashes_db = {
        "avghashes": row["avghashes"],
        "whashes":   row["whashes"],
        "phashes":   row["phashes"],
        "dhashes":   row["dhashes"],
    }
    print_hashes("DB Hashes", hashes_db)
    print_hashes("Live Hashes", hashes_live)
    # Compare
    print("Hash differences (Hamming distance):")
    for k in hashes_db:
        db_h = _hx(hashes_db[k])
        live_h = _hx(hashes_live[k])
        print(f"  {k}: {db_h - live_h}")

if __name__ == "__main__":
    main()
