from dataclasses import dataclass
from typing import List, Optional
import imagehash

@dataclass
class CardRow:
    card_id: str
    name: str
    set_name: str
    ext_number: str
    subtype_name: str
    image_path: str
    avg:  List[imagehash.ImageHash]  # [n, mir, ud, udmir]
    wh:   List[imagehash.ImageHash]
    ph:   List[imagehash.ImageHash]
    dh:   List[imagehash.ImageHash]

def _hx(s: str) -> imagehash.ImageHash:
    return imagehash.hex_to_hash(str(s))

def fetch_all_cards(SB, table: str, set_filter: Optional[str]) -> List[CardRow]:
    rows: List[CardRow] = []
    page, off = 1000, 0
    while True:
        q = SB.table(table).select(
            "card_id,name,set_name,ext_number,subtype_name,image_path,"
            "avghashes,avghashesmir,avghashesud,avghashesudmir,"
            "whashes,whashesmir,whashesud,whashesudmir,"
            "phashes,phashesmir,phashesud,phashesudmir,"
            "dhashes,dhashesmir,dhashesud,dhashesudmir"
        ).range(off, off+page-1)
        if set_filter:
            q = q.eq("set_name", set_filter)
        data = q.execute().data or []
        if not data:
            break
        for r in data:
            try:
                rows.append(CardRow(
                    card_id=r["card_id"],
                    name=r["name"],
                    set_name=r["set_name"],
                    ext_number=r["ext_number"],
                    subtype_name=r["subtype_name"],
                    image_path=r["image_path"],
                    avg=[_hx(r["avghashes"]), _hx(r["avghashesmir"]), _hx(r["avghashesud"]), _hx(r["avghashesudmir"])],
                    wh =[ _hx(r["whashes"]),  _hx(r["whashesmir"]),  _hx(r["whashesud"]),  _hx(r["whashesudmir"])],
                    ph =[ _hx(r["phashes"]),  _hx(r["phashesmir"]),  _hx(r["phashesud"]),  _hx(r["phashesudmir"])],
                    dh =[ _hx(r["dhashes"]),  _hx(r["dhashesmir"]),  _hx(r["dhashesud"]),  _hx(r["dhashesudmir"])],
                ))
            except Exception:
                # skip malformed row
                pass
        off += page
        if len(data) < page: break
    return rows
