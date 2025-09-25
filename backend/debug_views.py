from fastapi import APIRouter
from fastapi.responses import HTMLResponse, Response
import threading
import cv2

router = APIRouter()

_LAST_MONTAGE_JPG = None
_LOCK = threading.Lock()

def update_last_montage_jpg(bgr_montage):
    """Compress and store the last 2×4 montage as JPEG for the desktop view."""
    if bgr_montage is None:
        return
    ok, buf = cv2.imencode(".jpg", bgr_montage, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if ok:
        with _LOCK:
            global _LAST_MONTAGE_JPG
            _LAST_MONTAGE_JPG = bytes(buf)

@router.get("/debug", response_class=HTMLResponse)
def debug_page():
    return """
<!doctype html>
<title>Razz Debug</title>
<style>body{margin:0;background:#111;color:#eee}img{max-width:100vw;height:auto;display:block;margin:auto}</style>
<h3 style="padding:10px">Last Scan Montage</h3>
<img src="/debug/montage.jpg?nocache=1">
"""

@router.get("/debug/montage.jpg")
def debug_montage():
    with _LOCK:
        if _LAST_MONTAGE_JPG is None:
            return Response(content=b"", media_type="image/jpeg")
        return Response(content=_LAST_MONTAGE_JPG, media_type="image/jpeg")
