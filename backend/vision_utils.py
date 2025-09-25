import numpy as np
import cv2
from typing import Optional, Tuple

# card image size
CARD_W, CARD_H = 330, 440  # (width, height)

# Biggest-quad filter
MIN_AREA = 5000

def biggestContour_model(contours):
    biggest = np.array([])
    maxArea = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_AREA:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest, maxArea

def _sort_vals(vals):
    idxs = list(range(len(vals)))
    for i in range(len(vals)):
        index = i
        minval = vals[i]
        for j in range(i, len(vals)):
            if vals[j] < minval:
                minval = vals[j]; index = j
        vals[i], vals[index] = vals[index], vals[i]
        idxs[i], idxs[index] = idxs[index], idxs[i]
    return vals, idxs

def reorderCorners_model(corners):
    # corners: [[x0,y0]], [[x1,y1]], [[x2,y2]], [[x3,y3]]
    xvals = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]]
    yvals = [corners[0][1], corners[1][1], corners[2][1], corners[3][1]]

    yvals, idxs = _sort_vals(yvals)
    temp = xvals.copy()
    for i in range(len(idxs)):
        xvals[i] = temp[idxs[i]]

    # normalize order to top-left first
    if yvals[0] == yvals[1] and xvals[1] < xvals[0]:
        xvals[0], xvals[1] = xvals[1], xvals[0]

    import math
    dist1 = math.dist((xvals[1], yvals[1]), (xvals[0], yvals[0]))
    dist2 = math.dist((xvals[2], yvals[2]), (xvals[0], yvals[0]))
    dist3 = math.dist((xvals[3], yvals[3]), (xvals[0], yvals[0]))
    dists = [dist1, dist2, dist3]
    dSorted, idxsDist = _sort_vals(dists.copy())
    idxsDist.insert(0, 0)
    idxsDist[1] += 1; idxsDist[2] += 1; idxsDist[3] += 1

    # tilt handling 
    # also random note - the crobat card does better when captured at a slight tilt
    if yvals[0] == yvals[1]:
        if dists[0] == dSorted[0]:
            topleft    = [xvals[idxsDist[0]], yvals[idxsDist[0]]]
            topright   = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            bottomright= [xvals[idxsDist[3]], yvals[idxsDist[3]]]
            bottomleft = [xvals[idxsDist[2]], yvals[idxsDist[2]]]
        else:
            topleft    = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            topright   = [xvals[idxsDist[0]], yvals[idxsDist[0]]]
            bottomright= [xvals[idxsDist[2]], yvals[idxsDist[2]]]
            bottomleft = [xvals[idxsDist[3]], yvals[idxsDist[3]]]
    else:
        if xvals[idxsDist[1]] == min(xvals):
            topleft    = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            topright   = [xvals[idxsDist[0]], yvals[idxsDist[0]]]
            bottomright= [xvals[idxsDist[2]], yvals[idxsDist[2]]]
            bottomleft = [xvals[idxsDist[3]], yvals[idxsDist[3]]]
        else:
            topleft    = [xvals[idxsDist[0]], yvals[idxsDist[0]]]
            topright   = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            bottomright= [xvals[idxsDist[3]], yvals[idxsDist[3]]]
            bottomleft = [xvals[idxsDist[2]], yvals[idxsDist[2]]]

    return [[topleft],[topright],[bottomleft],[bottomright]]

def drawRectangle_model(img, corners):
    thickness = 10
    cv2.line(img, tuple(corners[0][0]), tuple(corners[1][0]), (0,255,0), thickness)
    cv2.line(img, tuple(corners[0][0]), tuple(corners[2][0]), (0,255,0), thickness)
    cv2.line(img, tuple(corners[3][0]), tuple(corners[2][0]), (0,255,0), thickness)
    cv2.line(img, tuple(corners[3][0]), tuple(corners[1][0]), (0,255,0), thickness)
    return img

def segment_card_model(bgr_image: np.ndarray):
    """
    Model-project parity:
      gray → GaussianBlur(3x3) → Canny(100,200) → dilate→erode →
      biggestContour_model (4-gon) → reorderCorners_model → warp to (CARD_W,CARD_H)
    Returns (warped_bgr, gray, blur, edges, bigContourImg) for montage.
    """
    H, W = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blur, threshold1=100, threshold2=200)

    k = np.ones((5,5), np.uint8)
    dial = cv2.dilate(edges, k, iterations=2)
    thr  = cv2.erode(dial, k, iterations=1)

    contourFrame = bgr_image.copy()
    bigContour   = bgr_image.copy()
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contourFrame, contours, -1, (0,255,0), 10)

    warped = None
    corners, maxArea = biggestContour_model(contours)
    if len(corners) == 4:
        corners = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]]
        corners = reorderCorners_model(corners)
        bigContour = drawRectangle_model(bigContour, corners)

        pts1 = np.float32(corners)
        pts2 = np.float32([[0,0],[CARD_W,0],[0,CARD_H],[CARD_W,CARD_H]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(bgr_image, M, (CARD_W, CARD_H))

    # Return intermediates for 2×4 montage
    return warped, gray, blur, edges, bigContour

def _label(img, text):
    # white band on top and black label
    h, w = img.shape[:2]
    bar = np.full((32, w, 3), 255, np.uint8)
    out = np.vstack([bar, img if img.ndim==3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)])
    cv2.putText(out, text, (10, 23), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 2)
    return out

def build_montage(src=None, gray=None, blur=None, edges=None, contour=None, warped=None, matching=None):
    """
    Stack up to 8 images in a 2×4 grid with labels, following the model demo.
    Any missing panels are replaced with blanks.
    """
    W, H = CARD_W, CARD_H
    blank = np.zeros((H, W, 3), np.uint8)

    def _nz(x): 
        if x is None: return blank
        xx = x
        if xx.ndim==2: xx = cv2.cvtColor(xx, cv2.COLOR_GRAY2BGR)
        return cv2.resize(xx, (W, H))
    r0 = [_label(_nz(src)     , "Original"),
          _label(_nz(gray)    , "Gray"),
          _label(_nz(blur)    , "Blurred"),
          _label(_nz(edges)   , "Edges")]
    r1 = [_label(_nz(contour) , "Contours"),
          _label(_nz(contour) , "Biggest Contour"),
          _label(_nz(warped)  , "Warped Perspective"),
          _label(_nz(matching), "Matching Card")]
    top = np.hstack(r0); bot = np.hstack(r1)
    return np.vstack([top, bot])
