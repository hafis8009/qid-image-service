from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import numpy as np
import io, base64

app = FastAPI()
# CORS open for testing; restrict later if you want
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

def pil_to_b64(img, quality=92):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def grayscale_np(img):
    g = ImageOps.grayscale(img)
    return np.array(g, dtype=np.float32)

def smooth_1d(x, win):
    if win <= 1: return x
    k = np.ones(win, dtype=np.float32)/win
    return np.convolve(x, k, mode="same")

def find_split_y(img_rgb):
    """Find split between two vertically stacked cards using row brightness valley."""
    g = grayscale_np(img_rgb)  # HxW
    h, w = g.shape
    if h < 200 or w < 200:
        return None

    # row mean brightness (0..255)
    row = g.mean(axis=1)
    # normalize 0..1 and smooth
    denom = row.max() - row.min()
    row_n = (row - row.min()) / (denom + 1e-6)
    win = max(5, int(h * 0.02))
    sm = smooth_1d(row_n, win)

    # search central band
    y0, y1 = int(h*0.30), int(h*0.70)
    yi = y0 + int(np.argmin(sm[y0:y1]))

    # validate halves and contrast
    topH, botH = yi, h - yi
    if topH < h*0.25 or botH < h*0.25:
        return None
    pad = max(5, int(h*0.02))
    topPeak = sm[:max(yi-pad, 1)].max() if yi - pad > 1 else 0
    botPeak = sm[min(yi+pad, h-1):].max() if yi + pad < h else 0
    valley = sm[yi]
    # Need a real valley between two bright bands
    if valley < 0.5*min(topPeak, botPeak) and valley < 0.4:
        return yi
    return None

def safe_trim(img_rgb):
    """Trim outer margins by detecting background (black or white), add tiny safety margin."""
    g = np.array(ImageOps.grayscale(img_rgb), dtype=np.uint8)
    h, w = g.shape
    if h < 8 or w < 8:
        return img_rgb

    corners = [g[0,0], g[0,-1], g[-1,0], g[-1,-1]]
    bg_black = (np.mean(corners) < 96)
    if bg_black:
        mask = g <= 40   # black background
    else:
        mask = g >= 230  # white background

    row_bg = (mask.sum(axis=1)/w) >= 0.97
    col_bg = (mask.sum(axis=0)/h) >= 0.97

    top = 0
    while top < h-1 and row_bg[top]: top += 1
    bottom = h-1
    while bottom > top and row_bg[bottom]: bottom -= 1
    left = 0
    while left < w-1 and col_bg[left]: left += 1
    right = w-1
    while right > left and col_bg[right]: right -= 1

    mY = max(2, h//100)
    mX = max(2, w//100)
    top = max(0, top - mY)
    bottom = min(h-1, bottom + mY)
    left = max(0, left - mX)
    right = min(w-1, right + mX)

    return img_rgb.crop((left, top, right+1, bottom+1))

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/process")
async def process(
    file: UploadFile = File(...),
    split: int = Form(1),      # 1=try to split vertically, 0=don't split
    trim: int = Form(1),       # 1=auto-trim background, 0=keep as-is
    enhance: int = Form(0),    # 1=upscale, 0=no
    scale: float = Form(2.0),  # upscale factor if enhance==1
):
    data = await file.read()
    img0 = Image.open(io.BytesIO(data)).convert("RGB")

    # Try to find split (two cards stacked vertically)
    parts = []
    if split == 1:
        yi = find_split_y(img0)
        if yi is not None:
            top = img0.crop((0, 0, img0.width, yi))
            bot = img0.crop((0, yi, img0.width, img0.height))
            parts = [top, bot]

    if not parts:
        parts = [img0]  # single part

    out64 = []
    for im in parts:
        if trim == 1:
            im = safe_trim(im)
        if enhance == 1 and scale > 1.0:
            im = im.resize((int(im.width*scale), int(im.height*scale)), resample=Image.LANCZOS)
        out64.append(pil_to_b64(im, quality=92))

    return {"count": len(out64), "images": out64}

@app.post("/resize")
async def resize_image(
    file: UploadFile = File(...),
    target_width: int = Form(3600)
):
    data = await file.read()
    img0 = Image.open(io.BytesIO(data)).convert("RGB")

    # Proportional resize
    aspect_ratio = img0.height / img0.width
    target_height = int(target_width * aspect_ratio)

    resized = img0.resize((target_width, target_height), resample=Image.LANCZOS)

    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"image": b64}
