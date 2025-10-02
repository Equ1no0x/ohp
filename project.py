# REPLACEMENT FOR ENTIRE FILE — project.py

import ctypes
import random
import time
import json
import os
import pyperclip
import psutil
import win32gui
import win32con
import win32process
import win32api
import configparser
import subprocess
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import sys

# Default success patterns for vnav path completion
VNAV_SUCCESS_PATTERNS = ["[INF] [vnavmesh] Pathfinding complete"]

# === Suppress console popup for Tesseract subprocess on Windows ===
_original_popen = subprocess.Popen

def hidden_popen(*args, **kwargs):
    if isinstance(args[0], list) and 'tesseract' in args[0][0].lower():
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        kwargs['startupinfo'] = si
    return _original_popen(*args, **kwargs)

subprocess.Popen = hidden_popen

import cv2
import numpy as np
import re
import difflib

# Always resolve files relative to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _to_abs(p: str) -> str:
    """Return absolute path; if relative, anchor at BASE_DIR."""
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)

def _exists_any(p: str) -> bool:
    """True if path exists either as-is or anchored at BASE_DIR."""
    return os.path.exists(p) or os.path.exists(_to_abs(p))

def _assets_fullpath(relpath: str) -> str:
    r = str(relpath).lstrip("/\\")
    low = r.lower()
    if low.startswith("assets/") or low.startswith(f"assets{os.sep}"):
        return _to_abs(r)
    return _to_abs(os.path.join("assets", r))

from PIL import ImageGrab
from ctypes import wintypes

# DLL and Input Setup
user32 = ctypes.WinDLL('user32', use_last_error=True)
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
MAPVK_VK_TO_VSC = 0

wintypes.ULONG_PTR = wintypes.WPARAM
last_ocr_match_center = None
last_img_match_center = None
last_detection_type = None
current_step = None

# OCR: default ceiling for general OCR calls
TESSERACT_TIMEOUT_SEC = 5
OCR_FUZZY_ENABLED = True
OCR_FUZZY_THRESHOLD = 0.90

# Debug printing for OCR when no match (or low fuzzy score)
OCR_DEBUG_DUMP = True          # set False to silence dumps
OCR_DEBUG_MAX_LINES = 1        # how many OCR "lines" to print

# Screen region of the in-game coordinates widget: (left, top, right, bottom)
VNAV_COORDS_REGION = (1298, 162, 1685, 181) # adjust if your UI scale changes
VNAV_DEFAULT_TOLERANCE = 1.6 # meters (vector3 distance)
VNAV_OCR_INTERVAL_MS = 250 # how often to re-OCR the coords
_coord_num_re = re.compile(r'[-+]?\d+(?:\.\d+)?')

mouse_buttons = {
    "VK_LBUTTON": ("left", 0x0002, 0x0004),
    "VK_RBUTTON": ("right", 0x0008, 0x0010),
    "VK_MBUTTON": ("middle", 0x0020, 0x0040),
    "VK_XBUTTON1": ("x1", 0x0080, 0x0100),
    "VK_XBUTTON2": ("x2", 0x0080, 0x0100)
}

class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx",          wintypes.LONG),
                ("dy",          wintypes.LONG),
                ("mouseData",   wintypes.DWORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk",         wintypes.WORD),
                ("wScan",       wintypes.WORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.dwFlags & KEYEVENTF_KEYUP:
            self.wScan = user32.MapVirtualKeyW(self.wVk, MAPVK_VK_TO_VSC)

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg",    wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                    ("mi", MOUSEINPUT),
                    ("hi", HARDWAREINPUT))
    _anonymous_ = ("_input",)
    _fields_ = (("type",   wintypes.DWORD),
                ("_input", _INPUT))

LPINPUT = ctypes.POINTER(INPUT)

# --- OCR engine with fallbacks (multi-line friendly) ---
def _ocr_image_to_data(img, region=None, timeout_sec=None):
    # Multi-line first; also try sparse text modes in case of tricky UI fonts
    psms = [6, 3, 11, 13]  # 6=block, 3=auto, 11/13=sparse
    langs = ["eng+ffxiv", "eng"]

    last_exc = None
    for lang in langs:
        for psm in psms:
            try:
                res = pytesseract.image_to_data(
                    img,
                    output_type=pytesseract.Output.DICT,
                    config=f"--psm {psm}",
                    timeout=(timeout_sec if timeout_sec is not None else TESSERACT_TIMEOUT_SEC),
                    lang=lang
                )
                if any((t or "").strip() for t in res.get("text", [])):
                    if OCR_DEBUG_DUMP:
                        non_empty = sum(1 for t in res["text"] if t and t.strip())
                        print(f"[dbg][ocr] used lang='{lang}' psm={psm} tokens={non_empty}")
                    return res, lang, psm
            except Exception as e:
                last_exc = e
                continue

    if last_exc:
        print(f"[dbg][ocr] OCR attempts failed; last error: {last_exc}")

    return {"text": [], "left": [], "top": [], "width": [], "height": [],
            "block_num": [], "par_num": [], "line_num": []}, None, None

# --- pick the best token-span inside a line for a target (exact or fuzzy) ---
def _best_span_for_target(tokens_norm, target_norm, max_extra=2):
    import difflib as _dl
    if not tokens_norm:
        return None
    # Quick win: if exact containment by words
    line_text = " ".join(tokens_norm)
    if target_norm and target_norm in line_text:
        # choose the shortest window that still contains target_norm
        best = (0, len(tokens_norm)-1, 1.0)
        # brute-force small windows around the target length
        tgt_words = [w for w in target_norm.split() if w]
        L = len(tokens_norm)
        min_len = max(1, len(tgt_words) - max_extra)
        max_len = min(L, len(tgt_words) + max_extra)
        for wlen in range(min_len, max_len + 1):
            for i in range(0, L - wlen + 1):
                chunk = " ".join(tokens_norm[i:i+wlen])
                if target_norm in chunk:
                    return (i, i + wlen - 1, 1.0)
        return best

    # Otherwise: sliding-window fuzzy search around target length
    tgt_words = [w for w in target_norm.split() if w]
    if not tgt_words:
        return None
    L = len(tokens_norm)
    min_len = max(1, len(tgt_words) - max_extra)
    max_len = min(L, len(tgt_words) + max_extra)
    best_ratio, best_span = 0.0, None
    for wlen in range(min_len, max_len + 1):
        for i in range(0, L - wlen + 1):
            chunk = " ".join(tokens_norm[i:i+wlen])
            r = _dl.SequenceMatcher(None, chunk, target_norm).ratio()
            if r > best_ratio:
                best_ratio, best_span = r, (i, i + wlen - 1)
    if best_span:
        return (best_span[0], best_span[1], best_ratio)
    return None

def perform_ocr_and_find_text(target_text, region=None):
    global last_ocr_match_center, last_detection_type

    print(f"[*] Performing OCR looking for: '{target_text}'")
    screenshot = ImageGrab.grab(bbox=region)

    results, _lang, _psm = _ocr_image_to_data(screenshot, region=region, timeout_sec=TESSERACT_TIMEOUT_SEC)

    # helpers
    def _norm_token(s: str) -> str:
        s = (s or "").lower()
        s = s.replace("’", "'").replace("‘", "'")
        s = re.sub(r"^[^\w]+|[^\w]+$", "", s)
        return s
    def _line_norm(tokens): return [t for t in (_norm_token(x) for x in tokens) if t]

    target_words = _line_norm(target_text.strip().split())
    target_norm_line = " ".join(target_words)

    # Group by visual line
    lines = {}
    for i, txt in enumerate(results["text"]):
        if not txt or not txt.strip():
            continue
        key = (results["block_num"][i], results["par_num"][i], results["line_num"][i])
        lines.setdefault(key, {"tokens": [], "boxes": []})
        lines[key]["tokens"].append(txt)
        x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]
        lines[key]["boxes"].append((x, y, w, h))

    # 1) Exact consecutive token match
    if target_words:
        for info in lines.values():
            norm_tokens = _line_norm(info["tokens"])
            n = len(target_words)
            for i0 in range(0, len(norm_tokens) - n + 1):
                if all(norm_tokens[i0 + j] == target_words[j] for j in range(n)):
                    # center of matched span
                    xs = [info["boxes"][i0 + k][0] for k in range(n)]
                    ys = [info["boxes"][i0 + k][1] for k in range(n)]
                    xe = [info["boxes"][i0 + k][0] + info["boxes"][i0 + k][2] for k in range(n)]
                    ye = [info["boxes"][i0 + k][1] + info["boxes"][i0 + k][3] for k in range(n)]
                    x0, y0, x1, y1 = min(xs), min(ys), max(xe), max(ye)
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    if region: cx += region[0]; cy += region[1]
                    last_ocr_match_center = (cx, cy)
                    last_detection_type = "ocr"
                    print(f"[*] Found '{target_text}' at ({cx}, {cy}) [exact]")
                    return True

    # 2) Fuzzy/containment fallback (90% default; substring accepted)
    if OCR_FUZZY_ENABLED and target_norm_line:
        best = (0.0, None, None)  # ratio, (cx,cy), dbg_text
        for info in lines.values():
            norm_tokens = _line_norm(info["tokens"])
            span = _best_span_for_target(norm_tokens, target_norm_line, max_extra=2)
            if not span: continue
            i0, i1, ratio = span
            # accept containment OR ratio ≥ threshold
            accept = (ratio >= OCR_FUZZY_THRESHOLD) or (" ".join(norm_tokens[i0:i1+1]).find(target_norm_line) >= 0)
            if accept and ratio >= best[0]:
                xs = [info["boxes"][k][0] for k in range(i0, i1 + 1)]
                ys = [info["boxes"][k][1] for k in range(i0, i1 + 1)]
                xe = [info["boxes"][k][0] + info["boxes"][k][2] for k in range(i0, i1 + 1)]
                ye = [info["boxes"][k][1] + info["boxes"][k][3] for k in range(i0, i1 + 1)]
                x0, y0, x1, y1 = min(xs), min(ys), max(xe), max(ye)
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                if region: cx += region[0]; cy += region[1]
                best = (ratio, (cx, cy), " ".join(norm_tokens[i0:i1+1]))
        if best[1] is not None:
            last_ocr_match_center = best[1]
            last_detection_type = "ocr"
            print(f"[*] Fuzzy matched '{target_text}' ≈ '{best[2]}' ({best[0]*100:.1f}%) at {best[1]}")
            return True

    # Miss: dump what we saw
    try:
        _dump_ocr_debug(results, region, target_norm_line)
        lines_dbg = _group_ocr_lines(results, region)
        if lines_dbg:
            preview = " | ".join(ln["raw"] for ln in lines_dbg[:3])
            print(f"[dbg][ocr] preview: {preview}")
    except Exception:
        pass
    print(f"[!] Text '{target_text}' not found on screen.")
    return False

def perform_img_search(template_path, threshold=0.8):
    global last_img_match_center, last_detection_type

    orig = template_path
    template_path = resolve_image_path(template_path) or template_path
    print(f"[*] Performing image search with template: {orig} → {template_path}")

    # Capture full screen
    screenshot = ImageGrab.grab()
    screenshot_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

    # Load template
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"[!] Failed to load template: {template_path}")
        return False

    h, w = template.shape[:2]

    # Match
    res = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val >= threshold:
        top_left = max_loc
        center_x = top_left[0] + w // 2
        center_y = top_left[1] + h // 2
        last_img_match_center = (center_x, center_y)
        last_detection_type = "img"
        print(f"[*] Found image at ({center_x}, {center_y}) with confidence {max_val:.2f}")
        return True
    else:
        print("[!] Image not found")
        return False

# --- Fuzzy OCR for chat lines (case-insensitive, punctuation-tolerant) ---
def _normalize_chat_text(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("’", "'").replace("‘", "'")  # unify apostrophes
    s = re.sub(r"[\.…]+$", "", s)              # drop trailing dots (OCR often misses them)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def perform_ocr_find_text_fuzzy(target_text: str, region=None, threshold: float = 0.90,
                                ocr_timeout_sec: float | None = None) -> bool:
    """
    Fuzzy search across the whole region. On success, centers on the matched token-span.
    """
    global last_ocr_match_center, last_detection_type

    target_norm = _normalize_chat_text(target_text)  # your existing normalizer is fine
    print(f"[*] Performing FUZZY OCR looking for (~{int(threshold*100)}%): '{target_norm}'")

    screenshot = ImageGrab.grab(bbox=region)
    results, _lang, _psm = _ocr_image_to_data(
        screenshot, region=region, timeout_sec=(ocr_timeout_sec if ocr_timeout_sec is not None else TESSERACT_TIMEOUT_SEC)
    )

    # Group per visual line
    lines = {}
    for i, txt in enumerate(results["text"]):
        if not txt or not txt.strip():
            continue
        key = (results["block_num"][i], results["par_num"][i], results["line_num"][i])
        lines.setdefault(key, {"tokens": [], "boxes": []})
        lines[key]["tokens"].append(txt)
        x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]
        lines[key]["boxes"].append((x, y, w, h))

    best_ratio, best_center, best_text = 0.0, None, None
    for info in lines.values():
        # reuse the generic normalizer (punctuation-trimmed, lower)
        norm_tokens = [_norm_token_generic(t) for t in info["tokens"] if _norm_token_generic(t)]
        if not norm_tokens: continue
        span = _best_span_for_target(norm_tokens, target_norm, max_extra=2)
        if not span: continue
        i0, i1, ratio = span
        # containment OR similarity
        chunk = " ".join(norm_tokens[i0:i1+1])
        if (target_norm in chunk) or (ratio >= threshold):
            xs = [info["boxes"][k][0] for k in range(i0, i1 + 1)]
            ys = [info["boxes"][k][1] for k in range(i0, i1 + 1)]
            xe = [info["boxes"][k][0] + info["boxes"][k][2] for k in range(i0, i1 + 1)]
            ye = [info["boxes"][k][1] + info["boxes"][k][3] for k in range(i0, i1 + 1)]
            x0, y0, x1, y1 = min(xs), min(ys), max(xe), max(ye)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            if region: cx += region[0]; cy += region[1]
            if ratio > best_ratio:
                best_ratio, best_center, best_text = ratio, (cx, cy), chunk

    if best_center is not None:
        last_ocr_match_center = best_center
        last_detection_type = "ocr"
        print(f"[*] Fuzzy match {best_ratio:.1%} for '{target_norm}' → '{best_text}' at {best_center}")
        return True

    # Miss: dump lines
    try:
        _dump_ocr_debug(results, region, target_norm)
    except Exception:
        pass
    print(f"[!] Fuzzy OCR below threshold ({best_ratio:.1%}) for '{target_norm}'")
    return False

def move_mouse_hardware(x, y):
    print(f"[*] Moving mouse to ({x}, {y})")
    user32.SetCursorPos(x, y)

def PressKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode, dwFlags=KEYEVENTF_KEYUP))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

def load_json(filename):
    tried = []

    # 0) allow direct read ONLY for the registry files themselves
    low = str(filename).lower().replace("\\", "/")
    if low.endswith("assets/files.json") or low.endswith("/files.json"):
        p = _to_abs(filename)
        tried.append(p)
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    if low.endswith("assets/images.json") or low.endswith("/images.json"):
        p = _to_abs(filename)
        tried.append(p)
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    # 1) resolve via registry
    _load_asset_indices()  # builds ASSET_FILE_INDEX/ASSET_IMAGE_INDEX from disk directly
    key = os.path.basename(str(filename)).lower()
    rel = ASSET_FILE_INDEX.get(key) or ASSET_FILE_INDEX.get(f"{key}.json")

    if rel:
        rp = _assets_fullpath(rel)
        tried.append(rp)
        with open(rp, "r", encoding="utf-8") as f:
            return json.load(f)

    # No direct path fallbacks; force everything through the registry.
    raise FileNotFoundError(f"No such JSON via registry: {filename} (tried: {tried})")

def _substitute_params(obj, params):
    """Replace $key placeholders; keep non-string types when the placeholder is the entire string."""
    if isinstance(obj, str):
        # Exact-token replacement (preserve type): "…": "$var"
        if obj.startswith("$") and params and (obj[1:] in params):
            val = params[obj[1:]]
            # Only use exact-token rule when the placeholder is the entire string;
            # otherwise fall back to substring replacements below.
            if isinstance(val, (list, tuple, dict, int, float, bool, type(None))):
                return val
        # Substring replacement (stringify)
        out = obj
        for k, v in (params or {}).items():
            out = out.replace(f"${k}", str(v))
        return out
    if isinstance(obj, list):
        return [_substitute_params(x, params) for x in obj]
    if isinstance(obj, dict):
        return {k: _substitute_params(v, params) for k, v in obj.items()}
    return obj

# ---- Asset registries (optional QoL) ----
ASSET_FILE_INDEX = {}
ASSET_IMAGE_INDEX = {}

def _build_asset_index(spec):
    """
    Accepts dict{name->relpath} or list[relpath]. Returns {name_lower: normalized_relpath}.
    """
    out = {}
    try:
        if isinstance(spec, dict):
            for k, v in spec.items():
                if not v: continue
                name = os.path.basename(str(k)).lower()
                out[name] = str(v).lstrip("/\\")
        elif isinstance(spec, list):
            for p in spec:
                if not p: continue
                p = str(p)
                out[os.path.basename(p).lower()] = p.lstrip("/\\")
    except Exception:
        pass
    return out

def _load_asset_indices():
    """Load asset registries from disk without going through load_json()."""
    global ASSET_FILE_INDEX, ASSET_IMAGE_INDEX

    files_json  = None
    images_json = None

    for cand in ("assets/files.json", "assets/FILES.json"):
        p = _to_abs(cand)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                files_json = json.load(f)
            break

    for cand in ("assets/images.json", "assets/IMAGES.json"):
        p = _to_abs(cand)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                images_json = json.load(f)
            break

    ASSET_FILE_INDEX  = _build_asset_index(files_json)  if files_json  else {}
    ASSET_IMAGE_INDEX = _build_asset_index(images_json) if images_json else {}

def resolve_file_path(name_or_path: str) -> str | None:
    """Resolve by registry key (basename) only; normalize result."""
    p = str(name_or_path)
    key = os.path.basename(p).lower()
    _load_asset_indices()  # ensure registry is available
    rel = ASSET_FILE_INDEX.get(key) or ASSET_FILE_INDEX.get(f"{key}.json")
    if rel:
        return os.path.normpath(_assets_fullpath(rel))
    return None

def resolve_image_path(name_or_path: str) -> str | None:
    p = str(name_or_path)
    if _exists_any(p):
        return _to_abs(p)
    if p.lower().startswith("assets" + os.sep) or p.lower().startswith("assets/"):
        return _to_abs(p)
    key = os.path.basename(p).lower()
    rel = ASSET_IMAGE_INDEX.get(key)
    if rel:
        return _assets_fullpath(rel)
    return None

def find_process_window(process_name):
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            print(f"[+] Found process {process_name} (PID: {proc.pid})")
            return get_window_by_pid(proc.info['pid'])
    return None

def get_window_by_pid(pid):
    def callback(hwnd, windows):
        _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
        if found_pid == pid and win32gui.IsWindowVisible(hwnd):
            windows.append(hwnd)
        return True
    windows = []
    win32gui.EnumWindows(callback, windows)
    for hwnd in windows:
        title = win32gui.GetWindowText(hwnd)
        if title:
            print(f"[+] Window Handle: {hwnd}, Title: {title}")
            return hwnd
    return None

def focus_game_window(hwnd):
    try:
        current_foreground = win32gui.GetForegroundWindow()
        if current_foreground == hwnd:
            return

        print("[*] Bringing game window to front...")

        foreground_thread_id = win32process.GetWindowThreadProcessId(current_foreground)[0]
        target_thread_id = win32process.GetWindowThreadProcessId(hwnd)[0]
        current_thread_id = win32api.GetCurrentThreadId()

        if foreground_thread_id != target_thread_id:
            win32process.AttachThreadInput(current_thread_id, foreground_thread_id, True)
            win32process.AttachThreadInput(current_thread_id, target_thread_id, True)

        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.25)

        win32process.AttachThreadInput(current_thread_id, foreground_thread_id, False)
        win32process.AttachThreadInput(current_thread_id, target_thread_id, False)

        print("[*] Game window is now in focus.")

    except Exception as e:
        print(f"[!] Failed to bring window to front: {e}")

def send_chat_command(command, keymap):
    # Copy text
    pyperclip.copy("")
    time.sleep(0.05)
    pyperclip.copy(command)

    # Open chat
    enter_vk = int(keymap["VK_RETURN"], 16)
    PressKey(enter_vk); time.sleep(0.05); ReleaseKey(enter_vk)
    time.sleep(0.2)

    # Paste
    ctrl = int(keymap["VK_LCONTROL"], 16)
    vkey = int(keymap["V"], 16)
    PressKey(ctrl); time.sleep(0.05)
    PressKey(vkey); time.sleep(0.05)
    ReleaseKey(vkey); time.sleep(0.05)
    ReleaseKey(ctrl); time.sleep(0.2)

    # Send
    PressKey(enter_vk); time.sleep(0.05); ReleaseKey(enter_vk)
    time.sleep(0.1)

def execute_command_sequence(macro_array, keymap, command_text=None):
    i = 0
    while i < len(macro_array):
        cmd = macro_array[i]

        if cmd in ("press", "release") and macro_array[i+1] in mouse_buttons:
            btn, down_flag, up_flag = mouse_buttons[macro_array[i+1]]
            if cmd == "press":
                print(f"[*] Pressing {btn} mouse button")
                if "XBUTTON" in macro_array[i+1]:
                    user32.mouse_event(down_flag, 0, 0, 0, 1 if btn == "x1" else 2)
                else:
                    user32.mouse_event(down_flag, 0, 0, 0, 0)
            else:
                print(f"[*] Releasing {btn} mouse button")
                if "XBUTTON" in macro_array[i+1]:
                    user32.mouse_event(up_flag, 0, 0, 0, 1 if btn == "x1" else 2)
                else:
                    user32.mouse_event(up_flag, 0, 0, 0, 0)
            i += 2

        elif cmd == "press":
            key_name = macro_array[i+1]
            vk = int(keymap[key_name], 16)
            print(f"[*] Pressing {key_name}")
            PressKey(vk)
            i += 2

        elif cmd == "release":
            key_name = macro_array[i+1]
            vk = int(keymap[key_name], 16)
            print(f"[*] Releasing {key_name}")
            ReleaseKey(vk)
            i += 2
            
        elif cmd == "wait":
            duration = macro_array[i+1]
            print(f"[*] Wait: {duration}ms")
            time.sleep(duration / 1000)
            i += 2

        elif cmd == "rand":
            low = int(macro_array[i+1])
            high = int(macro_array[i+2])
            delay = random.randint(low, high)
            print(f"[*] Random wait: {delay}ms")
            time.sleep(delay / 1000)
            i += 3

        else:
            print(f"[!] Unknown macro command: {cmd}")
            i += 1

def is_macro_step(step):
    return isinstance(step, dict) and any(k in step for k in ("press", "release", "wait", "rand", "paste"))

def _quest_key_for_path(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0].lower()

def _quest_source_for(qkey: str, manifest_list) -> str:
    # default to "class" if not found
    for entry in (manifest_list or []):
        if isinstance(entry, dict) and str(entry.get("quest", "")).strip().lower() == qkey:
            return str(entry.get("source", "class")).lower()
    return "class"

def _progress_load_root() -> dict:
    try:
        return load_json("progress.json") or {}
    except Exception:
        return {}

def _progress_save_root(root: dict) -> None:
    p = resolve_file_path("progress.json") or "progress.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(root, f, indent=2)

def load_progress(file_path: str) -> int:
    qkey = _quest_key_for_path(file_path)
    source = _quest_source_for(qkey, globals().get("manifest_list"))
    root = _progress_load_root()
    if source == "shared":
        val = (root.get("shared", {}).get(qkey, {}) or {}).get("step")
    else:
        val = (root.get("jobs", {}).get(globals().get("selected_class", ""), {}).get(qkey, {}) or {}).get("step")
    try:
        v = int(val) if val is not None else 1
        return 1 if v <= 0 else v
    except Exception:
        return 1

def save_progress(file_path: str, step: int) -> None:
    qkey = _quest_key_for_path(file_path)
    source = _quest_source_for(qkey, globals().get("manifest_list"))
    root = _progress_load_root()

    # ensure structure
    root.setdefault("shared", {})
    root.setdefault("jobs", {})
    root["jobs"].setdefault(globals().get("selected_class", ""), {})

    entry = {"completed": False, "step": int(step if step >= 1 else 1)}
    if source == "shared":
        root["shared"].setdefault(qkey, {}).update(entry)
    else:
        root["jobs"][globals().get("selected_class", "")].setdefault(qkey, {}).update(entry)

    # keep "current" in sync
    root.setdefault("current", {})
    root["current"].update({
        "job": globals().get("selected_class", ""),
        "quest": qkey,
        "step": int(step if step >= 1 else 1),
    })

    _progress_save_root(root)

def resolve_goto_chain(file, step):
    """Follow goto-only steps until a real step is found."""
    visited = set()
    while True:
        data = load_json(file)
        step_list = data.get(str(step))
        if not step_list:
            break
        only_goto = all(isinstance(s, dict) and "goto" in s for s in step_list)
        if not only_goto:
            break
        # handle goto
        goto_entry = step_list[0]["goto"]
        target_file, target_step = goto_entry
        if (not target_file) and int(target_step) == 0:
            break
        if (file, step) in visited:
            print("[!] Circular goto chain detected.")
            break
        visited.add((file, step))
        if os.path.exists(target_file):
            print(f"[*] Resolving startup GOTO → {target_file}, step {target_step}")
            file = target_file
            step = int(target_step)
        else:
            print(f"[!] GOTO target file not found: {target_file}")
            break
    return file, step

def start_combat(keymap):
    print("[*] Starting combat rotation...")
    send_chat_command("/rotation Manual", keymap)
    time.sleep(0.2)
    send_chat_command("/bmrai on", keymap)
    time.sleep(0.2)
    send_chat_command("/vnav movetarget", keymap)
    time.sleep(0.2)

def end_combat(keymap):
    print("[*] Ending combat rotation...")
    send_chat_command("/rotation off", keymap)
    time.sleep(0.2)
    send_chat_command("/bmrai off", keymap)
    time.sleep(0.2)

def handle_combat_step(combat_data, keymap):
    names = combat_data.get("names", [])
    counts = combat_data.get("counts", [])
    end_command = combat_data.get("end_command")

    print(f"[*] Combat started. Targets: {dict(zip(names, counts))}")

    kills = {name: 0 for name in names}
    start_time = time.time()

    # Hard-coded OCR region for mob name
    name_region = (750, 31, 1190, 52)

    while any(kills[name] < counts[i] for i, name in enumerate(names)):
        # TAB target cycle
        # Build list of remaining targets (respect requested counts)
        remaining = [n for i, n in enumerate(names) if kills[n] < counts[i]]
        if not remaining:
            print("[+] All requested kills completed.")
            break

        # TAB target cycle
        execute_command_sequence(["press", "VK_TAB", "rand", 56, 126, "release", "VK_TAB"], keymap)
        time.sleep(1.5)

        # Step 1: OCR check for current target (inside fixed region), only among remaining
        found_target = None
        for n in remaining:
            if perform_ocr_and_find_text(n, name_region):
                found_target = n
                break

        # If the OCR hit is already at quota (race condition), skip it
        if found_target is not None:
            idx = names.index(found_target)
            if kills[found_target] >= counts[idx]:
                print(f"[i] Already met quota for {found_target}, skipping.")
                found_target = None

        if found_target:
            print(f"[*] Engaging {found_target}...")

            # Move mouse to the last OCR match center before attacking
            if last_detection_type == "ocr" and last_ocr_match_center:
                move_mouse_hardware(*last_ocr_match_center)
                time.sleep(0.2)
            elif last_detection_type == "img" and last_img_match_center:
                move_mouse_hardware(*last_img_match_center)
                time.sleep(0.2)
            else:
                print("[!] No valid target position available for mouse move.")

            # Right-click to target
            execute_command_sequence(
                ["press", "VK_RBUTTON", "rand", 40, 80, "release", "VK_RBUTTON"], keymap
            )
            time.sleep(0.3)

            # Left-click to confirm interaction/attack
            execute_command_sequence(
                ["press", "VK_LBUTTON", "rand", 40, 80, "release", "VK_LBUTTON"], keymap
            )
            # Start combat helper
            start_combat(keymap)
            time.sleep(0.5)

            # Step 2: Switch to Battle tab and wait for the defeat line.
            # We "arm" first by requiring the line to be ABSENT, then look for it to appear.
            success = await_battle_defeat(found_target.lower(), keymap, interval_ms=1000, timeout_ms=90000)
            if success:
                idx = names.index(found_target)
                if kills[found_target] < counts[idx]:
                    kills[found_target] += 1
                if kills[found_target] > counts[idx]:
                    kills[found_target] = counts[idx]

                progress_str = ", ".join(
                    f"{n}: {kills[n]}/{counts[i]}" for i, n in enumerate(names)
                )
                print(f"[*] {found_target} defeated. Progress [{progress_str}]")
            else:
                print(f"[!] Timed out waiting for Battle chat defeat line for {found_target}.")

            # End combat helper (either way)
            end_combat(keymap)

        else:
            # No valid mob found
            print("[!] No valid target found, waiting (fallback 10s)...")
            time.sleep(0.5)

        # Safety check for timeout
        if time.time() - start_time > 60 * 5:  # 5 min cap
            print("[!] Combat timed out.")
            break
        
        # Rotating in place to look for more mobs
        execute_command_sequence(
                ["press", "VK_RIGHT", "rand", 156, 256, "release", "VK_RIGHT"], keymap
            )
        time.sleep(0.3)

        # Always return to scouting spot if end_command is defined
        # if end_command:
        #    print(f"[*] Returning to scout position with: {end_command}")
        #    send_chat_command(end_command, keymap)

    print(f"[+] Combat finished. Returning to scout position: {end_command}")
    if end_command:
        send_chat_command(end_command, keymap)

def parse_vector3_string(s: str):
    nums = _coord_num_re.findall(s)
    if len(nums) < 3:
        raise ValueError(f"Could not parse three coordinates from: {s!r}")
    x, y, z = map(float, nums[:3])
    return x, y, z

def read_player_coords_from_screen(region=VNAV_COORDS_REGION):
    try:
        img = ImageGrab.grab(bbox=region)
        text = pytesseract.image_to_string(
            img,
            lang="eng+ffxiv",
            config="--psm 7 -c tessedit_char_whitelist=0123456789.-, "
        )
        text_clean = text.replace("\n", " ").replace("  ", " ").strip()
        return parse_vector3_string(text_clean)
    except Exception:
        return None

def vec3_distance(a, b):
    ax, ay, az = a
    bx, by, bz = b
    dx, dy, dz = ax - bx, ay - by, az - bz
    return (dx*dx + dy*dy + dz*dz) ** 0.5

def normalize_vnav_value(value, *, default_timeout=120000, default_tolerance=VNAV_DEFAULT_TOLERANCE, label="vnav"):
    """
    Normalize JSON 'vnav' value to (coord_str, timeout_ms, tolerance).
    Accepts: "X Y Z"  OR  ["X Y Z", timeout_ms]  OR  ["X Y Z", timeout_ms, tolerance].
    """
    if isinstance(value, str):
        return value.strip(), default_timeout, default_tolerance
    if isinstance(value, list) and len(value) >= 1:
        coord_str = str(value[0]).strip()
        timeout_ms = int(value[1]) if len(value) >= 2 else default_timeout
        tolerance = float(value[2]) if len(value) >= 3 else default_tolerance
        return coord_str, timeout_ms, tolerance
    raise ValueError(f"Invalid 'vnav' value for {label}: {value!r}")

def normalize_vnav_path(value, *, default_timeout=120000, default_tolerance=VNAV_DEFAULT_TOLERANCE, label="vnav"):
    # String => single leg
    if isinstance(value, str):
        return [normalize_vnav_value(value, default_timeout=default_timeout,
                                     default_tolerance=default_tolerance, label=label)]

    # List => decide between single-leg array vs path (list of legs)
    if isinstance(value, list):
        # Heuristic: treat as a **single leg** if timeout/tolerance are numeric;
        # otherwise treat as a **path** whose elements are legs.
        is_single_candidate = (
            len(value) == 1
            or (len(value) >= 2 and isinstance(value[1], (int, float)))
        )
        if is_single_candidate and (len(value) < 3 or isinstance(value[2], (int, float))):
            # ["X Y Z"] or ["X Y Z", 60000] or ["X Y Z", 60000, 1.6]
            leg = normalize_vnav_value(value, default_timeout=default_timeout,
                                       default_tolerance=default_tolerance, label=label)
            return [leg]

        # Otherwise: treat as path; each element is a leg (string or ["X Y Z", t, tol])
        legs = []
        for i, leg_val in enumerate(value, start=1):
            leg = normalize_vnav_value(leg_val, default_timeout=default_timeout,
                                       default_tolerance=default_tolerance,
                                       label=f"{label} leg {i}")
            legs.append(leg)
        return legs

    raise ValueError(f"Invalid 'vnav' path value for {label}: {value!r}")

def vnav_go(coord_str: str, timeout_ms: int, tolerance: float, keymap) -> bool:
    try:
        target_vec = parse_vector3_string(coord_str)
    except ValueError as e:
        print(f"[!] vnav: {e}")
        return False

    cmd = f"/vnav moveto {coord_str}"
    print(f"  [*] vnav → {cmd} (tolerance={tolerance})")
    send_chat_command(cmd, keymap)

    # NEW: negative timeout => wait forever
    wait_forever = (timeout_ms is None) or (int(timeout_ms) < 0)

    start = time.time()
    last_print = 0.0
    while wait_forever or ((time.time() - start) * 1000.0 < timeout_ms):
        live = read_player_coords_from_screen(VNAV_COORDS_REGION)
        if live is not None:
            dist = vec3_distance(live, target_vec)
            now = time.time()
            if now - last_print > 1.0:
                print(f"    [vnav] pos={live[0]:.3f},{live[1]:.3f},{live[2]:.3f}  dist={dist:.3f}")
                last_print = now
            if dist <= tolerance:
                print("[*] vnav: arrived (within tolerance).")
                return True
        time.sleep(VNAV_OCR_INTERVAL_MS / 1000.0)

    # Only reachable when wait_forever is False and time elapsed:
    print(f"[!] vnav: timed out after {timeout_ms}ms waiting to reach target (tolerance={tolerance}).")
    if "VK_DECIMAL" in keymap:
        execute_command_sequence(
            ["press", "VK_DECIMAL", "rand", 56, 76, "release", "VK_DECIMAL"],
            keymap,
            "vnav_timeout_fallback"
        )
    return False

def wait_until_near(coord_str: str, timeout_ms: int, tolerance: float) -> bool:
    try:
        target_vec = parse_vector3_string(coord_str)
    except ValueError as e:
        print(f"[!] destination: {e}")
        return False

    print(f"  [*] destination → target {coord_str} (tolerance={tolerance})")

    wait_forever = (timeout_ms is None) or (int(timeout_ms) < 0)
    start = time.time()
    last_print = 0.0

    while wait_forever or ((time.time() - start) * 1000.0 < timeout_ms):
        live = read_player_coords_from_screen(VNAV_COORDS_REGION)
        if live is not None:
            dist = vec3_distance(live, target_vec)
            now = time.time()
            if now - last_print > 1.0:
                print(f"    [dest] pos={live[0]:.3f},{live[1]:.3f},{live[2]:.3f}  dist={dist:.3f}")
                last_print = now
            if dist <= tolerance:
                print("[*] destination: arrived (within tolerance).")
                return True
        time.sleep(VNAV_OCR_INTERVAL_MS / 1000.0)

    print(f"[!] destination: timed out after {timeout_ms}ms (tolerance={tolerance}).")
    return False

def execute_one_step(step, keymap):
    # These are set/used by your existing OCR/IMG helpers and mouse movers.
    global last_ocr_text, last_ocr_region, last_img_template
    global last_detection_type, last_ocr_match_center, last_img_match_center

    if not isinstance(step, dict):
        return False

    if "command" in step:
        key = step["command"]
        if isinstance(key, str) and key.startswith("/"):
            print(f"[*] Sending chat command: {key}")
            send_chat_command(key, keymap)
        else:
            seq = commands_data.get(key)
            if seq:
                print(f"[*] Executing command macro: {key}")
                execute_command_sequence(seq, keymap)
            else:
                print(f"[!] Unknown command macro: {key}")
        return True

    if "macro" in step:
        print("[*] Executing inline macro...")
        execute_command_sequence(step["macro"], keymap)
        return True

    if "region" in step:
        try:
            rv = step["region"]
            if isinstance(rv, (list, tuple)) and len(rv) == 4:
                last_ocr_region = tuple(int(x) for x in rv)
            elif isinstance(rv, dict):
                # accept l/t/r/b or L/T/R/B
                def _g(d, k): 
                    return d.get(k) if k in d else d.get(k.upper())
                last_ocr_region = (int(_g(rv, "l")), int(_g(rv, "t")), int(_g(rv, "r")), int(_g(rv, "b")))
            else:
                # string "L,T,R,B" with optional whitespace
                parts = re.split(r"\s*,\s*", str(rv).strip())
                if len(parts) != 4:
                    raise ValueError("expected 4 comma-separated integers")
                last_ocr_region = tuple(int(p) for p in parts)
            print(f"[*] OCR region set to {last_ocr_region}")
        except Exception as e:
            print(f"[!] Bad region value: {step['region']} ({e})")
        return True

    if "ocr" in step:
        last_ocr_text = step["ocr"]
        # Optional immediate probe (keeps previous behavior where an OCR step often probed)
        perform_ocr_and_find_text(last_ocr_text, last_ocr_region)
        last_detection_type = "ocr"
        return True

    if "img_search" in step:
        last_img_template = step["img_search"]
        perform_img_search(last_img_template)
        last_detection_type = "img"
        return True

    if "mouse_move" in step:
        mv = step["mouse_move"]
        if isinstance(mv, list) and len(mv) == 2:
            x, y = int(mv[0]), int(mv[1])
            move_mouse_hardware(x, y)
        elif mv == "move_to":
            if last_detection_type == "ocr" and last_ocr_match_center:
                move_mouse_hardware(*last_ocr_match_center)
            elif last_detection_type == "img" and last_img_match_center:
                move_mouse_hardware(*last_img_match_center)
            else:
                print("[!] mouse_move(move_to): no detection center available.")
        else:
            print(f"[!] Unknown mouse_move value: {mv}")
        return True

    if "wait" in step:
        ms = int(step["wait"])
        print(f"[*] Wait: {ms}ms")
        time.sleep(ms / 1000.0)
        return True

    if "rand" in step:
        low, high = step["rand"]
        delay = random.randint(int(low), int(high))
        print(f"[*] Random wait: {delay}ms")
        time.sleep(delay / 1000.0)
        return True

    if "destination" in step:
        try:
            coord_str, timeout_ms, tolerance = normalize_vnav_value(
                step["destination"], default_timeout=120000, default_tolerance=VNAV_DEFAULT_TOLERANCE, label="destination"
            )
        except ValueError as e:
            print(f"[!] {e}")
            return True
        wait_until_near(coord_str, timeout_ms, tolerance)
        return True

    if "await" in step:
        # Shapes (back-compat):
        # - OCR/IMG: [interval, "ocr"/"img", expect_found, key_to_press, max_time, (optional) fallback_bool]
        # - LOG:     [interval, "log",       expect_found, key_to_press, max_time, cond_dict, (optional) fallback_bool]
        arr = step["await"]
        if len(arr) < 5:
            print("[!] await: expected at least 5 elements.")
            return True

        interval, mode, expect_found, key_to_press, max_time = arr[:5]

        cond_spec = None
        fallback_decimal = True
        if mode == "log":
            if len(arr) >= 6 and isinstance(arr[5], dict):
                cond_spec = arr[5]
            if len(arr) >= 7 and isinstance(arr[6], bool):
                fallback_decimal = arr[6]
        else:
            if len(arr) >= 6 and isinstance(arr[5], bool):
                fallback_decimal = arr[5]

        print(f"[*] Awaiting {str(mode).upper()} every {interval}ms, "
              f"expect={'found' if expect_found else 'not found'}, "
              f"key={key_to_press if key_to_press else 'none'}, max={max_time}ms, "
              f"fallback_decimal={'on' if fallback_decimal else 'off'}")

        # --- LOG mode removed ---
        if mode == "log":
            return True

        # --- OCR/IMG polling: EXACT order per request ---
        elapsed = 0
        success = False
        while elapsed < max_time:
            # 1) send specified key (if any), then wait 0.5s to let UI update
            if key_to_press and key_to_press in keymap:
                execute_command_sequence(
                    ["press", key_to_press, "rand", 30, 60, "release", key_to_press],
                    keymap, "await_loop"
                )
            time.sleep(0.5)

            # 2) perform the check
            if mode == "ocr":
                if not last_ocr_text:
                    print("[!] await(ocr): no prior 'ocr' text set.")
                    break
                found = perform_ocr_and_find_text(last_ocr_text, last_ocr_region)
            elif mode == "img":
                if not last_img_template:
                    print("[!] await(img): no prior 'img_search' template set.")
                    break
                found = perform_img_search(last_img_template)
            else:
                print(f"[!] await: unknown mode '{mode}'")
                break

            # Success criteria
            if (expect_found and found) or ((not expect_found) and (not found)):
                success = True
                break

            # Not successful this cycle: wait the remainder of the interval, then fallback
            remaining = max(0.0, (interval / 1000.0) - 0.5)
            if remaining > 0:
                time.sleep(remaining)
            elapsed += interval

            if fallback_decimal and "VK_DECIMAL" in keymap:
                execute_command_sequence(
                    ["press", "VK_DECIMAL", "rand", 30, 60, "release", "VK_DECIMAL"],
                    keymap, "await_fallback"
                )

        if not success and mode in ("ocr", "img"):
            print("[!] await: condition not satisfied.")
        return True

    # Not a primitive we handle here
    return False

# --- Battle chat helpers ---
BATTLE_CHAT_REGION = (30, 951, 590, 967)  # left, top, right, bottom (as requested)

def _click_battle_tab(keymap) -> bool:
    if perform_img_search("assets/chat_battle.png"):
        if last_img_match_center:
            move_mouse_hardware(*last_img_match_center)
            execute_command_sequence(
                ["press", "VK_LBUTTON", "rand", 40, 80, "release", "VK_LBUTTON"], keymap
            )
            time.sleep(0.2)
            return True
    print("[!] Battle tab image not found; continuing without click.")
    return False

def _battle_defeat_phrase(monster_name: str) -> str:
    return f"you defeat the {monster_name}".strip().lower()

def await_battle_defeat(monster_name: str, keymap,
                        interval_ms: int | None = None,           # legacy single-interval
                        timeout_ms: int = 90000,
                        fuzzy_threshold: float = 0.80,
                        poll_min_ms: int = 150, poll_max_ms: int = 250,  # NEW fast polling window
                        ocr_timeout_sec: float = 0.8) -> bool:           # NEW per-scan OCR timeout

    _click_battle_tab(keymap)

    phrase = _battle_defeat_phrase(monster_name)  # already lower-cased / no trailing '.'

    def _sleep():
        # keep some jitter so we don't alias the chat rendering cadence
        delay = (interval_ms if interval_ms is not None else random.randint(poll_min_ms, poll_max_ms))
        time.sleep(delay / 1000.0)
        return delay

    # Arm: require the line to be absent first
    elapsed = 0
    while elapsed < timeout_ms:
        found = perform_ocr_find_text_fuzzy(phrase, BATTLE_CHAT_REGION,
                                            threshold=fuzzy_threshold,
                                            ocr_timeout_sec=ocr_timeout_sec)
        if not found:
            break
        elapsed += _sleep()

    # Wait for it to appear
    elapsed = 0
    while elapsed < timeout_ms:
        found = perform_ocr_find_text_fuzzy(phrase, BATTLE_CHAT_REGION,
                                            threshold=fuzzy_threshold,
                                            ocr_timeout_sec=ocr_timeout_sec)
        if found:
            return True
        elapsed += _sleep()

    return False

# --- OCR Debugging chat helpers ---
def _norm_token_generic(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s)  # trim punctuation at ends
    return s

def _group_ocr_lines(results, region=None):
    """Group Tesseract tokens into visual lines; return [{raw,norm,bbox,center}, ...]."""
    lines = {}
    for i, txt in enumerate(results["text"]):
        if not txt or not txt.strip():
            continue
        key = (results["block_num"][i], results["par_num"][i], results["line_num"][i])
        lines.setdefault(key, {"texts": [], "boxes": []})
        lines[key]["texts"].append(txt)
        x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]
        lines[key]["boxes"].append((x, y, w, h))

    out = []
    for info in lines.values():
        tokens = info["texts"]
        raw = " ".join(tokens).strip()
        norm = " ".join(_norm_token_generic(t) for t in tokens if _norm_token_generic(t))
        xs = [b[0] for b in info["boxes"]]; ys = [b[1] for b in info["boxes"]]
        xe = [b[0] + b[2] for b in info["boxes"]]; ye = [b[1] + b[3] for b in info["boxes"]]
        x0, y0, x1, y1 = min(xs), min(ys), max(xe), max(ye)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if region:
            cx += region[0]; cy += region[1]
            x0 += region[0]; x1 += region[0]; y0 += region[1]; y1 += region[1]
        out.append({"raw": raw, "norm": norm, "bbox": (x0, y0, x1, y1), "center": (cx, cy)})
    return out

def _dump_ocr_debug(results, region=None, target=None):
    """Print a compact dump of OCR lines to help debug misses."""
    if not OCR_DEBUG_DUMP:
        return
    try:
        lines = _group_ocr_lines(results, region)
        hdr = f"[dbg][ocr] lines={len(lines)}"
        if target: hdr += f" (target='{target}')"
        print(hdr)
        for i, ln in enumerate(lines[:OCR_DEBUG_MAX_LINES], 1):
            print(f"  {i:>2}. raw='{ln['raw']}' | norm='{ln['norm']}' | center={ln['center']} bbox={ln['bbox']}")
        if len(lines) > OCR_DEBUG_MAX_LINES:
            print(f"  … {len(lines) - OCR_DEBUG_MAX_LINES} more line(s) omitted")
    except Exception as e:
        print(f"[dbg][ocr] dump failed: {e}")

# === Main Logic ===
single_run_mode = False

def reload_jsons():
    global commands_data, locations_data, keymap, npc_data, actions_data, interaction_types

    # load registries first so load_json() can resolve via files.json
    _load_asset_indices()

    mapping = [
        ("commands_data",      "commands.json"),
        ("locations_data",     "locations.json"),
        ("keymap",             "keymaps.json"),
        ("npc_data",           "npcs.json"),
        ("interaction_types",  "interaction_types.json"),
    ]

    for var, fname in mapping:
        try:
            # let load_json use registry fallback transparently
            globals()[var] = load_json(fname)
        except Exception as e:
            print(f"[!] Failed to load {fname}: {e}")
            globals()[var] = {}

reload_jsons()

# --- Class selection by id or name (via assets/classes.json) ---
def _load_classes_index():
    try:
        classes_spec = load_json("classes.json")  # resolved via assets/files.json
    except Exception:
        return {}
    entries = classes_spec.get("classes", []) if isinstance(classes_spec, dict) else []
    idx = {}
    for e in entries:
        cid = str(e.get("id", "")).lower()
        cname = str(e.get("name", "")).lower()
        f = e.get("file")
        if cid and f:
            idx[cid] = f
        if cname and f:
            idx[cname] = f
    return idx

# Game window
TARGET_PROCESS = "ffxiv_dx11.exe"
hwnd = find_process_window(TARGET_PROCESS)

# --- Job selection via classes.json (single canonical block) ---
classes_index = _load_classes_index()  # {"pld": "pld.json", "paladin": "pld.json", ...}
if not classes_index:
    print("[!] No classes found in classes.json.")
    raise SystemExit(1)

user_in = input("Enter job id or name (e.g. 'pld' or 'paladin'): ").strip().lower()
job_file = classes_index.get(user_in)
if not job_file:
    print(f"[!] Unknown job '{user_in}'. Check classes.json.")
    raise SystemExit(1)

# Canonicalize to the job ID (e.g., 'pld' even if user typed 'paladin')
cls_spec = load_json("classes.json")
selected_entry = next(
    (e for e in cls_spec.get("classes", [])
     if str(e.get("file","")).lower() == os.path.basename(job_file).lower()
        or user_in in (str(e.get("id","")).lower(), str(e.get("name","")).lower())),
    {}
)
selected_class = selected_entry.get("id", user_in)

# Optional: show the resolved path for clarity (still loaded via the registry)
job_file_resolved = resolve_file_path(job_file) or job_file
print(f"[+] Selected job: {selected_class} → {job_file_resolved}")

# --- ALWAYS load the CLASS file (manifest) first ---
manifest_raw = load_json(job_file)  # e.g. "pld.json" → list of {"quest": "...", "source": "shared"|"class"}
print(f"[+] Loaded data from {job_file}")

# Normalize manifest to a list
manifest_list = manifest_raw["quests"] if isinstance(manifest_raw, dict) and "quests" in manifest_raw else (
    manifest_raw if isinstance(manifest_raw, list) else []
)

# --- Read progress.json via the registry and prepare helpers ---
progress_path = resolve_file_path("progress.json") or "progress.json"
try:
    progress = load_json("progress.json")
except Exception:
    progress = {}

def _is_completed(qkey: str, source: str) -> bool:
    s = (source or "class").lower()
    if s == "shared":
        return bool(progress.get("shared", {}).get(qkey, {}).get("completed"))
    return bool(progress.get("jobs", {}).get(selected_class, {}).get(qkey, {}).get("completed"))

def _saved_step(qkey: str, source: str) -> int:
    s = (source or "class").lower()
    if s == "shared":
        v = progress.get("shared", {}).get(qkey, {}).get("step")
    else:
        v = progress.get("jobs", {}).get(selected_class, {}).get(qkey, {}).get("step")
    try:
        return int(v) if v is not None else 1
    except Exception:
        return 1

def _set_current(qkey: str, step: int):
    progress.setdefault("current", {})
    progress["current"].update({"job": selected_class, "quest": qkey, "step": int(step or 1)})
    # write to resolved file path so it actually persists
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)

# --- Choose the active quest/step ---
current_file = None
current_step = 1
current_quest_key = None

# 1) Resume progress.current if it matches this job and is not completed
cur = progress.get("current") or {}
if cur.get("job") == selected_class and cur.get("quest"):
    qkey = str(cur["quest"]).strip().lower()
    # resolve "<qkey>.json" first, then bare qkey (both via files.json)
    qpath = resolve_file_path(f"{qkey}.json") or resolve_file_path(qkey)
    if qpath:
        # if it’s not completed in either shared or class space, resume it
        if not _is_completed(qkey, "shared") and not _is_completed(qkey, "class"):
            current_file = qpath
            current_step = int(cur.get("step") or _saved_step(qkey, "shared") or _saved_step(qkey, "class") or 1)
            current_quest_key = qkey

# 2) Otherwise, scan the class manifest (in order) and pick the first incomplete quest
if not current_file:
    for entry in manifest_list:
        if not isinstance(entry, dict):
            continue
        qkey = str(entry.get("quest") or "").strip().lower()
        if not qkey:
            continue
        source = str(entry.get("source") or "class").lower()
        if _is_completed(qkey, source):
            continue
        qpath = resolve_file_path(f"{qkey}.json") or resolve_file_path(qkey)
        if not qpath:
            print(f"[!] Could not resolve quest file for {qkey!r} via assets/files.json; skipping.")
            continue
        current_file = qpath
        current_step = _saved_step(qkey, source)
        current_quest_key = qkey
        break

# 3) Nothing to do if all quests are completed
if not current_file:
    print("[+] All quests in manifest appear completed for this job.")
    raise SystemExit(0)

# Persist the 'current' pointer now that we know what to run
_set_current(current_quest_key, current_step)

print(f"[+] Job file: {current_file}")

# Load the QUEST macro file (not the class manifest) and normalize step
data = load_json(current_file)
current_step = 1 if not isinstance(current_step, int) or current_step <= 0 else current_step

# Ensure the loaded step is valid
step_keys = sorted(int(k) for k in data.keys())
if not step_keys:
    print(f"[!] No steps defined in {os.path.basename(current_file)}")
    exit(0)
if str(current_step) not in data:
    next_keys = [k for k in step_keys if k >= current_step]
    if not next_keys:
        # No more steps in this file → drop into interactive console
        print(f"[+] Finished {current_file} at step {current_step-1}.")
        print("[+] Enter a command (/exec <file> [step] [continue] or /resume), or 'exit' to quit.")

        while True:
            user_cmd = input("> ").strip()
            if user_cmd.lower() in ("exit", "quit"):
                print("[+] Exiting.")
                try: logwatch.stop()
                except Exception: pass
                exit(0)
            elif user_cmd:
                # Fake a step with a 'command' entry
                data = { "1": [ { "command": user_cmd } ] }
                reload_jsons()
                current_step = 1
                break
            continue
    current_step = min(next_keys)

last_ocr_text = None
last_ocr_region = None
last_img_template = None

while True:
    # Re-sync current_step with saved progress in case of restart
    saved_step = load_progress(current_file)
    if saved_step > current_step:
        current_step = saved_step

    # Refresh step keys in case file was changed
    step_keys = sorted(int(k) for k in data.keys())
    if not step_keys:
        print(f"[!] No steps defined in {os.path.basename(current_file)}")
        break

    # If saved/loaded step doesn't exist, move to next available or idle
    if str(current_step) not in data:
        next_keys = [k for k in step_keys if k >= current_step]
        if not next_keys:
            # No more steps in this file → open interactive console
            print(f"[+] Finished {os.path.basename(current_file)} at step {current_step-1}.")
            print("[+] Enter a command (e.g. '/exec <quest|file> [step]' or '/resume'), or 'exit' to quit.")
            while True:
                user_cmd = input("> ").strip()
                if not user_cmd:
                    continue
                low = user_cmd.lower()
                if low in ("exit", "quit"):
                    print("[+] Exiting.")
                    exit(0)
                if low.startswith("/resume"):
                    # leave current_file/current_step as-is; re-scan steps next loop
                    break
                if low.startswith("/exec "):
                    parts = user_cmd.split()
                    target = parts[1] if len(parts) >= 2 else ""
                    step_arg = parts[2] if len(parts) >= 3 else None
                    cont_arg = parts[3] if len(parts) >= 4 else None

                    # Allow forms:
                    #   /exec 568_close_to_home
                    #   /exec 568_close_to_home 2
                    #   /exec 568_close_to_home 2 true   (or "continue"/"yes"/"1")
                    #   /exec assets/quests/.../foo.json 3 continue
                    new_file = resolve_file_path(f"{target}.json") or resolve_file_path(target) or target
                    if not _exists_any(new_file):
                        print(f"[!] /exec: could not resolve '{target}' via assets/files.json")
                        continue

                    # Normalize step
                    try:
                        step_num = int(step_arg) if (step_arg is not None and str(step_arg).isdigit()) else 1
                    except Exception:
                        step_num = 1
                    if step_num <= 0: step_num = 1

                    # Update runtime
                    current_file = new_file
                    data = load_json(current_file)
                    current_step = step_num

                    # Update progress.json current pointer and saved step
                    qk = os.path.splitext(os.path.basename(current_file))[0].lower()
                    # ensure the manifest is accessible for source detection
                    globals().setdefault("manifest_list", manifest_list)
                    # save progress for this step (does not mark completed)
                    save_progress(current_file, current_step)

                    # Optional “continue” flag to auto-resume
                    cont = str(cont_arg or "").strip().lower()
                    if cont in ("true", "yes", "1", "continue"):
                        # just break out of console and let the main loop run
                        pass
                    # otherwise we still break out of console; main loop will run from the chosen file/step
                    break

                # default: run as one-off chat command
                data = {"1": [ {"command": user_cmd} ]}
                current_step = 1
                break
            continue
        current_step = min(next_keys)

    step_list = data[str(current_step)]
    print(f"\n[*] Step {current_step} from {os.path.basename(current_file)}:")

    goto_triggered = False

    for step in step_list:
        if not isinstance(step, dict):
            continue

        # keep game focused before actions
        focus_game_window(hwnd)
        time.sleep(0.1)

        # Primitive step handler (keeps runner lean and consistent everywhere)
        if any(k in step for k in ("command","macro","region","ocr","img_search","mouse_move","wait","rand","await","destination")):
            handled = execute_one_step(step, keymap)
            if handled:
                continue

        elif "interaction_type" in step:
            # Format: ["name", {args...}, {more_args...}, ...]
            name, argmap = None, {}
            val = step["interaction_type"]
            if isinstance(val, list) and len(val) >= 1:
                name = val[0]
                # merge any number of dicts after the name (left-to-right)
                for part in val[1:]:
                    if isinstance(part, dict):
                        argmap.update(part)
            if not name:
                print("[!] interaction_type: missing name.")
                continue
            if name not in interaction_types:
                print(f"[!] interaction_type '{name}' not found in interaction_types.json")
                continue
            print(f"  [*] interaction {name} args: {argmap}")
            raw_steps = interaction_types[name]
            expanded_steps = _substitute_params(raw_steps, argmap)

            print(f"  [*] interaction {name}: {len(expanded_steps)} steps")
            focus_game_window(hwnd); time.sleep(0.05)

            for i_idx, i_step in enumerate(expanded_steps, start=1):
                print(f"    [*] interaction {name} step {i_idx}/{len(expanded_steps)}")
                focus_game_window(hwnd); time.sleep(0.05)
                if not execute_one_step(i_step, keymap):
                    print(f"    [!] Unknown interaction step: {i_step}")
            continue

        elif "vnav" in step:
            try:
                legs = normalize_vnav_path(step["vnav"], default_timeout=120000, label="step vnav")
            except ValueError as e:
                print(f"[!] {e}")
                continue

            total = len(legs)
            for i, (coord_str, timeout_ms, tolerance) in enumerate(legs, start=1):
                print(f"  [*] vnav leg {i}/{total} → {coord_str} (tolerance={tolerance})")
                ok = vnav_go(coord_str, timeout_ms, tolerance, keymap)
                if not ok:
                    print(f"[!] vnav leg {i}/{total} failed; aborting remaining legs.")
                    break
                if i < total:
                    dwell = random.randint(250, 350)
                    print(f"  [*] vnav dwell between legs: {dwell}ms")
                    time.sleep(dwell / 1000.0)
            continue

        elif "map" in step:
            map_name = step["map"]
            poi_key = (step.get("location") or step.get("npc") or "").lower()
            data_type = "location" if "location" in step else "npc"
            source_file = locations_data if data_type == "location" else npc_data
            map_entry = source_file.get(map_name)

            if not map_entry:
                print(f"[!] Map '{map_name}' not found in {data_type}s.json.")
                continue

            # Resolve the specific POI entry
            entry = None
            if isinstance(map_entry, dict):
                entry = {k.lower(): v for k, v in map_entry.items()}.get(poi_key)
            elif isinstance(map_entry, list):
                for e in map_entry:
                    if e.get("name", "").lower() == poi_key:
                        entry = e
                        break

            if not entry:
                print(f"[!] POI '{poi_key}' not found in {data_type}s.json.")
                continue

            # 1) Prefer new 'vnav' key if present
            if "vnav" in entry:
                try:
                    legs = normalize_vnav_path(
                        entry["vnav"], default_timeout=120000, label=f"map {map_name}/{poi_key}"
                    )
                except ValueError as e:
                    print(f"[!] {e}")
                    continue

                total = len(legs)
                for i, (coord_str, timeout_ms, tolerance) in enumerate(legs, start=1):
                    print(f"  [*] From {map_name}: vnav leg {i}/{total} → {coord_str} (tolerance={tolerance})")
                    ok = vnav_go(coord_str, timeout_ms, tolerance, keymap)
                    if not ok:
                        print(f"[!] vnav leg {i}/{total} failed; aborting remaining legs for {poi_key}.")
                        break
                    if i < total:
                        dwell = random.randint(250, 350)
                        print(f"  [*] vnav dwell between legs: {dwell}ms")
                        time.sleep(dwell / 1000.0)
                continue

            # 1b) Or passively watch for arrival at a destination (no /vnav command)
            if "destination" in entry:
                try:
                    coord_str, timeout_ms, tolerance = normalize_vnav_value(
                        entry["destination"], default_timeout=120000, default_tolerance=VNAV_DEFAULT_TOLERANCE,
                        label=f"map {map_name}/{poi_key} destination"
                    )
                except ValueError as e:
                    print(f"[!] {e}")
                    continue

                print(f"  [*] From {map_name}: destination check → {coord_str} (tolerance={tolerance})")
                wait_until_near(coord_str, timeout_ms, tolerance)
                continue

            # 2) Legacy combo: ctype + command (macro name in ctype; parameter in command)
            if "ctype" in entry and "command" in entry:
                command_type = entry["ctype"]
                command_value = entry["command"]
                sequence = commands_data.get(command_type)
                if sequence:
                    print(f"  [*] From {map_name}: {command_type} => {command_value}")
                    execute_command_sequence(sequence, keymap, command_value)
                else:
                    print(f"[!] Unknown ctype '{command_type}' in {map_name} for {poi_key}")
                continue

            # 3) Plain 'command': either a chat command (starts with '/') or a macro key
            if "command" in entry:
                command_value = entry["command"]
                if isinstance(command_value, str) and command_value.startswith("/"):
                    print(f"  [*] From {map_name}: sending chat command {command_value}")
                    send_chat_command(command_value, keymap)
                else:
                    sequence = commands_data.get(command_value)
                    if sequence:
                        print(f"  [*] From {map_name}: executing macro {command_value}")
                        execute_command_sequence(sequence, keymap)
                    else:
                        print(f"[!] Unknown command key in {map_name}: {command_value}")
                continue

            print(f"[!] No 'vnav', 'ctype'+'command', or 'command' defined for {poi_key} in {map_name}.")
            continue

        elif "combat" in step:
            handle_combat_step(step["combat"], keymap)
            continue

        elif "goto" in step:
            goto_val = step["goto"]  # supports: 568  |  "568_close_to_home"  |  "…json"  |  [fileOrKeyOrId, step]
            target_key = None
            explicit_step = None
            target_path = None

            # end-of-chain legacy marker: ["", 0] or [null, 0] or 0
            if (isinstance(goto_val, (list, tuple)) and len(goto_val) >= 2 and not goto_val[0] and str(goto_val[1]) == "0") \
               or (goto_val == 0):
                # Save the just-executed step and mark current quest completed, then exit
                save_progress(current_file, current_step)
                try:
                    _root = _progress_load_root()
                    _q = _quest_key_for_path(current_file)
                    _src = _quest_source_for(_q, manifest_list)
                    if _src == "shared":
                        _root.setdefault("shared", {}).setdefault(_q, {})["completed"] = True
                    else:
                        _root.setdefault("jobs", {}).setdefault(selected_class, {}).setdefault(_q, {})["completed"] = True
                    _progress_save_root(_root)
                except Exception:
                    pass
                print("[+] Reached end-of-chain marker. Exiting script.")
                exit(0)

            def _stem(s: str) -> str:
                b = os.path.basename(str(s)).lower()
                return b[:-5] if b.endswith(".json") else b

            # Parse all supported forms
            if isinstance(goto_val, int):
                # new form: { "goto": 568 }
                target_key = str(goto_val)
            elif isinstance(goto_val, str):
                # key or path
                if _exists_any(goto_val):
                    target_path = _to_abs(goto_val)
                    target_key = _stem(goto_val)
                else:
                    target_key = _stem(goto_val)
            elif isinstance(goto_val, (list, tuple)) and len(goto_val) >= 1:
                g0 = goto_val[0]
                if len(goto_val) >= 2:
                    try:
                        explicit_step = int(goto_val[1])
                    except Exception:
                        explicit_step = None
                if isinstance(g0, int):
                    target_key = str(g0)
                elif isinstance(g0, str):
                    if _exists_any(g0):
                        target_path = _to_abs(g0)
                        target_key = _stem(g0)
                    else:
                        target_key = _stem(g0)
                else:
                    print("[!] goto: unsupported target type in list.")
                    continue
            else:
                print("[!] goto: unsupported format.")
                continue

            # Resolve target path via registry helper first, then by key
            if not target_path and target_key:
                try:
                    # resolve_quest_target(key) → (full_path, quest_key)
                    res = resolve_quest_target(target_key)
                    if isinstance(res, tuple) and len(res) >= 1:
                        target_path = res[0]
                        # prefer normalized quest key if provided
                        if len(res) >= 2 and res[1]:
                            target_key = res[1]
                except Exception:
                    target_path = None
                if not target_path:
                    target_path = resolve_file_path(f"{target_key}.json") or resolve_file_path(target_key)

            if not target_path:
                print(f"[!] GOTO target not found: {goto_val!r}")
                continue

            # Determine next step: explicit > saved > 1
            if explicit_step is not None and explicit_step > 0:
                next_step = explicit_step
            else:
                try:
                    next_step = load_progress(target_path)
                    if not isinstance(next_step, int) or next_step <= 0:
                        next_step = 1
                except Exception:
                    next_step = 1

            print(f"[*] GOTO requested → {target_path}, step {next_step}")

            # Save progress of the step we just performed in the current file
            save_progress(current_file, current_step)

            # Mark the current quest completed in progress.json
            try:
                _root = _progress_load_root()
                _q = _quest_key_for_path(current_file)
                _src = _quest_source_for(_q, manifest_list)
                if _src == "shared":
                    _root.setdefault("shared", {}).setdefault(_q, {})["completed"] = True
                else:
                    _root.setdefault("jobs", {}).setdefault(selected_class, {}).setdefault(_q, {})["completed"] = True
                _progress_save_root(_root)
            except Exception as e:
                print(f"[!] Failed to mark quest completed: {e}")

            # Switch file and set requested step
            current_file = target_path
            data = load_json(current_file)
            current_step = int(next_step)

            # Update current pointer
            try:
                _root = _progress_load_root()
                _root.setdefault("current", {})
                _root["current"].update({
                    "job": selected_class,
                    "quest": _quest_key_for_path(current_file),
                    "step": current_step
                })
                _progress_save_root(_root)
            except Exception:
                pass

            # Reset OCR/IMG context on file switch
            last_ocr_text = None
            last_ocr_region = None
            last_img_template = None

            goto_triggered = True
            break
    
    save_progress(current_file, current_step)
    current_step += 1

    # After finishing ALL items in this step_list:
    if goto_triggered:
        # the while-loop restarts with the new file/step
        continue
    
    # Check if this step_list was a pure goto-only step
    only_goto = all(isinstance(s, dict) and "goto" in s for s in step_list)

    if not only_goto:
        # Save the step number we JUST performed (no +1)
        save_progress(current_file, current_step)

    if single_run_mode:
        print("[+] Single-run step executed. Exiting script.")
        exit(0)

    # If this was the last step in the current file, drop into interactive mode
    if current_step == max(step_keys):
        print(f"[+] Finished {os.path.basename(current_file)} at step {current_step}.")
        print("[+] Enter a command (/exec <file> [step] [continue], /resume, or 'exit' to quit.)")

        while True:
            user_cmd = input("> ").strip()
            if not user_cmd:
                continue

            if user_cmd.lower() in ("exit", "quit"):
                print("[+] Exiting.")
                try: logwatch.stop()
                except Exception: pass
                exit(0)

            elif user_cmd.startswith("/exec") or user_cmd.startswith("/exe"):
                parts = user_cmd.split()
                if len(parts) >= 2:
                    target_file = parts[1]
                    target_step = int(parts[2]) if len(parts) >= 3 else 1
                    do_continue = (parts[3].lower() == "true") if len(parts) >= 4 else True
                    if os.path.exists(target_file):
                        print(f"[*] EXEC requested → {target_file}, step {target_step}, continue={do_continue}")
                        reload_jsons()
                        current_file = target_file
                        data = load_json(current_file)
                        current_step = target_step
                        if not do_continue:
                            single_run_mode = True
                        break
                    else:
                        print(f"[!] EXEC file not found: {target_file}")
                else:
                    print("[!] Usage: /exec <file> [step] [continue]")

            elif user_cmd.startswith("/resume"):
                print("[*] RESUME requested.")
                saved_step = load_progress(current_file)
                if saved_step:
                    reload_jsons()
                    current_step = saved_step + 1
                    data = load_json(current_file)
                    break
                else:
                    print("[!] No saved progress to resume from.")

            else:
                # Treat as regular chat command
                print(f"  [*] Sending chat command: {user_cmd}")
                send_chat_command(user_cmd, keymap)
                break

        continue

    # Then advance to the next step for execution (but do not save it yet)
    current_step += 1