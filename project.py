import ctypes
import random
import time
import json
import os
from datetime import datetime, timezone, timedelta
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

# Import ACT log watcher module
from modules.act_log_watcher import ACTLogWatcher

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

PROGRESS_STATE_FILE = "progress.json"

def _to_abs(p: str) -> str:
    """Return absolute path; if relative, anchor at BASE_DIR."""
    return _normalize_slashes(p if os.path.isabs(p) else os.path.join(BASE_DIR, p))

def _exists_any(p: str) -> bool:
    """True if path exists either as-is or anchored at BASE_DIR."""
    return os.path.exists(p) or os.path.exists(_to_abs(p))

def _normalize_slashes(path: str) -> str:
    return path.replace('\\', '/')

def _assets_fullpath(relpath: str) -> str:
    r = str(relpath).lstrip("/\\")
    low = r.lower()
    if low.startswith("assets/") or low.startswith(f"assets{os.sep}"):
        return _to_abs(r)
    return _normalize_slashes(_to_abs(os.path.join("assets", r)))

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

# --- GOTO call stack for nested quest jumps ---
GOTO_STACK = []  # each frame: (return_file_fullpath, return_step_int)

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
VNAV_OCR_INTERVAL_MS = 100 # how often to re-OCR the coords (reduced from 250ms for more responsive updates)
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
def _iter_ocr_configs(langs, psms):
    for lang in langs:
        for psm in psms:
            yield lang, psm

def _try_ocr_image(img, lang, psm, timeout):
    try:
        result = pytesseract.image_to_data(
            img,
            output_type=pytesseract.Output.DICT,
            config=f"--psm {psm}",
            timeout=timeout,
            lang=lang
        )
        return result, None
    except Exception as exc:
        return None, exc

def _ocr_has_tokens(payload):
    return any((token or "").strip() for token in payload.get("text", []))

def _empty_ocr_payload():
    return {
        "text": [],
        "left": [],
        "top": [],
        "width": [],
        "height": [],
        "block_num": [],
        "par_num": [],
        "line_num": []
    }

def _debug_ocr_success(lang, psm, payload):
    if not OCR_DEBUG_DUMP:
        return
    non_empty = sum(1 for token in payload.get("text", []) if token and token.strip())
    print(f"[dbg][ocr] used lang='{lang}' psm={psm} tokens={non_empty}")

def _ocr_image_to_data(img, timeout_sec=None):
    psms = [6, 7, 3, 11, 13]
    langs = ['eng+ffxiv', 'eng']
    timeout = timeout_sec if timeout_sec is not None else TESSERACT_TIMEOUT_SEC

    last_exc = None
    for lang, psm in _iter_ocr_configs(langs, psms):
        payload, err = _try_ocr_image(img, lang, psm, timeout)
        if err is not None:
            last_exc = err
            continue
        if not _ocr_has_tokens(payload):
            continue
        _debug_ocr_success(lang, psm, payload)
        return payload, lang, psm

    if last_exc:
        print(f"[dbg][ocr] OCR attempts failed; last error: {last_exc}")

    return _empty_ocr_payload(), None, None

# --- token span search helpers ---
def _split_target_words(target_norm):
    return [word for word in target_norm.split() if word]

def _span_length_range(token_count, target_len, max_extra):
    if token_count <= 0 or target_len <= 0:
        return range(0)
    min_len = max(1, target_len - max_extra)
    max_len = min(token_count, target_len + max_extra)
    return range(min_len, max_len + 1)

def _iter_candidate_chunks(tokens_norm, lengths):
    token_count = len(tokens_norm)
    for window_len in lengths:
        limit = token_count - window_len + 1
        if limit <= 0:
            continue
        for start in range(limit):
            end = start + window_len
            yield start, end, " ".join(tokens_norm[start:end])

def _exact_span_for_target(tokens_norm, target_norm, target_words, max_extra):
    if target_norm not in " ".join(tokens_norm):
        return None
    default_span = (0, len(tokens_norm) - 1, 1.0)
    lengths = _span_length_range(len(tokens_norm), len(target_words), max_extra)
    for start, end, chunk in _iter_candidate_chunks(tokens_norm, lengths):
        if target_norm in chunk:
            return (start, end - 1, 1.0)
    return default_span

def _fuzzy_span_for_target(tokens_norm, target_norm, target_words, max_extra, matcher_factory):
    best_ratio, best_span = 0.0, None
    lengths = _span_length_range(len(tokens_norm), len(target_words), max_extra)
    for start, end, chunk in _iter_candidate_chunks(tokens_norm, lengths):
        ratio = matcher_factory(None, chunk, target_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_span = (start, end - 1, ratio)
    return best_span

def _normalize_target_text(target_text):
    words = [_norm_token_generic(part) for part in target_text.strip().split()]
    words = [w for w in words if w]
    return words, " ".join(words)

def _span_center_from_boxes(boxes, start, end):
    xs = [boxes[i][0] for i in range(start, end + 1)]
    ys = [boxes[i][1] for i in range(start, end + 1)]
    xe = [boxes[i][0] + boxes[i][2] for i in range(start, end + 1)]
    ye = [boxes[i][1] + boxes[i][3] for i in range(start, end + 1)]
    x0, y0, x1, y1 = min(xs), min(ys), max(xe), max(ye)
    return ((x0 + x1) // 2, (y0 + y1) // 2)

def _locate_exact_ocr_match(lines, target_words):
    if not target_words:
        return None
    span_len = len(target_words)
    for info in lines:
        norm_tokens = info.get('norm_tokens') or []
        if len(norm_tokens) < span_len:
            continue
        for start in range(len(norm_tokens) - span_len + 1):
            if norm_tokens[start:start + span_len] == target_words:
                center = _span_center_from_boxes(info['boxes'], start, start + span_len - 1)
                return center
    return None

def _locate_fuzzy_ocr_match(lines, target_norm_line, threshold):
    if not target_norm_line:
        return None
    best_ratio, best_center, best_text = 0.0, None, None
    for info in lines:
        norm_tokens = info.get('norm_tokens') or []
        if not norm_tokens:
            continue
        span = _best_span_for_target(norm_tokens, target_norm_line, max_extra=2)
        if not span:
            continue
        start, end, ratio = span
        chunk_norm = " ".join(norm_tokens[start:end + 1])
        if ratio < threshold and target_norm_line not in chunk_norm:
            continue
        if ratio >= best_ratio:
            center = _span_center_from_boxes(info['boxes'], start, end)
            matched_text = " ".join(norm_tokens[start:end + 1])
            best_ratio, best_center, best_text = ratio, center, matched_text
    if best_center is not None:
        return best_ratio, best_center, best_text
    return None

def _best_fuzzy_ratio(lines, target_norm_line):
    best_ratio = 0.0
    for info in lines:
        norm_tokens = info.get('norm_tokens') or []
        if not norm_tokens:
            continue
        span = _best_span_for_target(norm_tokens, target_norm_line, max_extra=2)
        if span:
            best_ratio = max(best_ratio, span[2])
    return best_ratio

def _handle_ocr_miss(results, region, target_text, target_norm_line):
    try:
        _dump_ocr_debug(results, region, target_norm_line)
        lines_dbg = _group_ocr_lines(results, region)
        if lines_dbg:
            preview = " | ".join(ln['raw'] for ln in lines_dbg[:3])
            print(f"[dbg][ocr] preview: {preview}")
    except Exception:
        pass
    print(f"[!] Text '{target_text}' not found on screen.")
    return False

# --- pick the best token-span inside a line for a target (exact or fuzzy) ---
def _best_span_for_target(tokens_norm, target_norm, max_extra=2):
    import difflib as _dl
    if not tokens_norm or not target_norm:
        return None
    target_words = _split_target_words(target_norm)
    if not target_words:
        return None
    exact = _exact_span_for_target(tokens_norm, target_norm, target_words, max_extra)
    if exact:
        return exact
    return _fuzzy_span_for_target(tokens_norm, target_norm, target_words, max_extra, _dl.SequenceMatcher)

def perform_ocr_and_find_text(target_text, region=None):
    global last_ocr_match_center, last_detection_type

    print(f"[*] Performing OCR looking for: '{target_text}'")
    screenshot = ImageGrab.grab(bbox=region)

    results, _lang, _psm = _ocr_image_to_data(screenshot, timeout_sec=TESSERACT_TIMEOUT_SEC)
    target_words, target_norm_line = _normalize_target_text(target_text)
    lines = _group_ocr_lines(results, region)

    exact_center = _locate_exact_ocr_match(lines, target_words)
    if exact_center:
        last_ocr_match_center = exact_center
        last_detection_type = "ocr"
        print(f"[*] Found '{target_text}' at {exact_center} [exact]")
        return True

    fuzzy_hit = None
    if OCR_FUZZY_ENABLED and target_norm_line:
        fuzzy_hit = _locate_fuzzy_ocr_match(lines, target_norm_line, OCR_FUZZY_THRESHOLD)
    if fuzzy_hit:
        ratio, center, matched_text = fuzzy_hit
        last_ocr_match_center = center
        last_detection_type = "ocr"
        print(f"[*] Fuzzy matched '{target_text}' ~ '{matched_text}' ({ratio*100:.1f}%) at {center}")
        return True

    return _handle_ocr_miss(results, region, target_text, target_norm_line)

def perform_img_search(template_path, threshold=0.8):
    global last_img_match_center, last_detection_type

    orig = template_path
    template_path = resolve_image_path(template_path) or template_path
    print(f"[*] Performing image search with template: {orig} â†’ {template_path}")

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
STOPWORDS_COMMON = {
    'the', 'a', 'an', 'to', 'for', 'with', 'and', 'or', 'you', 'your', 'from', 'at', 'on', 'in', 'it', 'is', 'are', 'was', 'were', 'of', 'this', 'that', 'these', 'those'
}

CONSOLE_PROMPT = "[+] Enter a command (e.g. '/exec <quest|file> [step]' or '/resume'), or 'exit' to quit."

def _normalize_chat_text(s: str) -> str:
    s = (s or "").lower()
    s = s.translate({0x2019: ord("'"), 0x2018: ord("'"), 0x0060: ord("'")})
    s = s.rstrip('.')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _important_tokens(text: str) -> set[str]:
    tokens = [tok for tok in text.split() if tok and tok not in STOPWORDS_COMMON]
    return set(tokens)

def perform_ocr_find_text_fuzzy(target_text: str, region=None, threshold: float = 0.90,
                                ocr_timeout_sec: float | None = None) -> bool:
    """Fuzzy search across the region; centers on the matched token span when successful."""
    global last_ocr_match_center, last_detection_type

    target_norm = _normalize_chat_text(target_text)
    target_tokens = _important_tokens(target_norm)
    print(f"[*] Performing FUZZY OCR looking for (~{int(threshold*100)}%): '{target_norm}'")

    screenshot = ImageGrab.grab(bbox=region)
    timeout = ocr_timeout_sec if ocr_timeout_sec is not None else TESSERACT_TIMEOUT_SEC
    results, _lang, _psm = _ocr_image_to_data(screenshot, timeout_sec=timeout)
    lines = _group_ocr_lines(results, region)

    match = _locate_fuzzy_ocr_match(lines, target_norm, threshold)
    if match:
        ratio, center, matched_text = match
        matched_norm = _normalize_chat_text(matched_text)
        matched_tokens = _important_tokens(matched_norm)
        if target_tokens and not target_tokens.issubset(matched_tokens):
            if OCR_DEBUG_DUMP:
                missing = sorted(target_tokens - matched_tokens)
                print(f"[dbg][ocr] rejecting fuzzy hit; missing tokens {missing}")
        else:
            last_ocr_match_center = center
            last_detection_type = "ocr"
            print(f"[*] Fuzzy match {ratio:.1%} for '{target_norm}' ~ '{matched_text}' at {center}")
            return True
    best_ratio = _best_fuzzy_ratio(lines, target_norm)
    try:
        _dump_ocr_debug(results, region, target_norm)
    except Exception:
        pass
    print(f"[!] Fuzzy OCR below threshold ({best_ratio:.1%}) for '{target_norm}'")
    return False

def move_mouse_hardware(x, y):
    print(f"[*] Moving mouse to ({x}, {y})")
    user32.SetCursorPos(x, y)

def press_key(hex_key_code):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hex_key_code))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

def release_key(hex_key_code):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hex_key_code, dwFlags=KEYEVENTF_KEYUP))
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
        # Exact-token replacement (preserve type): "â€¦": "$var"
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
    press_key(enter_vk); time.sleep(0.05); release_key(enter_vk)
    time.sleep(0.2)

    # Paste
    ctrl = int(keymap["VK_LCONTROL"], 16)
    vkey = int(keymap["V"], 16)
    press_key(ctrl); time.sleep(0.05)
    press_key(vkey); time.sleep(0.05)
    release_key(vkey); time.sleep(0.05)
    release_key(ctrl); time.sleep(0.2)

    # Send
    press_key(enter_vk); time.sleep(0.05); release_key(enter_vk)
    time.sleep(0.1)

def _macro_arg(macro_array, index, offset, cmd):
    try:
        return macro_array[index + offset]
    except IndexError:
        print(f"[!] Macro '{cmd}' expects more arguments (index {index})")
        return None

def _handle_mouse_button(action, button_id):
    btn, down_flag, up_flag = mouse_buttons[button_id]
    flag = down_flag if action == 'press' else up_flag
    print(f"[*] {'Pressing' if action == 'press' else 'Releasing'} {btn} mouse button")
    extra = 0
    if 'XBUTTON' in button_id:
        extra = 1 if btn == 'x1' else 2
    user32.mouse_event(flag, 0, 0, 0, extra)

def _handle_virtual_key(action, key_name, keymap):
    try:
        raw_code = keymap[key_name]
    except KeyError:
        print(f"[!] Unknown key name: {key_name}")
        return
    try:
        key_code = int(raw_code, 16)
    except (TypeError, ValueError):
        print(f"[!] Invalid key code for {key_name}: {raw_code}")
        return
    print(f"[*] {'Pressing' if action == 'press' else 'Releasing'} {key_name}")
    if action == 'press':
        press_key(key_code)
    else:
        release_key(key_code)

def _sleep_ms(duration_ms):
    try:
        time.sleep(float(duration_ms) / 1000.0)
    except (TypeError, ValueError):
        print(f"[!] Invalid wait value: {duration_ms}")

def _handle_press_release(action, macro_array, index, keymap):
    target = _macro_arg(macro_array, index, 1, action)
    if target is None:
        return len(macro_array)
    if target in mouse_buttons:
        _handle_mouse_button(action, target)
    else:
        _handle_virtual_key(action, target, keymap)
    return index + 2

def _handle_press(macro_array, index, keymap):
    return _handle_press_release('press', macro_array, index, keymap)

def _handle_release(macro_array, index, keymap):
    return _handle_press_release('release', macro_array, index, keymap)

def _handle_wait(macro_array, index, keymap):
    duration = _macro_arg(macro_array, index, 1, 'wait')
    if duration is None:
        return len(macro_array)
    print(f"[*] Wait: {duration}ms")
    _sleep_ms(duration)
    return index + 2

def _handle_rand(macro_array, index, keymap):
    low = _macro_arg(macro_array, index, 1, 'rand')
    high = _macro_arg(macro_array, index, 2, 'rand')
    if low is None or high is None:
        return len(macro_array)
    try:
        low_i = int(low)
        high_i = int(high)
    except (TypeError, ValueError):
        print(f"[!] Invalid rand bounds: {low}, {high}")
        return index + 3
    if low_i > high_i:
        low_i, high_i = high_i, low_i
    delay = random.randint(low_i, high_i)
    print(f"[*] Random wait: {delay}ms")
    _sleep_ms(delay)
    return index + 3

_MACRO_HANDLERS = {
    'press': _handle_press,
    'release': _handle_release,
    'wait': _handle_wait,
    'rand': _handle_rand,
}

def _set_current_pointer_state(file_path: str, step: int) -> None:
    global progress
    try:
        root = _progress_load_root()
        root.setdefault('current', {})
        root['current'].update({
            'job': selected_class,
            'quest': _quest_key_for_path(file_path),
            'step': int(step),
        })
        _progress_save_root(root)
        progress = root
    except Exception:
        pass

def _write_progress_entry(file_path: str, step: int, *, completed: bool, update_current: bool) -> None:
    global progress
    try:
        step_value = int(step)
    except Exception:
        step_value = 1
    if step_value <= 0:
        step_value = 1
    root = _progress_load_root()
    quest_key = _quest_key_for_path(file_path)
    source = _quest_source_for(quest_key, globals().get('manifest_list'))
    if source == 'shared':
        entry = root.setdefault('shared', {}).setdefault(quest_key, {})
    else:
        entry = root.setdefault('jobs', {}).setdefault(selected_class, {}).setdefault(quest_key, {})
    entry['completed'] = bool(completed)
    entry['step'] = step_value
    if update_current:
        root.setdefault('current', {})
        root['current'].update({'job': selected_class, 'quest': quest_key, 'step': step_value})
    _progress_save_root(root)
    progress = root

def _mark_quest_completed(file_path: str, last_step: int | None) -> None:
    step_value = last_step
    try:
        if step_value is None:
            step_value = load_progress(file_path)
        step_value = int(step_value)
    except Exception:
        step_value = 1
    if step_value <= 0:
        step_value = 1
    _write_progress_entry(file_path, step_value, completed=True, update_current=False)

def _return_to_goto_caller() -> bool:
    global current_file, current_step, data
    if not GOTO_STACK:
        return False
    ret_file, ret_step = GOTO_STACK.pop()
    try:
        data = load_json(ret_file)
        current_file = ret_file
        current_step = int(ret_step)
        _set_current_pointer_state(current_file, current_step)
        print(f'[*] Returning to {os.path.basename(current_file)} at step {current_step} (via goto stack).')
        return True
    except Exception as exc:
        print(f'[!] Failed to return to caller: {exc}')
        return False

def _classify_console_command(raw: str):
    cmd = raw.strip()
    if not cmd:
        return 'empty', cmd
    low = cmd.lower()
    if low in ('exit', 'quit'):
        return 'exit', cmd
    if low.startswith('/resume'):
        return 'resume', cmd
    if low.startswith('/exec') or low.startswith('/exe'):
        return 'exec', cmd
    return 'chat', cmd

def _resolve_exec_command(command: str):
    parts = command.split()
    if len(parts) < 2:
        print('[!] Usage: /exec <quest|file> [step] [continue]')
        return None
    target = parts[1]
    step_arg = parts[2] if len(parts) >= 3 else None
    cont_arg = parts[3] if len(parts) >= 4 else None
    path_candidate = resolve_file_path(f"{target}.json") or resolve_file_path(target) or target
    if not _exists_any(path_candidate):
        print(f"[!] /exec: could not resolve '{target}' via assets/files.json")
        return None
    try:
        step_num = int(step_arg) if (step_arg is not None and str(step_arg).isdigit()) else 1
    except Exception:
        step_num = 1
    if step_num <= 0:
        step_num = 1
    single_run = bool(cont_arg and str(cont_arg).lower() in {'false', '0', 'no'})
    return path_candidate, step_num, single_run

def _interactive_console_loop(prompt: str) -> None:
    global current_file, current_step, data, single_run_mode
    print(prompt)
    while True:
        user_cmd = input('> ').strip()
        kind, payload = _classify_console_command(user_cmd)
        if kind == 'empty':
            continue
        if kind == 'exit':
            print('[+] Exiting.')
            sys.exit(0)
        if kind == 'resume':
            _resume_from_saved_step()
            return
        if kind == 'exec':
            parsed = _resolve_exec_command(payload)
            if not parsed:
                continue
            new_file, step_num, single_run = parsed
            current_file = new_file
            data = load_json(current_file)
            current_step = step_num
            save_progress(current_file, current_step)
            _set_current_pointer_state(current_file, current_step)
            if single_run:
                single_run_mode = True
            return
        data = {'1': [{'command': payload}]}
        current_step = 1
        return

def _finalize_current_quest(last_step: int, prompt: str) -> None:
    print(f"[+] Finished {os.path.basename(current_file)} at step {last_step}.")
    _mark_quest_completed(current_file, last_step)
    if _return_to_goto_caller():
        return
    try:
        if _auto_advance_to_next_incomplete():
            return
    except Exception as exc:
        print(f"[!] Auto-advance failed: {exc}")
    _interactive_console_loop(prompt)

def _finalize_and_prompt(last_step: int) -> bool:
    _finalize_current_quest(last_step, CONSOLE_PROMPT)
    return True

def execute_command_sequence(macro_array, keymap, command_text=None):
    _ = command_text  # preserved for compatibility
    index = 0
    length = len(macro_array)
    while index < length:
        cmd = macro_array[index]
        handler = _MACRO_HANDLERS.get(cmd)
        if handler is None:
            print(f"[!] Unknown macro command: {cmd}")
            index += 1
            continue
        next_index = handler(macro_array, index, keymap)
        if next_index <= index:
            index += 1
        else:
            index = next_index
    return

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
        return load_json(PROGRESS_STATE_FILE) or {}
    except Exception:
        return {}

def _progress_save_root(root: dict) -> None:
    p = resolve_file_path(PROGRESS_STATE_FILE) or _to_abs(PROGRESS_STATE_FILE)
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
    _write_progress_entry(file_path, step, completed=False, update_current=True)

def resolve_goto_chain(file, step):
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
            print(f"[*] Resolving startup GOTO â†’ {target_file}, step {target_step}")
            file = target_file
            step = int(target_step)
        else:
            print(f"[!] GOTO target file not found: {target_file}")
            break
    return file, step

def _cycle_target_selection(keymap):
    execute_command_sequence(["press", "VK_TAB", "rand", 56, 126, "release", "VK_TAB"], keymap)
    time.sleep(1.5)

def _pick_target_via_ocr(candidates, limits, kills, region, combat_type=None):
    for name in candidates:
        found = perform_ocr_and_find_text(name, region)
        
        if found:
            # Check if this was a partial match of a longer name
            # e.g. if looking for "Quiveron" but found "Quiveron guard", don't count it
            # as "Quiveron" until we've seen a defeat message
            if kills[name] >= limits[name]:
                print(f"[i] Already met quota for {name}, skipping.")
                continue
            
            return name
    return None

def _move_mouse_to_last_detection():
    if last_detection_type == "ocr" and last_ocr_match_center:
        move_mouse_hardware(*last_ocr_match_center)
        time.sleep(0.2)
        return True
    if last_detection_type == "img" and last_img_match_center:
        move_mouse_hardware(*last_img_match_center)
        time.sleep(0.2)
        return True
    print("[!] No valid target position available for mouse move.")
    return False

def _open_with_mouse(keymap):
    execute_command_sequence(["press", "VK_RBUTTON", "rand", 40, 80, "release", "VK_RBUTTON"], keymap)
    time.sleep(0.3)
    execute_command_sequence(["press", "VK_LBUTTON", "rand", 40, 80, "release", "VK_LBUTTON"], keymap)

def _is_still_in_combat(act_log_watcher, timeout=5.0):
    try:
        with act_log_watcher._combat_log.open("r", encoding="utf-8") as f:
            lines = f.readlines()[-20:]  # Get last 20 messages
            
        if not lines:
            return False

        # Parse last line's timestamp to get timezone info
        latest_ts = None
        for line in reversed(lines):
            try:
                ts_str = line.split(" | ", 1)[0].strip()
                latest_ts = datetime.fromisoformat(ts_str)
                break
            except Exception:
                continue
                
        if not latest_ts:
            return False
            
        # Use same timezone as log files for comparison
        current_time = datetime.now(latest_ts.tzinfo)
        cutoff_time = current_time - timedelta(seconds=timeout)
        
        for line in reversed(lines):  # Start with most recent
            try:
                timestamp, message = line.split(" | ", 1)
                # Parse actual timestamp for accurate comparison
                ts = datetime.fromisoformat(timestamp.strip())
                
                # Skip messages older than our cutoff
                if ts < cutoff_time:
                    continue
                    
                msg_lower = message.lower().strip()
                if any(x in msg_lower for x in ["hits you for", "you hit"]):
                    return True
            except Exception as e:
                print(f"[DEBUG] Error parsing combat line: {e}")
                continue
                
        return False
    except Exception as e:
        print(f"[!] Error checking combat status: {e}")
        return False

def _await_defeat_line(target, keymap, kills, limits, valid_targets=None):
    """Wait for target defeat using ACT log monitoring"""
    global act_log_watcher
    
    start_time = time.time()
    combat_timeout = 90  # Maximum time to wait for combat
    checking_defeat = True
    last_combat_time = time.time()  # Initialize this at the start
    
    # Wait for combat to start first
    print("[*] Waiting for combat to begin...")
    combat_wait_start = time.time()
    while time.time() - combat_wait_start < 5.0:  # Give 5 seconds to start combat
        if _is_still_in_combat(act_log_watcher):
            print("[*] Combat detected, monitoring for defeats...")
            last_combat_time = time.time()  # Update when combat is detected
            break
        time.sleep(0.1)
    
    while time.time() - start_time < combat_timeout:
        # Check for any defeats
        if checking_defeat:
            act_result, defeated_target = _await_combat_completion_via_act(target, timeout_sec=2.0, valid_targets=valid_targets)
            if act_result:
                # Find the actual name from our target list that matches (preserving case)
                defeated_target_lower = defeated_target.lower()
                actual_name = next((name for name in valid_targets if name.lower() == defeated_target_lower), target)
                
                # Update kills for the actually defeated target
                if actual_name in kills:
                    # Update kill counter
                    old_count = kills[actual_name]
                    new_count = min(limits[actual_name], kills[actual_name] + 1)
                    kills[actual_name] = new_count
                    
                    # If the kill counted toward our total
                    if new_count > old_count:
                        progress_str = ", ".join(f"{name}: {kills[name]}/{limits[name]}" for name in limits)
                        print(f"[DEBUG][Combat] +1 {actual_name}")
                        print(f"[*] Progress: [{progress_str}]")
                        
                        # Check if we've met all our target counts
                        all_complete = all(kills[name] >= limits[name] for name in limits)
                        if all_complete:
                            print("[+] All targets defeated, exiting combat")
                            return True
                    
                    # If we're still in combat, keep monitoring
                    if _is_still_in_combat(act_log_watcher):
                        last_combat_time = time.time()
                        print("[DEBUG] Got a defeat but still in combat, continuing to monitor...")
                        checking_defeat = True
                        continue
                    else:
                        print("[DEBUG] Combat appears to be over")
                        return True
                else:
                    print(f"[!] Defeated {actual_name} but it wasn't in our target list!")
                    checking_defeat = True  # Keep checking for valid defeats
                    continue
            elif _is_still_in_combat(act_log_watcher):
                # Still in combat but no defeat detected
                last_combat_time = time.time()
                checking_defeat = True
                continue
            elif time.time() - last_combat_time > 2.0:  # If no combat for 2 seconds
                # No recent combat activity and no defeat detected
                print("[DEBUG] No combat activity for 2 seconds, likely finished current engagement")
                return False
        
        # Brief pause before next check
        time.sleep(0.1)
    
    print(f"[!] Combat monitoring timed out after {combat_timeout}s")
    return False

def _rotate_for_additional_targets(keymap):
    execute_command_sequence(["press", "VK_RIGHT", "rand", 156, 256, "release", "VK_RIGHT"], keymap)
    time.sleep(0.3)

def _combat_timed_out(start_time, limit_seconds=300):
    if time.time() - start_time > limit_seconds:
        print("[!] Combat timed out.")
        return True
    return False

def _handle_no_target_found():
    print("[!] No valid target found; waiting briefly before retrying...")
    time.sleep(0.5)

def _finalize_combat(end_command, keymap):
    print(f"[+] Combat finished. Returning to scout position: {end_command}")
    if end_command:
        send_chat_command(end_command, keymap)

def _engage_target(name, keymap, kills, limits, style='full', valid_targets=None):
    print(f"[*] Engaging {name}...")
    _move_mouse_to_last_detection()
    _open_with_mouse(keymap)
    start_combat(keymap, style)
    time.sleep(0.5)
    # Pass all valid targets for completion
    _await_defeat_line(name, keymap, kills, limits, valid_targets=valid_targets)
    end_combat(keymap)

def start_combat(keymap, style="rotation"):
    print(f"[*] Starting combat rotation (style={style})...")

    if style in ("full", "rotation"):
        send_chat_command("/rotation Manual", keymap)
        time.sleep(0.2)

    if style in ("full", "ai"):
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

def _build_target_limits(names, counts):
    default = counts[0] if counts else 1
    return {name: (counts[idx] if idx < len(counts) else default)
            for idx, name in enumerate(names)}

def handle_combat_step(combat_data, keymap):
    names = combat_data.get('names', [])
    counts = combat_data.get('counts', [])
    end_command = combat_data.get('end_command')
    style = combat_data.get('style', 'full')  # Default to 'full' if not specified
    combat_type = combat_data.get('type')     # Default to None if not specified

    target_limits = _build_target_limits(names, counts)
    if not target_limits:
        print('[!] No combat targets provided.')
        return

    print(f"[*] Combat started. Targets: {target_limits}")
    print(f"[*] Combat type: {combat_type or 'default'}, Style: {style}")

    kills = {name: 0 for name in target_limits}
    start_time = time.time()
    name_region = (750, 31, 1190, 52)

    while True:
        # Check if all targets are complete before doing anything else
        remaining = [name for name, limit in target_limits.items() if kills[name] < limit]
        if not remaining:
            print('[+] All requested kills completed.')
            break

        _cycle_target_selection(keymap)
        target = _pick_target_via_ocr(remaining, target_limits, kills, name_region, combat_type)

        if target:
            _engage_target(target, keymap, kills, target_limits, style=style, valid_targets=names)
        else:
            _handle_no_target_found()

        if _combat_timed_out(start_time):
            break

        _rotate_for_additional_targets(keymap)

    _finalize_combat(end_command, keymap)

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
            config="--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789.-, "
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
    print(f"  [*] vnav â†’ {cmd} (tolerance={tolerance})")
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

    print(f"  [*] destination â†’ target {coord_str} (tolerance={tolerance})")

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

# --- Combat monitoring via ACT logs ---
def _await_combat_completion_via_act(target_name, timeout_sec=90, valid_targets=None):
    """Wait for combat completion using ACT logs
    
    Args:
        target_name: Primary target we're tracking progress for
        timeout_sec: How long to wait for completion
        valid_targets: Optional list of alternative target names that can count as completion
        
    Returns:
        tuple: (bool, str) - (Success, Defeated target name) or (False, None) on timeout/error
    """
    if not act_log_watcher:
        print("[!] ACT log watcher not available")
        return False, None
    
    try:
        # Convert all targets to lowercase for comparison
        valid_targets_lower = {t.lower(): t for t in (valid_targets or [target_name])}
        
        start_time = time.time()
        last_combat_time = None  # Will be set when we first detect combat
        combat_started = False   # Flag to track if we've entered combat
        combat_reported = False  # Flag to avoid duplicate combat state messages
        
        # Get the current timestamp from ACT logs
        current_ts = None
        try:
            with act_log_watcher._combat_log.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-1:]  # Just get last line
                if lines:
                    current_ts = lines[0].split(" | ", 1)[0].strip()
        except Exception:
            pass
            
        no_combat_start = None  # Track when we first noticed no combat after being in combat
        no_defeat_start = time.time()  # Track when we started waiting for defeat
        
        while (time.time() - start_time) < timeout_sec:
            time_without_defeat = time.time() - no_defeat_start
            if time_without_defeat > 60.0:  # 1 minute timeout for no defeats
                return False, None

            # Check combat status
            in_combat = _is_still_in_combat(act_log_watcher)
            if in_combat:
                combat_started = True  # We've detected combat activity
                if not combat_reported:
                    print("[DEBUG] In combat")
                    combat_reported = True
                last_combat_time = time.time()
                no_combat_start = None  # Reset the no-combat timer
            elif combat_started:  # Only check for combat exit if we were in combat first
                # Start tracking time since last combat if we haven't already
                if no_combat_start is None:
                    no_combat_start = time.time()
                    combat_reported = False  # Reset so we can report next combat entry
                elif time.time() - no_combat_start > 2.0:  # If no combat for 2 seconds
                    return False, None
                
            # Check if any valid target was defeated since our timestamp
            for target_lower in valid_targets_lower:
                # Use a shorter timeout (2s) for each individual check since we're in a loop
                result, defeated_name = act_log_watcher.check_combat_completion(target_lower, timeout=2.0, since_timestamp=current_ts)
                if result:
                    return True, defeated_name

            # Only sleep briefly between checks
            time.sleep(0.1)
            
        return False, None
    except Exception as e:
        print(f"[!] Error monitoring ACT logs: {e}")
        return False, None

# --- OCR Debugging chat helpers ---
def _norm_token_generic(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("â€™", "'").replace("â€˜", "'")
    s = re.sub(r"(?:^[^\w]+)|(?:[^\w]+$)", "", s)  # trim punctuation at ends
    return s

def _group_ocr_lines(results, region=None):
    """Group OCR tokens into visual lines; include raw/norm text, boxes, centers, and token lists."""
    lines = {}
    for i, txt in enumerate(results["text"]):
        if not txt or not txt.strip():
            continue
        key = (results["block_num"][i], results["par_num"][i], results["line_num"][i])
        lines.setdefault(key, {"texts": [], "boxes": []})
        lines[key]["texts"].append(txt)
        x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]
        if region:
            x += region[0]; y += region[1]
        lines[key]["boxes"].append((x, y, w, h))

    out = []
    for info in lines.values():
        tokens = list(info["texts"])
        norm_tokens = [_norm_token_generic(t) for t in tokens]
        norm_tokens = [t for t in norm_tokens if t]
        raw = " ".join(tokens).strip()
        norm = " ".join(norm_tokens)
        xs = [b[0] for b in info["boxes"]]; ys = [b[1] for b in info["boxes"]]
        xe = [b[0] + b[2] for b in info["boxes"]]; ye = [b[1] + b[3] for b in info["boxes"]]
        x0, y0, x1, y1 = min(xs), min(ys), max(xe), max(ye)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        entry = {
            "raw": raw,
            "norm": norm,
            "bbox": (x0, y0, x1, y1),
            "center": (cx, cy),
            "tokens": tokens,
            "boxes": info["boxes"],
            "norm_tokens": norm_tokens
        }
        out.append(entry)
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
            print(f"  â€¦ {len(lines) - OCR_DEBUG_MAX_LINES} more line(s) omitted")
    except Exception as e:
        print(f"[dbg][ocr] dump failed: {e}")

# === Main Logic ===
single_run_mode = False

# Global ACT log watcher instance
act_log_watcher = None

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

# Initialize ACT log watcher
try:
    act_log_watcher = ACTLogWatcher()
    print("[+] ACT log watcher initialized successfully")
    print(f"[+] ACT Logs directory: {act_log_watcher._log_dir}")
    print(f"[+] Combat events will be written to: {act_log_watcher._combat_log}")
    
    # Register a debug callback to show combat events in main console
    def debug_combat_callback(monster_name: str):
        print(f"[DEBUG][Combat] Defeated: {monster_name}")
    
    act_log_watcher.register_combat_callback(debug_combat_callback)
    
    # Start the watcher
    act_log_watcher.start()
    print("[+] Combat logging active")
    
except Exception as e:
    print(f"[!] Failed to initialize ACT log watcher: {e}")
    act_log_watcher = None

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
print(f"[+] Selected job: {selected_class} â†’ {job_file_resolved}")

# --- ALWAYS load the CLASS file (manifest) first ---
manifest_raw = load_json(job_file)  # e.g. "pld.json" â†’ list of {"quest": "...", "source": "shared"|"class"}
print(f"[+] Loaded data from {job_file}")

# Normalize manifest to a list
manifest_list = manifest_raw["quests"] if isinstance(manifest_raw, dict) and "quests" in manifest_raw else (
    manifest_raw if isinstance(manifest_raw, list) else []
)

# --- Read progress.json via the registry and prepare helpers ---
progress_path = resolve_file_path(PROGRESS_STATE_FILE) or _to_abs(PROGRESS_STATE_FILE)
try:
    progress = load_json(PROGRESS_STATE_FILE)
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
        # if itâ€™s not completed in either shared or class space, resume it
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

def _resume_from_saved_step() -> bool:
    """
    Reload current_file and jump to the next defined step after saved_step.
    If the quest had been marked completed but a new step now exists, set completed=False.
    If no next step exists, auto-advance to the next incomplete quest from the manifest.
    """
    global current_file, current_step, data, progress
    print("[*] RESUME requested.")

    # Reload registries & current quest file to see newly-added steps
    reload_jsons()
    data = load_json(current_file)

    # All defined step keys in the current file
    try:
        keys = sorted(int(k) for k in (data.keys() if isinstance(data, dict) else []))
    except Exception:
        keys = []

    # Baseline = saved step (defaults to 1 if not present)
    saved_step = load_progress(current_file)
    try:
        baseline = int(saved_step) if isinstance(saved_step, int) else 0
    except Exception:
        baseline = 0

    # Next defined step strictly after baseline
    next_defined = next((k for k in keys if k > baseline), None)

    if next_defined is None:
        if baseline > 0:
            _mark_quest_completed(current_file, baseline)
        return _auto_advance_to_next_incomplete()
    root = _progress_load_root()
    qkey = _quest_key_for_path(current_file)
    src  = _quest_source_for(qkey, manifest_list)
    if src == "shared":
        root.setdefault("shared", {}).setdefault(qkey, {})["completed"] = False
        root["shared"][qkey]["step"] = int(baseline)
    else:
        root.setdefault("jobs", {}).setdefault(selected_class, {}).setdefault(qkey, {})["completed"] = False
        root["jobs"][selected_class][qkey]["step"] = int(baseline)

    # Update current pointer and persist
    current_step = int(next_defined)
    root.setdefault("current", {})
    root["current"].update({"job": selected_class, "quest": qkey, "step": current_step})
    _progress_save_root(root)
    progress = root

    print(f"[*] RESUME → {os.path.basename(current_file)} step {current_step}")
    return True

def _auto_advance_to_next_incomplete() -> bool:
    """
    Scan manifest_list in order; pick the first quest not completed in progress.json.
    If found, load it and set current_step from saved progress (else 1).
    Returns True if we switched file/step; False otherwise.
    """
    global current_file, current_step, data, progress

    root = _progress_load_root()

    def _is_incomplete(qkey: str, source: str) -> bool:
        s = (source or "class").lower()
        if s == "shared":
            return not bool(root.get("shared", {}).get(qkey, {}).get("completed"))
        return not bool(root.get("jobs", {}).get(selected_class, {}).get(qkey, {}).get("completed"))

    for ent in (manifest_list or []):
        if not isinstance(ent, dict):
            continue
        qk = str(ent.get("quest") or "").strip().lower()
        if not qk:
            continue
        src = str(ent.get("source") or "class").lower()
        if not _is_incomplete(qk, src):
            continue

        qp = resolve_file_path(f"{qk}.json") or resolve_file_path(qk)
        if not qp:
            continue

        # Switch to that quest
        current_file = qp
        data = load_json(current_file)
        current_step = 1  # always start at 1 when auto-advancing after a finished quest

        _set_current_pointer_state(current_file, 1)
        print(f"[+] Next quest: {os.path.basename(current_file)} (step 1)")
        return True

    return False

def save_progress_quiet(file_path: str, step: int) -> None:
    _write_progress_entry(file_path, step, completed=False, update_current=False)

print(f"[+] Job file: {current_file}")

# Load the QUEST macro file (not the class manifest) and normalize step
data = load_json(current_file)
current_step = 1 if not isinstance(current_step, int) or current_step <= 0 else current_step

# Ensure the loaded step is valid (loop until we have a runnable step/file)
while True:
    step_keys = sorted(int(k) for k in data.keys())
    if not step_keys:
        print(f"[!] No steps defined in {os.path.basename(current_file)}")
        sys.exit(0)

    if str(current_step) not in data:
        next_keys = [k for k in step_keys if k >= current_step]
        if not next_keys:
            # No more steps in this file â†’ mark completed, return via stack if any,
            # else auto-advance to next incomplete quest; only open console if none exist.
            if _finalize_and_prompt(current_step - 1):
                continue

        # else: step exists later in file â†’ advance to it
        current_step = min(next_keys)

    # If we reached here, current_step is valid for current_file
    break

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
            # No more steps in this file → mark completed, return via stack if any,
            # else auto-advance to next incomplete quest; only open console if none exist.
            if _finalize_and_prompt(current_step - 1):
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
                print(f"  [*] vnav leg {i}/{total} â†’ {coord_str} (tolerance={tolerance})")
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
                    print(f"  [*] From {map_name}: vnav leg {i}/{total} â†’ {coord_str} (tolerance={tolerance})")
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

                print(f"  [*] From {map_name}: destination check â†’ {coord_str} (tolerance={tolerance})")
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
            # Normalize formats: 253 / "253_way..." / ["253_way...", 1] / [253, 1]
            raw = step["goto"]
            target_file, target_step = None, 1

            if isinstance(raw, list) and len(raw) >= 1:
                target = raw[0]
                target_step = int(raw[1]) if len(raw) >= 2 and str(raw[1]).isdigit() else 1
            else:
                target = raw

            # Resolve the target path (accept id, bare name, or full/relative path)
            if isinstance(target, (int, str)):
                tgt_str = str(target).strip()
                # try id/name with/without .json via registry
                target_file = resolve_file_path(f"{tgt_str}.json") or resolve_file_path(tgt_str) or tgt_str
            else:
                target_file = str(target)

            if not _exists_any(target_file):
                print(f"[!] GOTO target not found: {raw}")
                continue

            # IMPORTANT: save the SOURCE quest at current_step+1 to avoid re-running this step
            # and DO NOT update progress['current'] to the target.
            save_progress(current_file, int(current_step) + 1)

            # Push return frame so finishing the target can come back here at the next step
            GOTO_STACK.append((current_file, int(current_step) + 1))

            print(f"[*] GOTO → {target_file}, step {target_step}")

            # Switch to target (no current-pointer update here)
            current_file = target_file
            data = load_json(current_file)
            current_step = int(target_step) if int(target_step) > 0 else 1

            # Reset any transient detection state if you keep it
            last_ocr_text = None
            last_ocr_region = None
            last_img_template = None

            # Re-enter loop on the new file/step
            continue
    
    save_progress(current_file, current_step)

    # If a goto switched files, don't advance here.
    if goto_triggered:
        goto_triggered = False
    else:
        try:
            # Find the next numeric step key present in this file
            _keys_now = sorted(int(k) for k in data.keys())
            _next_defined = next((k for k in _keys_now if k > int(current_step)), None)
            if _next_defined is not None:
                current_step = int(_next_defined)
            else:
                # No further defined steps; let the top-of-loop "no more steps" handler deal with it
                current_step = int(current_step) + 1
        except Exception:
            # Safe fallback
            current_step = int(current_step) + 1

    # Go straight to next iteration so the loop re-evaluates with the new step
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
        if _finalize_and_prompt(current_step):
            continue

    # Then advance to the next step for execution (but do not save it yet)
    current_step += 1
