import pytesseract
import re
import difflib
from PIL import ImageGrab
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
import json
import functools
from datetime import datetime, timedelta

# === OCR Configuration Constants ===
TESSERACT_TIMEOUT_SEC = 5
OCR_FUZZY_ENABLED = True
OCR_FUZZY_THRESHOLD = 0.90
OCR_DEBUG_DUMP = True
OCR_DEBUG_MAX_LINES = 1

# Common stopwords for fuzzy matching
STOPWORDS_COMMON = {
    'the', 'a', 'an', 'to', 'for', 'with', 'and', 'or', 'you', 'your', 
    'from', 'at', 'on', 'in', 'it', 'is', 'are', 'was', 'were', 'of',
    'this', 'that', 'these', 'those'
}

class OCRCache:
    """Cache for OCR results to avoid repeated processing of same regions."""
    def __init__(self, max_size: int = 100, ttl_seconds: int = 5):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        
    def _make_key(self, region: Optional[Tuple[int, int, int, int]], query: str) -> str:
        """Create a cache key from region and query."""
        return f"{region}:{query}"
        
    def get(self, region: Optional[Tuple[int, int, int, int]], query: str) -> Optional[Any]:
        """Get cached result if valid."""
        key = self._make_key(region, query)
        if key not in self.cache:
            return None
            
        result, timestamp = self.cache[key]
        if datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds):
            del self.cache[key]
            return None
            
        return result
        
    def set(self, region: Optional[Tuple[int, int, int, int]], query: str, result: Any):
        """Cache a result."""
        key = self._make_key(region, query)
        self.cache[key] = (result, datetime.now())
        
        # Maintain max size
        if len(self.cache) > self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

class OCRConfig:
    """Configuration management for OCR system."""
    def __init__(self):
        self.tesseract_timeout = TESSERACT_TIMEOUT_SEC
        self.fuzzy_enabled = OCR_FUZZY_ENABLED
        self.fuzzy_threshold = OCR_FUZZY_THRESHOLD
        self.debug_dump = OCR_DEBUG_DUMP
        self.debug_max_lines = OCR_DEBUG_MAX_LINES
        self.cache_enabled = True
        self.cache_ttl = 5  # seconds
        self.cache_size = 100
        self.max_retries = 3
        self.retry_delay = 0.2  # seconds
        
    def load_from_file(self, config_path: Union[str, Path]):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except Exception as e:
            print(f"[!] Error loading OCR config: {e}")
            
    def save_to_file(self, config_path: Union[str, Path]):
        """Save current configuration to JSON file."""
        try:
            config = {
                key: value for key, value in vars(self).items()
                if not key.startswith('_')
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"[!] Error saving OCR config: {e}")

class OCRSystem:
    """Core OCR system with caching, error handling, and UI element support."""
    
    def __init__(self):
        self.config = OCRConfig()
        self.cache = OCRCache(
            max_size=self.config.cache_size,
            ttl_seconds=self.config.cache_ttl
        )
        self.last_ocr_text = None
        self.last_ocr_region = None
        self.last_detection_type = None
        self.last_ocr_match_center = None
        
    def _iter_ocr_configs(self, langs: List[str], psms: List[int]):
        """Iterate through OCR configuration combinations."""
        for lang in langs:
            for psm in psms:
                yield lang, psm

    def _try_ocr_image(self, img, lang: str, psm: int, timeout: float) -> Tuple[Optional[Dict], Optional[Exception]]:
        """Attempt OCR with specific configuration."""
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

    def _ocr_has_tokens(self, payload: Dict) -> bool:
        """Check if OCR result contains any tokens."""
        return any((token or "").strip() for token in payload.get("text", []))

    def _empty_ocr_payload(self) -> Dict:
        """Create empty OCR result structure."""
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

    def _debug_ocr_success(self, lang: str, psm: int, payload: Dict):
        """Log OCR success details if debug enabled."""
        if not self.config.debug_dump:
            return
        non_empty = sum(1 for token in payload.get("text", []) if token and token.strip())
        # print(f"[dbg][ocr] used lang='{lang}' psm={psm} tokens={non_empty}")

    def _ocr_image_to_data(self, img, timeout_sec: Optional[float] = None) -> Tuple[Dict, Optional[str], Optional[int]]:
        """Process image through OCR with fallback configurations."""
        psms = [6, 7, 3, 11, 13]
        langs = ['eng+ffxiv', 'eng']
        timeout = timeout_sec if timeout_sec is not None else self.config.tesseract_timeout

        last_exc = None
        for lang, psm in self._iter_ocr_configs(langs, psms):
            payload, err = self._try_ocr_image(img, lang, psm, timeout)
            if err is not None:
                last_exc = err
                continue
            if not self._ocr_has_tokens(payload):
                continue
            self._debug_ocr_success(lang, psm, payload)
            return payload, lang, psm

        if last_exc:
            print(f"[dbg][ocr] OCR attempts failed; last error: {last_exc}")

        return self._empty_ocr_payload(), None, None

    def _normalize_target_text(self, target_text: str) -> Tuple[List[str], str]:
        """Normalize target text for matching."""
        words = [self._norm_token_generic(part) for part in target_text.strip().split()]
        words = [w for w in words if w]
        return words, " ".join(words)

    def _norm_token_generic(self, s: str) -> str:
        """Normalize a single token."""
        s = (s or "").lower()
        s = s.replace("â€™", "'").replace("â€˜", "'")
        s = re.sub(r"(?:^[^\w]+)|(?:[^\w]+$)", "", s)
        return s

    def _normalize_chat_text(self, s: str) -> str:
        """Normalize chat text for fuzzy matching."""
        s = (s or "").lower()
        s = s.translate({0x2019: ord("'"), 0x2018: ord("'"), 0x0060: ord("'")})
        s = s.rstrip('.')
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _important_tokens(self, text: str) -> set:
        """Extract important tokens for matching."""
        tokens = [tok for tok in text.split() if tok and tok not in STOPWORDS_COMMON]
        return set(tokens)

    def _group_ocr_lines(self, results: Dict, region: Optional[Tuple[int, int, int, int]] = None) -> List[Dict]:
        """Group OCR tokens into visual lines."""
        lines = {}
        for i, txt in enumerate(results["text"]):
            if not txt or not txt.strip():
                continue
            key = (results["block_num"][i], results["par_num"][i], results["line_num"][i])
            lines.setdefault(key, {"texts": [], "boxes": []})
            lines[key]["texts"].append(txt)
            x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]
            if region:
                x += region[0]
                y += region[1]
            lines[key]["boxes"].append((x, y, w, h))

        out = []
        for info in lines.values():
            tokens = list(info["texts"])
            norm_tokens = [self._norm_token_generic(t) for t in tokens]
            norm_tokens = [t for t in norm_tokens if t]
            raw = " ".join(tokens).strip()
            norm = " ".join(norm_tokens)
            xs = [b[0] for b in info["boxes"]]
            ys = [b[1] for b in info["boxes"]]
            xe = [b[0] + b[2] for b in info["boxes"]]
            ye = [b[1] + b[3] for b in info["boxes"]]
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

    def _dump_ocr_debug(self, results: Dict, region: Optional[Tuple[int, int, int, int]], target: Optional[str] = None):
        """Debug dump of OCR results."""
        if not self.config.debug_dump:
            return
        try:
            lines = self._group_ocr_lines(results, region)
            hdr = f"[dbg][ocr] lines={len(lines)}"
            if target:
                hdr += f" (target='{target}')"
            print(hdr)
            for i, ln in enumerate(lines[:self.config.debug_max_lines], 1):
                print(f"  {i:>2}. raw='{ln['raw']}' | norm='{ln['norm']}' | center={ln['center']} bbox={ln['bbox']}")
            if len(lines) > self.config.debug_max_lines:
                print(f"  ... {len(lines) - self.config.debug_max_lines} more line(s) omitted")
        except Exception as e:
            print(f"[dbg][ocr] dump failed: {e}")

    @functools.lru_cache(maxsize=1000)
    def _best_span_for_target(self, tokens_norm_str: str, target_norm: str, max_extra: int = 2) -> Optional[Tuple[int, int, float]]:
        """Find best matching span for target text (cached)."""
        tokens_norm = tokens_norm_str.split()
        if not tokens_norm or not target_norm:
            return None
        
        target_words = target_norm.split()
        if not target_words:
            return None
            
        # Try exact match first
        if target_norm in tokens_norm_str:
            for i in range(len(tokens_norm)):
                for j in range(i + 1, len(tokens_norm) + 1):
                    span = " ".join(tokens_norm[i:j])
                    if target_norm in span:
                        return (i, j - 1, 1.0)
                        
        # Fall back to fuzzy match
        best_ratio = 0.0
        best_span = None
        for i in range(len(tokens_norm)):
            for j in range(i + 1, min(i + len(target_words) + max_extra + 1, len(tokens_norm) + 1)):
                span = " ".join(tokens_norm[i:j])
                ratio = difflib.SequenceMatcher(None, span, target_norm).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_span = (i, j - 1, ratio)
                    
        return best_span

    def _locate_fuzzy_ocr_match(self, lines: List[Dict], target_norm_line: str, threshold: float) -> Optional[Tuple[float, Tuple[int, int], str]]:
        """Locate fuzzy OCR match in lines."""
        if not target_norm_line:
            return None
            
        best_ratio, best_center, best_text = 0.0, None, None
        for info in lines:
            norm_tokens = info.get('norm_tokens') or []
            if not norm_tokens:
                continue
                
            norm_tokens_str = " ".join(norm_tokens)
            span = self._best_span_for_target(norm_tokens_str, target_norm_line)
            if not span:
                continue
                
            start, end, ratio = span
            if ratio < threshold and target_norm_line not in " ".join(norm_tokens[start:end + 1]):
                continue
                
            if ratio >= best_ratio:
                xs = [info['boxes'][i][0] for i in range(start, end + 1)]
                ys = [info['boxes'][i][1] for i in range(start, end + 1)]
                xe = [info['boxes'][i][0] + info['boxes'][i][2] for i in range(start, end + 1)]
                ye = [info['boxes'][i][1] + info['boxes'][i][3] for i in range(start, end + 1)]
                x0, y0, x1, y1 = min(xs), min(ys), max(xe), max(ye)
                center = ((x0 + x1) // 2, (y0 + y1) // 2)
                matched_text = " ".join(norm_tokens[start:end + 1])
                best_ratio, best_center, best_text = ratio, center, matched_text
                
        if best_center is not None:
            return best_ratio, best_center, best_text
        return None

    def find_text(self, target_text: str, region: Optional[Tuple[int, int, int, int]] = None, retry_count: int = 0) -> bool:
        """Find text in screen region with retries."""
        self.last_ocr_text = target_text
        self.last_ocr_region = region
        
        # Check cache first
        if self.config.cache_enabled:
            cached = self.cache.get(region, target_text)
            if cached is not None:
                if isinstance(cached, tuple) and len(cached) == 2:
                    found, center = cached
                    if found:
                        self.last_ocr_match_center = center
                        self.last_detection_type = "ocr"
                        return True
                return False
        
        # Take screenshot and process
        screenshot = ImageGrab.grab(bbox=region)
        results, _lang, _psm = self._ocr_image_to_data(screenshot)
        
        target_words, target_norm_line = self._normalize_target_text(target_text)
        lines = self._group_ocr_lines(results, region)
        
        # Try exact match
        for info in lines:
            norm_tokens = info.get('norm_tokens') or []
            if len(norm_tokens) < len(target_words):
                continue
            for start in range(len(norm_tokens) - len(target_words) + 1):
                if norm_tokens[start:start + len(target_words)] == target_words:
                    center = self._span_center_from_boxes(info['boxes'], start, start + len(target_words) - 1)
                    self.last_ocr_match_center = center
                    self.last_detection_type = "ocr"
                    if self.config.cache_enabled:
                        self.cache.set(region, target_text, (True, center))
                    return True
        
        # Try fuzzy match if enabled
        if self.config.fuzzy_enabled:
            fuzzy_result = self._locate_fuzzy_ocr_match(lines, target_norm_line, self.config.fuzzy_threshold)
            if fuzzy_result:
                ratio, center, matched_text = fuzzy_result
                self.last_ocr_match_center = center
                self.last_detection_type = "ocr"
                print(f"[*] Fuzzy matched '{target_text}' ~ '{matched_text}' ({ratio*100:.1f}%) at {center}")
                if self.config.cache_enabled:
                    self.cache.set(region, target_text, (True, center))
                return True
        
        # Handle miss
        self._dump_ocr_debug(results, region, target_norm_line)
        if self.config.cache_enabled:
            self.cache.set(region, target_text, (False, None))
            
        # Retry logic
        if retry_count < self.config.max_retries:
            import time
            time.sleep(self.config.retry_delay)
            return self.find_text(target_text, region, retry_count + 1)
            
        print(f"[!] Text '{target_text}' not found on screen.")
        return False

    def _span_center_from_boxes(self, boxes: List[Tuple[int, int, int, int]], start: int, end: int) -> Tuple[int, int]:
        """Calculate center point from span of boxes."""
        xs = [boxes[i][0] for i in range(start, end + 1)]
        ys = [boxes[i][1] for i in range(start, end + 1)]
        xe = [boxes[i][0] + boxes[i][2] for i in range(start, end + 1)]
        ye = [boxes[i][1] + boxes[i][3] for i in range(start, end + 1)]
        x0, y0, x1, y1 = min(xs), min(ys), max(xe), max(ye)
        return ((x0 + x1) // 2, (y0 + y1) // 2)

    # Game UI Elements - Example methods (add more as needed)
    def read_target_name(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[str]:
        """Read target name from UI."""
        region = region or (750, 31, 1190, 52)  # Default target name region
        screenshot = ImageGrab.grab(bbox=region)
        results, _, _ = self._ocr_image_to_data(screenshot)
        lines = self._group_ocr_lines(results, region)
        return lines[0]["raw"] if lines else None