import os
import json
from typing import Dict, List, Any, Optional, Union

class AssetManager:
    """Centralized asset management for files and images.
    Provides consistent path resolution, JSON handling, and registry-based asset lookup."""
    def __init__(self, base_dir: str):
        # Initialize manager with base directory as root for path resolution
        self.base_dir = base_dir
        self.file_index: Dict[str, str] = {}  # Registry for file assets
        self.image_index: Dict[str, str] = {}  # Registry for image assets
        self.progress_state_file = "progress.json"
        self._load_asset_indices()

    @property
    def progress_file_path(self) -> str:
        """Get absolute path to the progress state file."""
        return self._to_abs(self.progress_state_file)

    def _to_abs(self, p: str) -> str:
        """Convert a path to absolute, anchoring at base_dir if relative."""
        return self._normalize_slashes(p if os.path.isabs(p) else os.path.join(self.base_dir, p))

    def _normalize_slashes(self, path: str) -> str:
        """Normalize path separators to forward slashes."""
        return path.replace('\\', '/')

    def _exists_any(self, p: str) -> bool:
        """Check if a path exists in either form (absolute or relative to base_dir)."""
        return os.path.exists(p) or os.path.exists(self._to_abs(p))

    def _assets_fullpath(self, relpath: str) -> str:
        """Ensure path is under assets directory and convert to absolute."""
        r = str(relpath).lstrip("/\\")
        low = r.lower()
        if low.startswith("assets/") or low.startswith(f"assets{os.sep}"):
            return self._to_abs(r)
        return self._normalize_slashes(self._to_abs(os.path.join("assets", r)))

    def _build_asset_index(self, spec: Union[Dict[str, str], List[str]]) -> Dict[str, str]:
        """Map filenames to normalized paths from dict/list."""
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

    def _load_asset_indices(self):
        def try_load_json(candidates: tuple[str, ...]) -> Optional[Any]:
            """Try to load JSON from first existing candidate path."""
            for cand in candidates:
                p = self._to_abs(cand)
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
            return None

        files_json = try_load_json(("assets/files.json", "assets/FILES.json"))
        images_json = try_load_json(("assets/images.json", "assets/IMAGES.json"))
        
        self.file_index = self._build_asset_index(files_json) if files_json else {}
        self.image_index = self._build_asset_index(images_json) if images_json else {}

    def load_json(self, filename: str) -> Any:
        """Load and parse JSON file from registry or direct path, raises FileNotFoundError if not found."""
        tried = []
        low = str(filename).lower().replace("\\", "/")

        def try_load(filepath: str) -> Optional[Any]:
            """Helper to attempt JSON load from a path."""
            tried.append(filepath)
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None

        # Check direct registry files first
        if low.endswith(("assets/files.json", "/files.json", "assets/images.json", "/images.json")):
            result = try_load(self._to_abs(filename))
            if result is not None:
                return result

        # Try registry lookup
        key = os.path.basename(str(filename)).lower()
        rel = self.file_index.get(key) or self.file_index.get(f"{key}.json")
        if rel:
            result = try_load(self._assets_fullpath(rel))
            if result is not None:
                return result

        raise FileNotFoundError(f"No such JSON via registry: {filename} (tried: {tried})")

    def resolve_file(self, name_or_path: str) -> Optional[str]:
        """Find absolute path for a file using registry lookup or None if not found."""
        p = str(name_or_path)
        key = os.path.basename(p).lower()
        rel = self.file_index.get(key) or self.file_index.get(f"{key}.json")
        if rel:
            return os.path.normpath(self._assets_fullpath(rel))
        return None

    def resolve_image(self, name_or_path: str) -> Optional[str]:
        """Find absolute path for an image using direct path, assets path, or registry lookup."""
        p = str(name_or_path)
        if self._exists_any(p):
            return self._to_abs(p)
        if p.lower().startswith("assets" + os.sep) or p.lower().startswith("assets/"):
            return self._to_abs(p)
        key = os.path.basename(p).lower()
        rel = self.image_index.get(key)
        if rel:
            return self._assets_fullpath(rel)
        return None

    def resolve_quest_by_id(self, quest_id: str) -> Optional[str]:
        """Find quest file path by ID using direct lookup or quests.json registry."""
        path = self.resolve_file(quest_id) or self.resolve_file(f"{quest_id}.json")  # Try ID directly
        if path:
            return path

        quest_index = self.resolve_file("quests.json")  # Try quest index lookup
        if quest_index and os.path.exists(quest_index):
            try:
                with open(quest_index, "r", encoding="utf-8") as f:
                    quests = json.load(f)
                    if quest_id in quests:
                        return self.resolve_file(quests[quest_id])
            except Exception:
                pass
        return None
        
    def load_progress_root(self) -> Dict[str, Any]:
        """Load progress state from file or return empty dict if file doesn't exist."""
        try:
            return self.load_json(self.progress_state_file) or {}
        except Exception:
            return {}
            
    def save_progress_root(self, root: Dict[str, Any]) -> None:
        """Write progress state to configured JSON file."""
        p = self.resolve_file(self.progress_state_file) or self._to_abs(self.progress_state_file)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(root, f, indent=2)