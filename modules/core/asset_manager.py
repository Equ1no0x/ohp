"""
Asset Manager Module for FF14 automation system.
Handles all asset loading, path resolution, and resource management.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union

class AssetManager:
    """
    Handles asset loading, path resolution, and resource management.
    All file operations should go through this class to ensure consistent handling.
    """

    def __init__(self, base_dir: str):
        """
        Initialize the AssetManager with a base directory.
        
        Args:
            base_dir (str): Base directory for all relative path resolutions
        """
        self.base_dir = base_dir
        # Registries for file and image assets
        self.file_index: Dict[str, str] = {}
        self.image_index: Dict[str, str] = {}
        # Default paths
        self.progress_state_file = "progress.json"
        # Load asset indices on initialization
        self._load_asset_indices()

    @property
    def progress_file_path(self) -> str:
        """Get the absolute path to progress.json"""
        return self._to_abs(self.progress_state_file)

    def _to_abs(self, p: str) -> str:
        """Convert a path to absolute, anchoring at base_dir if relative."""
        return self._normalize_slashes(
            p if os.path.isabs(p) else os.path.join(self.base_dir, p)
        )

    def _normalize_slashes(self, path: str) -> str:
        """Normalize path separators to system standard."""
        return path.replace('\\', '/')

    def _exists_any(self, p: str) -> bool:
        """Check if path exists either as-is or relative to base_dir."""
        return os.path.exists(p) or os.path.exists(self._to_abs(p))

    def _assets_fullpath(self, relpath: str) -> str:
        """Get full path to an asset, ensuring it's under the assets directory."""
        r = str(relpath).lstrip("/\\")
        low = r.lower()
        if low.startswith("assets/") or low.startswith(f"assets{os.sep}"):
            return self._to_abs(r)
        return self._normalize_slashes(self._to_abs(os.path.join("assets", r)))

    def _build_asset_index(self, spec: Union[Dict[str, str], List[str]]) -> Dict[str, str]:
        """
        Build an asset index from a specification.
        
        Args:
            spec: Dictionary mapping names to paths, or list of paths
            
        Returns:
            Dict mapping lowercase names to normalized paths
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

    def _load_asset_indices(self):
        """Load asset registry files (files.json and images.json) directly."""
        files_json = None
        images_json = None

        # Try to load files.json
        for cand in ("assets/files.json", "assets/FILES.json"):
            p = self._to_abs(cand)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    files_json = json.load(f)
                break

        # Try to load images.json
        for cand in ("assets/images.json", "assets/IMAGES.json"):
            p = self._to_abs(cand)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    images_json = json.load(f)
                break

        self.file_index = self._build_asset_index(files_json) if files_json else {}
        self.image_index = self._build_asset_index(images_json) if images_json else {}

    def load_json(self, filename: str) -> Any:
        """
        Load a JSON file, using the registry to resolve paths.
        
        Args:
            filename: Name or path of the JSON file
            
        Returns:
            Parsed JSON content
        
        Raises:
            FileNotFoundError: If file cannot be found via registry
        """
        tried = []

        # Allow direct read ONLY for registry files themselves
        low = str(filename).lower().replace("\\", "/")
        if low.endswith("assets/files.json") or low.endswith("/files.json"):
            p = self._to_abs(filename)
            tried.append(p)
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        if low.endswith("assets/images.json") or low.endswith("/images.json"):
            p = self._to_abs(filename)
            tried.append(p)
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        # Resolve via registry
        key = os.path.basename(str(filename)).lower()
        rel = self.file_index.get(key) or self.file_index.get(f"{key}.json")

        if rel:
            rp = self._assets_fullpath(rel)
            tried.append(rp)
            with open(rp, "r", encoding="utf-8") as f:
                return json.load(f)

        raise FileNotFoundError(f"No such JSON via registry: {filename} (tried: {tried})")

    def resolve_file(self, name_or_path: str) -> Optional[str]:
        """
        Resolve a file path using the registry.
        
        Args:
            name_or_path: File name or path to resolve
            
        Returns:
            Absolute path if found, None otherwise
        """
        p = str(name_or_path)
        key = os.path.basename(p).lower()
        rel = self.file_index.get(key) or self.file_index.get(f"{key}.json")
        if rel:
            return os.path.normpath(self._assets_fullpath(rel))
        return None

    def resolve_image(self, name_or_path: str) -> Optional[str]:
        """
        Resolve an image path using the registry.
        
        Args:
            name_or_path: Image name or path to resolve
            
        Returns:
            Absolute path if found, None otherwise
        """
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
        """
        Try to resolve a quest file by its ID.
        
        Args:
            quest_id: The ID of the quest to find
            
        Returns:
            Absolute path to quest file if found, None otherwise
        """
        # First try the ID directly (with and without .json)
        path = self.resolve_file(quest_id) or self.resolve_file(f"{quest_id}.json")
        if path:
            return path

        # Then try looking up the quest ID in the quest index if it exists
        quest_index = self.resolve_file("quests.json")
        if quest_index and os.path.exists(quest_index):
            try:
                with open(quest_index, "r", encoding="utf-8") as f:
                    quests = json.load(f)
                    if quest_id in quests:
                        return self.resolve_file(quests[quest_id])
            except Exception:
                pass
        return None