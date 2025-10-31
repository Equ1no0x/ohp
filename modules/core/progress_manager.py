"""
Progress Manager module for quest tracking and state management.

This module handles:
- Centralized progress tracking via progress.json
- Quest state management (completion, steps, goto stack)
- Quest flow control and navigation
- Progress persistence and resumption
"""

from typing import Optional, Tuple, Dict, Any, List
import os
import json

from .asset_manager import AssetManager

class ProgressManager:
    """Manages quest progress, state, and navigation."""
    
    def __init__(self, asset_manager: AssetManager):
        """Initialize ProgressManager with AssetManager instance."""
        self.asset_manager = asset_manager
        self.progress_state_file = "progress.json"
        self.goto_stack: List[Tuple[str, int]] = []  # [(return_file_fullpath, return_step_int), ...]
        
        # Current state tracking
        self.progress: Dict[str, Any] = {}
        self.current_file: Optional[str] = None
        self.current_step: int = 1
        self.current_quest_key: Optional[str] = None
        self.selected_class: Optional[str] = None
        self.manifest_list: List[Dict[str, Any]] = []

    def initialize(self, selected_class: str, manifest_list: List[Dict[str, Any]]) -> None:
        """Initialize with class and manifest data."""
        self.selected_class = selected_class
        self.manifest_list = manifest_list
        self._load_progress()

    def _load_progress(self) -> None:
        """Load progress state from file."""
        try:
            self.progress = self.asset_manager.load_progress_root()
        except Exception:
            self.progress = {}

    def _save_progress(self) -> None:
        """Save current progress state to file."""
        self.asset_manager.save_progress_root(self.progress)

    def _quest_key_for_path(self, file_path: str) -> str:
        """Convert file path to quest key."""
        return os.path.splitext(os.path.basename(file_path))[0].lower()

    def _quest_source_for(self, quest_key: str) -> str:
        """Determine if quest is shared or class-specific. Returns 'shared' or 'class'."""
        for entry in self.manifest_list:
            if isinstance(entry, dict) and str(entry.get("quest", "")).strip().lower() == quest_key:
                return str(entry.get("source", "class")).lower()
        return "class"

    def is_quest_completed(self, quest_key: str, source: Optional[str] = None) -> bool:
        """Check if a quest is marked as completed."""
        if not source:
            source = self._quest_source_for(quest_key)
            
        if source == "shared":
            return bool(self.progress.get("shared", {}).get(quest_key, {}).get("completed"))
        return bool(self.progress.get("jobs", {}).get(self.selected_class, {}).get(quest_key, {}).get("completed"))

    def get_saved_step(self, quest_key: str, source: Optional[str] = None) -> int:
        """Get the saved step number for a quest."""
        if not source:
            source = self._quest_source_for(quest_key)
            
        if source == "shared":
            val = self.progress.get("shared", {}).get(quest_key, {}).get("step")
        else:
            val = self.progress.get("jobs", {}).get(self.selected_class, {}).get(quest_key, {}).get("step")
            
        try:
            step = int(val) if val is not None else 1
            return max(1, step)  # Ensure step is at least 1
        except Exception:
            return 1

    def load_quest_progress(self, file_path: str, skip_goto: bool = False) -> int:
        """Load saved progress for a quest file.
        
        Args:
            file_path: Path to quest file
            skip_goto: If True, ignore steps reached via goto
            
        Returns:
            Current step number for the quest
        """
        quest_key = self._quest_key_for_path(file_path)
        source = self._quest_source_for(quest_key)
        
        if source == "shared":
            entry = self.progress.get("shared", {}).get(quest_key, {})
        else:
            entry = self.progress.get("jobs", {}).get(self.selected_class, {}).get(quest_key, {})
        entry = entry or {}
        
        try:
            base_step = int(entry.get("step", 1))
            base_step = max(1, base_step)
            
            if not skip_goto and "goto_steps" in entry:
                goto_steps = entry["goto_steps"]
                if goto_steps and any(s > base_step for s in goto_steps):
                    return max(s for s in goto_steps if s > base_step)
                    
            return base_step
        except Exception:
            return 1

    def save_quest_progress(self, file_path: str, step: int, *, completed: bool = False, 
                          update_current: bool = True, via_goto: bool = False) -> None:
        """Save progress for a quest file.
        
        Args:
            file_path: Path to quest file
            step: Current step number
            completed: Whether quest is completed
            update_current: Whether to update current pointer
            via_goto: Whether step was reached via goto command
        """
        try:
            step_value = max(1, int(step))
        except Exception:
            step_value = 1
            
        quest_key = self._quest_key_for_path(file_path)
        source = self._quest_source_for(quest_key)
        
        if source == "shared":
            entry = self.progress.setdefault("shared", {}).setdefault(quest_key, {})
        else:
            entry = self.progress.setdefault("jobs", {}).setdefault(self.selected_class, {}).setdefault(quest_key, {})
        
        if via_goto:
            entry.setdefault("goto_steps", []).append(step_value)
        else:
            entry["completed"] = bool(completed)
            entry["step"] = step_value
            if "goto_steps" in entry:
                entry["goto_steps"] = [s for s in entry["goto_steps"] if s > step_value]
        
        if update_current:
            self.progress.setdefault("current", {}).update({
                "job": self.selected_class,
                "quest": quest_key,
                "step": step_value
            })
            
        self._save_progress()

    def mark_quest_completed(self, file_path: str, last_step: Optional[int] = None) -> None:
        """Mark a quest as completed."""
        try:
            if last_step is None:
                last_step = self.load_quest_progress(file_path)
            step_value = max(1, int(last_step))
        except Exception:
            step_value = 1
            
        self.save_quest_progress(file_path, step_value, completed=True, update_current=False)

    def push_goto_stack(self, return_file: str, return_step: int) -> None:
        """Push a return point onto the goto stack."""
        self.goto_stack.append((return_file, return_step))

    def pop_goto_stack(self) -> Optional[Tuple[str, int]]:
        """Pop and return the top return point from the goto stack."""
        return self.goto_stack.pop() if self.goto_stack else None

    def return_to_goto_caller(self) -> Tuple[bool, Optional[str], Optional[int]]:
        """Return to the previous quest after a goto.
        
        Returns:
            Tuple of (success, file_path, step_number)
        """
        return_point = self.pop_goto_stack()
        if not return_point:
            return False, None, None
            
        ret_file, ret_step = return_point
        return True, ret_file, ret_step

    def find_next_incomplete_quest(self) -> Optional[Tuple[str, int]]:
        """Find the next incomplete quest from the manifest.
        
        Returns:
            Tuple of (file_path, initial_step) or None if no incomplete quests
        """
        for entry in self.manifest_list:
            if not isinstance(entry, dict):
                continue
                
            quest_key = str(entry.get("quest", "")).strip().lower()
            if not quest_key:
                continue
                
            source = str(entry.get("source", "class")).lower()
            if self.is_quest_completed(quest_key, source):
                continue
                
            quest_path = (self.asset_manager.resolve_file(f"{quest_key}.json") or 
                         self.asset_manager.resolve_file(quest_key))
            if not quest_path:
                continue
                
            initial_step = self.get_saved_step(quest_key, source)
            return quest_path, initial_step
            
        return None

    def set_current_quest(self, file_path: str, step: int) -> None:
        """Set the current quest and step."""
        self.current_file = file_path
        self.current_step = max(1, int(step))
        self.current_quest_key = self._quest_key_for_path(file_path)
        
        # Update progress tracking
        self.progress.setdefault("current", {}).update({
            "job": self.selected_class,
            "quest": self.current_quest_key,
            "step": self.current_step
        })
        self._save_progress()

    def resume_from_current(self) -> Tuple[Optional[str], Optional[int]]:
        """Resume from the current quest in progress.json.
        
        Returns:
            Tuple of (file_path, step) or (None, None) if no current quest
        """
        current = self.progress.get("current", {})
        if not current or current.get("job") != self.selected_class:
            return None, None
            
        quest_key = str(current.get("quest", "")).strip().lower()
        if not quest_key:
            return None, None
            
        quest_path = (self.asset_manager.resolve_file(f"{quest_key}.json") or 
                     self.asset_manager.resolve_file(quest_key))
        if not quest_path:
            return None, None
            
        if (self.is_quest_completed(quest_key, "shared") or 
            self.is_quest_completed(quest_key, "class")):
            return None, None
            
        step = max(1, int(current.get("step", 1)))
        return quest_path, step

    def get_quest_data(self) -> Tuple[Optional[str], int, Optional[str]]:
        """Get current quest state.
        
        Returns:
            Tuple of (current_file, current_step, current_quest_key)
        """
        return self.current_file, self.current_step, self.current_quest_key

    def set_quest_data(self, file: Optional[str], step: int, key: Optional[str]) -> None:
        """Set current quest state."""
        self.current_file = file
        self.current_step = step
        self.current_quest_key = key
