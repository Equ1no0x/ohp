"""
Manages execution of macro command sequences.

A macro sequence is an array of commands like:
["press", "VK_TAB", "rand", 56, 126, "release", "VK_TAB"]

Available commands:
- press <key>: Press a key
- release <key>: Release a key 
- wait <ms>: Wait for specified milliseconds
- rand <low> <high>: Wait for random milliseconds between low and high

Keys can be:
- Virtual key codes from keymaps.json (e.g. "VK_TAB", "VK_RETURN")  
- Letters (e.g. "A", "B", "C")
- Mouse buttons (e.g. "VK_LBUTTON", "VK_RBUTTON")
"""

from typing import Any, Optional
import random
import time

from .input_manager import InputManager
from .system_actions import SystemActions

# Initialize system actions for timing
system_actions = SystemActions()

# Mouse button mapping with flags for press/release - copied from v83
MOUSE_BUTTONS = {
    "VK_LBUTTON": ("left", 0x0002, 0x0004),
    "VK_RBUTTON": ("right", 0x0008, 0x0010),
    "VK_MBUTTON": ("middle", 0x0020, 0x0040),
    "VK_XBUTTON1": ("x1", 0x0080, 0x0100),
    "VK_XBUTTON2": ("x2", 0x0080, 0x0100)
}

class MacroManager:
    """Handles execution of macro command sequences."""
    
    @staticmethod
    def _macro_arg(macro_array: list, index: int, offset: int, cmd: str) -> Optional[Any]:
        """Get an argument from the macro array at index+offset."""
        try:
            return macro_array[index + offset]
        except IndexError:
            print(f"[!] Macro '{cmd}' expects more arguments (index {index})")
            return None

    @staticmethod
    def is_macro_step(step: dict) -> bool:
        """Check if a step is a macro step."""
        return isinstance(step, dict) and any(k in step for k in ("press", "release", "wait", "rand", "paste"))

    @staticmethod
    def _handle_mouse_button(action: str, button_id: str) -> None:
        """Handle mouse button press/release actions using v83 logic"""
        btn, _, _ = MOUSE_BUTTONS[button_id]
        print(f"[*] {'Pressing' if action == 'press' else 'Releasing'} {btn} mouse button")
        InputManager._handle_mouse_button(action, button_id)

    @staticmethod
    def _handle_press_release(action: str, macro_array: list, index: int, keymap: dict) -> int:
        """Common handler for press/release commands, matches v83 behavior"""
        target = MacroManager._macro_arg(macro_array, index, 1, action)
        if target is None:
            return len(macro_array)
            
        if target in MOUSE_BUTTONS:
            MacroManager._handle_mouse_button(action, target)
        else:
            if action == 'press':
                InputManager.press_virtual_key(target, keymap)
            else:
                InputManager.release_virtual_key(target, keymap)
        return index + 2

    @staticmethod
    def _handle_press(macro_array: list, index: int, keymap: dict) -> int:
        """Handle 'press' command in a macro sequence."""
        return MacroManager._handle_press_release('press', macro_array, index, keymap)

    @staticmethod
    def _handle_release(macro_array: list, index: int, keymap: dict) -> int:
        """Handle 'release' command in a macro sequence."""
        return MacroManager._handle_press_release('release', macro_array, index, keymap)

    @staticmethod
    def _handle_wait(macro_array: list, index: int, keymap: dict) -> int:
        """Handle 'wait' command in a macro sequence."""
        duration = MacroManager._macro_arg(macro_array, index, 1, 'wait')
        if duration is None:
            return len(macro_array)
            
        try:
            ms = int(duration)
            print(f"[*] Wait: {ms}ms")
            time.sleep(ms / 1000.0)
        except (TypeError, ValueError):
            print(f"[!] Invalid wait value: {duration}")
            return len(macro_array)
            
        return index + 2

    @staticmethod
    def _handle_rand(macro_array: list, index: int, keymap: dict) -> int:
        """Handle 'rand' command in a macro sequence."""
        low = MacroManager._macro_arg(macro_array, index, 1, 'rand')
        high = MacroManager._macro_arg(macro_array, index, 2, 'rand')
        
        if low is None or high is None:
            return len(macro_array)
            
        try:
            low_i = int(low)
            high_i = int(high)
            if low_i > high_i:
                low_i, high_i = high_i, low_i
            delay = random.randint(low_i, high_i)
            print(f"[*] Random wait: {delay}ms")
            time.sleep(delay / 1000.0)
        except (TypeError, ValueError):
            print(f"[!] Invalid rand bounds: {low}, {high}")
            return index + 3
            
        return index + 3

    # Map of command names to their handler functions
    _HANDLERS = {
        'press': _handle_press,
        'release': _handle_release,
        'wait': _handle_wait,
        'rand': _handle_rand,
    }

    @classmethod
    def execute_sequence(cls, macro_array: list, keymap: dict, command_text: Optional[str] = None) -> None:
        """Execute a macro command sequence, matching v83 behavior exactly"""
        index = 0
        length = len(macro_array)
        
        while index < length:
            cmd = macro_array[index]
            handler = cls._HANDLERS.get(cmd)
            
            if handler is None:
                print(f"[!] Unknown macro command: {cmd}")
                index += 1
                continue
                
            next_index = handler(macro_array, index, keymap)
            # v83 logic: only use next_index if it's greater than current index
            if next_index <= index:
                index += 1
            else:
                index = next_index