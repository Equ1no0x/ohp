import time
from typing import Optional

from .system_actions import SystemActions
from .input_manager import InputManager

class GameActions:
    """Handles high-level game actions like chat commands and combat toggles."""
    
    def __init__(self):
        """Initialize GameActions."""
        self.system_actions = SystemActions()
    
    def send_chat_command(self, command: str, keymap: dict) -> None:
        """
        Send a chat command to the game using the system's copy/paste functionality.
        
        Args:
            command: The chat command to send
            keymap: Dictionary mapping of virtual key names to their hex codes
        """
        # Use system paste instead of direct keystroke simulation for reliability
        self.system_actions.copy_to_clipboard(command)
        time.sleep(0.05)
        
        # Open chat with Enter
        InputManager.press_virtual_key("VK_RETURN", keymap)
        time.sleep(0.05)
        InputManager.release_virtual_key("VK_RETURN", keymap)
        time.sleep(0.2)
        
        # Paste with Ctrl+V
        InputManager.press_virtual_key("VK_CONTROL", keymap)
        InputManager.press_virtual_key("V", keymap)
        time.sleep(0.05)
        InputManager.release_virtual_key("V", keymap)
        InputManager.release_virtual_key("VK_CONTROL", keymap)
        time.sleep(0.2)
        
        # Send with Enter
        InputManager.press_virtual_key("VK_RETURN", keymap)
        time.sleep(0.05)
        InputManager.release_virtual_key("VK_RETURN", keymap)
        time.sleep(0.1)
    
    def toggle_rotation(self, keymap: dict, enable: bool = True) -> None:
        """
        Toggle the combat rotation system.
        
        Args:
            keymap: Dictionary mapping of virtual key names to their hex codes
            enable: True to enable rotation, False to disable
        """
        command = "/rotation Manual" if enable else "/rotation off"
        self.send_chat_command(command, keymap)
        time.sleep(0.2)
    
    def toggle_combat_ai(self, keymap: dict, enable: bool = True) -> None:
        """
        Toggle the combat AI system.
        
        Args:
            keymap: Dictionary mapping of virtual key names to their hex codes
            enable: True to enable AI, False to disable
        """
        command = "/bmrai on" if enable else "/bmrai off"
        self.send_chat_command(command, keymap)
        time.sleep(0.2)

