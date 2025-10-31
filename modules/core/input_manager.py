import ctypes
from ctypes import wintypes
import time

# Initialize user32.dll
user32 = ctypes.WinDLL('user32', use_last_error=True)

# Constants
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
MAPVK_VK_TO_VSC = 0

# Mouse button mapping with flags for press/release
MOUSE_BUTTONS = {
    "VK_LBUTTON": ("left", 0x0002, 0x0004),
    "VK_RBUTTON": ("right", 0x0008, 0x0010),
    "VK_MBUTTON": ("middle", 0x0020, 0x0040),
    "VK_XBUTTON1": ("x1", 0x0080, 0x0100),
    "VK_XBUTTON2": ("x2", 0x0080, 0x0100)
}

# Set up wintypes for input structures
wintypes.ULONG_PTR = wintypes.WPARAM

class MOUSEINPUT(ctypes.Structure):
    """Structure for mouse input events"""
    _fields_ = (("dx",          wintypes.LONG),
                ("dy",          wintypes.LONG),
                ("mouseData",   wintypes.DWORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

class KEYBDINPUT(ctypes.Structure):
    """Structure for keyboard input events"""
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
    """Structure for hardware input events"""
    _fields_ = (("uMsg",    wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))

class INPUT(ctypes.Structure):
    """Windows INPUT structure for SendInput"""
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                   ("mi", MOUSEINPUT),
                   ("hi", HARDWAREINPUT))
    _anonymous_ = ("_input",)
    _fields_ = (("type",   wintypes.DWORD),
               ("_input", _INPUT))

class InputManager:
    """
    Manages keyboard and mouse input operations, including both low-level API calls
    and high-level game-specific input sequences.
    """
    # Track last detected positions for game-specific operations
    last_detection_type = None
    last_ocr_match_center = None
    last_img_match_center = None

    # === Low-level input operations ===
    @staticmethod
    def move_mouse(x: int, y: int) -> None:
        """Move the mouse cursor to absolute screen coordinates"""
        user32.SetCursorPos(x, y)

    @staticmethod
    def press_key(hex_key_code: int) -> None:
        """Press a key given its hex key code"""
        x = INPUT(type=INPUT_KEYBOARD,
                 ki=KEYBDINPUT(wVk=hex_key_code))
        user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

    @staticmethod
    def release_key(hex_key_code: int) -> None:
        """Release a key given its hex key code"""
        x = INPUT(type=INPUT_KEYBOARD,
                 ki=KEYBDINPUT(wVk=hex_key_code, dwFlags=KEYEVENTF_KEYUP))
        user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

    @staticmethod
    def _handle_mouse_button(action: str, button_id: str) -> None:
        """Handle mouse button press/release actions"""
        if button_id not in MOUSE_BUTTONS:
            raise ValueError(f"Unknown mouse button: {button_id}")
            
        btn, down_flag, up_flag = MOUSE_BUTTONS[button_id]
        flag = down_flag if action == 'press' else up_flag
        extra = 0
        if 'XBUTTON' in button_id:
            extra = 1 if btn == 'x1' else 2
        # VOID WINAPI mouse_event(
        #   DWORD     dwFlags,      // flags for movement and button state
        #   DWORD     dx,           // change in x coordinate
        #   DWORD     dy,           // change in y coordinate 
        #   DWORD     dwData,       // wheel movement for X buttons
        #   ULONG_PTR dwExtraInfo   // extra info, usually 0
        # );
        user32.mouse_event(flag, 0, 0, 0, extra)

    @staticmethod
    def _handle_virtual_key(action: str, key_name: str, keymap: dict) -> None:
        """Handle virtual key press/release actions"""
        try:
            raw_code = keymap[key_name]
        except KeyError:
            raise ValueError(f"Unknown key name: {key_name}")

        try:
            key_code = int(raw_code, 16)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid key code for {key_name}: {raw_code}")

        if action == 'press':
            InputManager.press_key(key_code)
        else:
            InputManager.release_key(key_code)

    @classmethod
    def press_mouse_button(cls, button_id: str) -> None:
        """Press a mouse button"""
        cls._handle_mouse_button('press', button_id)

    @classmethod
    def release_mouse_button(cls, button_id: str) -> None:
        """Release a mouse button"""
        cls._handle_mouse_button('release', button_id)

    @classmethod
    def press_virtual_key(cls, key_name: str, keymap: dict) -> None:
        """Press a virtual key using the keymap"""
        cls._handle_virtual_key('press', key_name, keymap)

    @classmethod
    def release_virtual_key(cls, key_name: str, keymap: dict) -> None:
        """Release a virtual key using the keymap"""
        cls._handle_virtual_key('release', key_name, keymap)

    # === Game-specific input operations ===
    @classmethod
    def move_to_last_detection(cls) -> bool:
        """Move mouse to last detected position (OCR or image search)"""
        if cls.last_detection_type == "ocr" and cls.last_ocr_match_center:
            cls.move_mouse(*cls.last_ocr_match_center)
            time.sleep(0.2)
            return True
        if cls.last_detection_type == "img" and cls.last_img_match_center:
            cls.move_mouse(*cls.last_img_match_center)
            time.sleep(0.2)
            return True
        print("[!] No valid target position available for mouse move.")
        return False

    @classmethod
    def open_with_mouse(cls) -> None:
        """Open game interface element with right+left click sequence"""
        # Right click
        cls.press_mouse_button("VK_RBUTTON")
        time.sleep(0.05)
        cls.release_mouse_button("VK_RBUTTON")
        time.sleep(0.3)
        
        # Left click
        cls.press_mouse_button("VK_LBUTTON")
        time.sleep(0.05)
        cls.release_mouse_button("VK_LBUTTON")