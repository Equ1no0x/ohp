"""System-level actions for window management, timing, and clipboard operations."""
import random
import time
import win32api
import win32con
import win32gui
import win32process
import psutil
import pyperclip

class SystemActions:
    """Manages system-level operations like window management, timing, and clipboard."""
    
    def copy_to_clipboard(self, text: str) -> None:
        """Copy text to system clipboard."""
        pyperclip.copy("")  # Clear clipboard first
        time.sleep(0.05)
        pyperclip.copy(text)
    
    def sleep_ms(self, duration_ms: int) -> None:
        """Sleep for specified milliseconds."""
        try:
            time.sleep(float(duration_ms) / 1000.0)
        except (TypeError, ValueError):
            print(f"[!] Invalid wait value: {duration_ms}")
    
    def wait_sec(self, duration_sec: float) -> None:
        """Sleep for specified seconds."""
        try:
            time.sleep(float(duration_sec))
        except (TypeError, ValueError):
            print(f"[!] Invalid wait value: {duration_sec}")
    
    def find_process_window(self, process_name: str) -> int | None:
        """Find window handle for a process by name."""
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] == process_name:
                print(f"[+] Found process {process_name} (PID: {proc.pid})")
                return self.get_window_by_pid(proc.info['pid'])
        return None

    def get_window_by_pid(self, pid: int) -> int | None:
        """Get window handle for a process by PID."""
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

    def focus_window(self, hwnd: int) -> None:
        """Focus a window by handle."""
        try:
            current_foreground = win32gui.GetForegroundWindow()
            if current_foreground == hwnd:
                return

            print("[*] Bringing window to front...")

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

            print("[*] Window is now in focus.")

        except Exception as e:
            print(f"[!] Failed to bring window to front: {e}")

    def random_wait(self, low_ms: int, high_ms: int) -> None:
        """Sleep for a random duration between low_ms and high_ms milliseconds."""
        try:
            low_i = int(low_ms)
            high_i = int(high_ms)
            if low_i > high_i:
                low_i, high_i = high_i, low_i
            delay = random.randint(low_i, high_i)
            print(f"[*] Random wait: {delay}ms")
            self.sleep_ms(delay)
        except (TypeError, ValueError) as e:
            print(f"[!] Invalid random wait bounds: {e}")