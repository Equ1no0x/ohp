import os
import time
import threading
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

class ACTLogWatcher:
    """
    ACT log file watcher specifically for combat events.
    Monitors the most recent .log file in ACT's FFXIV logs directory and
    writes filtered combat events to a dedicated log file.
    """
    # Combat message constants
    MSG_HIT_THE = "you hit the"
    MSG_HITS_YOU = "hits you for"
    MSG_DEFEAT = "you defeat"
    MSG_DEFEAT_THE = "you defeat the"
    
    def __init__(self, poll_interval: float = 0.1):
        self.poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._combat_callbacks: List[callable] = []
        self._log_dir = self._resolve_log_dir()
        self._current_log: Optional[Path] = None
        
        # Combat event logging
        self._combat_log = Path(os.path.dirname(os.path.dirname(__file__))) / "assets" / "logs" / "combat_events.log"
        self._ensure_combat_log()
        
    def _resolve_log_dir(self) -> Path:
        """Get the ACT FFXIV logs directory path."""
        appdata = os.getenv('APPDATA', '')
        if not appdata:
            raise RuntimeError("Could not determine AppData directory")
        return Path(appdata) / "Advanced Combat Tracker" / "FFXIVLogs"

    def _find_latest_log(self) -> Optional[Path]:
        """Find the most recently modified .log file in the ACT logs directory."""
        try:
            log_files = list(self._log_dir.glob("*.log"))
            if not log_files:
                return None
            return max(log_files, key=lambda p: p.stat().st_mtime)
        except Exception as e:
            print(f"[!] Error finding latest log file: {e}")
            return None

    def register_combat_callback(self, callback: callable):
        """Register a callback for combat completion events."""
        with self._lock:
            self._combat_callbacks.append(callback)

    def _notify_combat_complete(self, monster_name: str):
        """Notify all registered callbacks about combat completion."""
        with self._lock:
            for callback in self._combat_callbacks:
                try:
                    callback(monster_name)
                except Exception as e:
                    print(f"[!] Error in combat callback: {e}")

    def start(self):
        """Start the log watcher thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ACTLogWatcher", daemon=True)
        self._thread.start()
        print("[*] ACT log watcher started")

    def stop(self):
        """Stop the log watcher thread."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("[*] ACT log watcher stopped")

    def _open_file(self, path: Path, start_at_end: bool = True):
        """Open the log file for reading."""
        f = path.open("r", encoding="utf-8", errors="replace")
        if start_at_end:
            f.seek(0, os.SEEK_END)
        return f

    def _ensure_combat_log(self):
        """Create combat log file if it doesn't exist."""
        try:
            if not self._combat_log.parent.exists():
                self._combat_log.parent.mkdir(parents=True, exist_ok=True)
            if not self._combat_log.exists():
                self._combat_log.write_text("")
        except Exception as e:
            print(f"[!] Error creating combat log file: {e}")

    def _write_combat_event(self, timestamp: str, monster_name: str):
        """Write a combat completion event to our filtered log."""
        try:
            event = {
                "timestamp": timestamp,
                "event": "defeat",
                "target": monster_name
            }
            with self._combat_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
            print(f"[*] Wrote combat event to log: {monster_name}")
        except Exception as e:
            print(f"[!] Failed to write combat event: {e}")

    def _process_line(self, line: str):
        """Process a single line from the log file.
        Expected format: 00|2025-10-29T21:11:57.0000000-06:00|0B3A||You defeat the sun bat.|d436c2cf2091c65b
        """
        try:
            # Split on pipe character, need at least 5 fields
            fields = line.strip().split("|")
            if len(fields) < 5:
                return
            
            # Extract timestamp and message
            timestamp = fields[1]
            message = fields[4]
            
            # Process and log combat-related messages
            msg_lower = message.lower()
            if any(x in msg_lower for x in [self.MSG_HIT_THE, self.MSG_HITS_YOU, self.MSG_DEFEAT]):
                # Only print defeat messages
                if self.MSG_DEFEAT in msg_lower:
                    print(f"[DEBUG] Combat: {message}")
                
                # Log all combat messages
                log_line = f"{timestamp} | {message}\n"
                with self._combat_log.open("a", encoding="utf-8") as f:
                    f.write(log_line)
                
                # For defeats, also notify callbacks
                if self.MSG_DEFEAT in msg_lower:
                    try:
                        # Extract monster name while preserving original case
                        if self.MSG_DEFEAT_THE in msg_lower:
                            index = message.lower().find(self.MSG_DEFEAT_THE) + len(self.MSG_DEFEAT_THE)
                            monster_name = message[index:].rstrip('.')
                        else:
                            index = message.lower().find(self.MSG_DEFEAT) + len(self.MSG_DEFEAT)
                            monster_name = message[index:].rstrip('.')
                        monster_name = monster_name.strip()
                        print(f"[DEBUG][Combat] Defeated: {monster_name}")
                        self._notify_combat_complete(monster_name)
                    except Exception as e:
                        print(f"[!] Error extracting monster name: {e}")
            
        except Exception as e:
            print(f"[!] Error processing log line: {e}")

    def get_recent_messages(self, count=5):
        """Get the most recent combat messages from the log."""
        try:
            with self._combat_log.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-count:]
            return [line.split(" | ", 1)[1].strip() for line in lines]
        except Exception as e:
            print(f"[!] Error reading recent messages: {e}")
            return []

    def _check_raw_combat_log(self, target_lower: str, since_timestamp: Optional[str] = None) -> tuple[bool, str | None]:
        """Check raw combat log for recent defeats.
        
        Args:
            target_lower: Lowercase target name to search for
            since_timestamp: Only check events after this timestamp (ACT format)
        """
        try:
            with self._combat_log.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-50:]  # Check last 50 lines
                
            for line in reversed(lines):
                try:
                    timestamp, message = line.split(" | ", 1)
                    if since_timestamp and timestamp <= since_timestamp:
                        continue  # Skip older messages
                        
                    message = message.strip()
                    msg_lower = message.lower()
                    if self.MSG_DEFEAT in msg_lower:
                        if self.MSG_DEFEAT_THE in msg_lower:
                            monster = msg_lower.split(self.MSG_DEFEAT_THE)[1].rstrip('.')
                        else:
                            monster = msg_lower.split(self.MSG_DEFEAT)[1].rstrip('.')
                        
                        if target_lower in monster:
                            print(f"[DEBUG] Found defeat in raw log ({timestamp}): {message}")
                            return True, monster, timestamp
                except Exception:
                    continue  # Skip malformed lines
        except Exception as e:
            print(f"[!] Error checking raw combat log: {e}")
        return False, None, None
    
    def _check_structured_events(self, target_lower: str, since_timestamp: Optional[str]) -> tuple[bool, str | None]:
        """Check structured event log for recent defeats."""
        try:
            with self._combat_log.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-1000:]  # Limit to last 1000 events
                print(f"[DEBUG] Reading {len(lines)} recent combat log entries")
            
            for line in reversed(lines):
                try:
                    event = json.loads(line)
                    print(f"[DEBUG] Checking combat event: {event}")
                    if (event["event"] == "defeat" and 
                        target_lower in event["target"].lower() and
                        (not since_timestamp or event["timestamp"] > since_timestamp)):
                        print(f"[DEBUG] Found matching combat completion in events log! Target: {event['target']}")
                        return True, event["target"]
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            print(f"[!] Error reading events log: {e}")
        return False, None
    
    def _wait_for_completion(self, target_name: str, timeout: float) -> tuple[bool, str | None]:
        """Wait for a new combat completion event."""
        found = threading.Event()
        defeated_name = [None]  # Use list to allow modification in callback
        
        def on_combat_complete(monster_name: str):
            if target_name.lower() in monster_name.lower():
                print(f"[DEBUG] Combat completion callback triggered for: {monster_name}")
                defeated_name[0] = monster_name
                found.set()
        
        # Register temp callback
        with self._lock:
            self._combat_callbacks.append(on_combat_complete)
            
        try:
            # Wait for the event
            result = found.wait(timeout)
            print(f"[DEBUG] Combat completion wait result: {result}")
            return result, defeated_name[0]
        finally:
            # Clean up callback
            with self._lock:
                try:
                    self._combat_callbacks.remove(on_combat_complete)
                except ValueError:
                    pass

    def check_combat_completion(self, target_name: str, timeout: float = 2.0, since_timestamp: Optional[str] = None) -> tuple[bool, str | None]:
        """Check if combat was completed for the given target.
        
        Args:
            target_name: The name of the target to check for
            timeout: How long to wait for new combat completions (seconds)
            since_timestamp: Optional timestamp to check from (in ACT log format)
                           If None, checks most recent events
                           
        Returns:
            tuple: (success, defeated_target_name or None)
        """
        print(f"[DEBUG] Checking combat completion for target: {target_name}")
        if since_timestamp:
            print(f"[DEBUG] Looking for defeats since: {since_timestamp}")
            
        target_lower = target_name.lower()
        last_timestamp = since_timestamp
        
        try:
            # First try finding in raw log
            success, monster, timestamp = self._check_raw_combat_log(target_lower, since_timestamp)
            if success:
                last_timestamp = timestamp
                return True, monster
                
            # Then try structured events
            success, monster = self._check_structured_events(target_lower, since_timestamp)
            if success:
                return True, monster
                
            # Finally wait for new completion
            print(f"[DEBUG] No existing completion found, waiting {timeout}s for new events")
            return self._wait_for_completion(target_name, timeout)
                        
        except Exception as e:
            print(f"[!] Error in combat completion check: {e}")
            return False, None

    def _process_chunk(self, chunk: str, buf: str) -> tuple[str, bool]:
        """Process a chunk of log data and return updated buffer."""
        buf += chunk
        while True:
            nl = buf.find("\n")
            if nl == -1:
                break
            line = buf[:nl]
            buf = buf[nl+1:]
            if line.endswith("\r"):
                line = line[:-1]
            self._process_line(line)
        return buf, True

    def _monitor_log_file(self, log_file: Path, file_handle) -> None:
        """Monitor a single log file for changes."""
        buf = ""
        while not self._stop.is_set():
            # Check if the file still exists and hasn't been rotated
            if not log_file.exists() or log_file != self._find_latest_log():
                break

            # Read new content
            chunk = file_handle.read()
            if not chunk:
                time.sleep(self.poll_interval)
                continue

            buf, _ = self._process_chunk(chunk, buf)

    def _run(self):
        """Main monitoring loop."""
        while not self._stop.is_set():
            try:
                # Check for the latest log file
                latest_log = self._find_latest_log()
                if not latest_log:
                    time.sleep(1.0)
                    continue

                # Open the appropriate file handle
                start_at_end = self._current_log != latest_log
                if start_at_end:
                    self._current_log = latest_log
                    print(f"[*] Monitoring log file: {latest_log.name}")
                
                f = self._open_file(latest_log, start_at_end=start_at_end)
                try:
                    self._monitor_log_file(latest_log, f)
                finally:
                    try:
                        f.close()
                    except Exception:
                        pass

            except Exception as e:
                print(f"[!] Error in log monitoring: {e}")
                time.sleep(1.0)  # Wait before retrying

if __name__ == "__main__":
    print("[*] Starting ACT Log Watcher in debug mode...")
    print("[*] This will monitor the ACT logs and print all combat completions.")
    print("[*] Press Ctrl+C to stop.")
    
    watcher = ACTLogWatcher()
    try:
        # Print the log directory we're monitoring
        print(f"\n[*] ACT Logs directory: {watcher._log_dir}")
        print(f"[*] Combat events will be written to: {watcher._combat_log}\n")
        
        # Start monitoring
        watcher.start()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[*] Stopping watcher...")
        watcher.stop()
        print("[*] Done.")