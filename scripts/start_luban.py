#!/usr/bin/env python3
"""
Start LuBanSuoEnv.exe from the project root.

This script launches the Luban Lock Unity environment executable and exits
immediately after successful startup.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def check_server_available(host: str = "127.0.0.1", port: int = 9999, timeout: float = 2.0) -> bool:
    """Check if the Unity server is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((host, port))
        return True
    except Exception:
        return False


def main():
    # Get project root (directory containing this script)
    project_root = Path(__file__).resolve().parent.parent
    exe_path = project_root / "assets" / "Luban" / "LuBanSuoEnv.exe"
    
    # Check if executable exists
    if not exe_path.exists():
        print(f"Error: LuBanSuoEnv.exe not found at: {exe_path}")
        print("Please make sure you are running this script from the project root directory.")
        sys.exit(1)
    
    print(f"Starting LuBanSuoEnv.exe...")
    print(f"Executable path: {exe_path}")
    
    try:
        # Change to the Luban directory and start the executable
        exe_dir = exe_path.parent
        process = subprocess.Popen(
            [str(exe_path)],
            cwd=str(exe_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
            start_new_session=True,
        )
        
        # Wait a moment for the process to start
        time.sleep(0.5)
        
        # Check if process is still running
        if process.poll() is not None:
            print(f"Error: Process exited immediately with code {process.returncode}")
            sys.exit(1)
        
        print(f"✓ Process started successfully (PID: {process.pid})")
        
        # Optionally check if server is available (wait up to 5 seconds)
        print("Waiting for Unity server to be ready...", end="", flush=True)
        server_ready = False
        for i in range(10):  # Check 10 times over 5 seconds
            if check_server_available():
                server_ready = True
                break
            time.sleep(0.5)
            print(".", end="", flush=True)
        
        print()  # New line after dots
        
        if server_ready:
            print("✓ Unity server is ready and accepting connections")
        else:
            print("⚠ Unity server not yet ready (this is normal, it may take longer)")
            print("  The process is running, but the server may still be initializing.")
        
        print()
        print("LuBanSuoEnv.exe is running in the background.")
        print(f"Process ID: {process.pid}")
        print("To stop the application, close the Unity window or kill the process.")
        print()
        
        # Exit immediately - process will continue running
        sys.exit(0)
            
    except Exception as e:
        print(f"Error starting LuBanSuoEnv.exe: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
