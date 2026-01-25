#!/usr/bin/env python3
"""
Helper script to download checkpoint.pth file.
Provides multiple download methods.
"""
import os
import sys
from pathlib import Path
import subprocess

checkpoint_path = Path("models/ball_detection_open_soccer_ball/checkpoint.pth")

if not checkpoint_path.exists():
    print(f"❌ Error: {checkpoint_path} not found!")
    sys.exit(1)

file_size = checkpoint_path.stat().st_size
file_size_mb = file_size / (1024 * 1024)

print(f"📦 Checkpoint file found:")
print(f"   Path: {checkpoint_path.absolute()}")
print(f"   Size: {file_size_mb:.1f} MB ({file_size:,} bytes)")
print()

print("🌐 Starting HTTP server to download the file...")
print(f"   Server will run on port 8034")
print(f"   Access: http://localhost:8034/checkpoint.pth")
print()
print("💡 Download methods:")
print("   1. Browser: Open http://localhost:8034/checkpoint.pth")
print("   2. wget: wget http://localhost:8034/checkpoint.pth")
print("   3. curl: curl -O http://localhost:8034/checkpoint.pth")
print()
print("⚠️  Press Ctrl+C to stop the server after downloading")
print()

# Change to the checkpoint directory to serve the file
os.chdir(checkpoint_path.parent)

# Start HTTP server
try:
    subprocess.run([
        sys.executable, "-m", "http.server", "8034", "--bind", "0.0.0.0"
    ])
except KeyboardInterrupt:
    print("\n✅ Server stopped")
