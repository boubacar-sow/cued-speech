#!/usr/bin/env python3
"""
Monitor the progress of batch video processing.
"""

import os
import time
from pathlib import Path

def monitor_progress():
    output_dir = "output/generator"
    log_file = "batch_processing.log"
    
    print("🔍 Monitoring batch processing progress...")
    print(f"📁 Output directory: {output_dir}")
    print(f"📋 Log file: {log_file}")
    print("=" * 60)
    
    # Check if log file exists
    if not os.path.exists(log_file):
        print("⚠️ Log file not found. Processing may not have started yet.")
        return
    
    # Count completed videos
    completed_videos = []
    if os.path.exists(output_dir):
        for file in Path(output_dir).glob("*.mp4"):
            if not file.name.endswith("_cued.mp4"):  # Only count final files
                completed_videos.append(file.name)
    
    print(f"✅ Completed videos: {len(completed_videos)}")
    if completed_videos:
        print("📹 Completed files:")
        for video in sorted(completed_videos):
            print(f"   - {video}")
    
    # Show recent log entries
    if os.path.exists(log_file):
        print(f"\n📋 Recent log entries:")
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                # Show last 10 lines
                for line in lines[-10:]:
                    print(f"   {line.strip()}")
        except Exception as e:
            print(f"   Error reading log: {e}")

if __name__ == "__main__":
    monitor_progress()


