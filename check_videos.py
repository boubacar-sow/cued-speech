#!/usr/bin/env python3
"""
Quick script to check what videos are available in the input directory.
"""

import os
from pathlib import Path

def check_videos():
    input_directory = "/pasteur/appa/homes/bsow/ACSR/data/generated_videos/"
    
    print(f"🔍 Checking videos in: {input_directory}")
    
    if not os.path.exists(input_directory):
        print(f"❌ Directory does not exist: {input_directory}")
        return
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    video_files = []
    
    directory_path = Path(input_directory)
    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    if video_files:
        print(f"📹 Found {len(video_files)} video files:")
        for i, video in enumerate(sorted(video_files), 1):
            size_mb = video.stat().st_size / (1024 * 1024)
            print(f"   {i}. {video.name} ({size_mb:.1f} MB)")
    else:
        print("⚠️ No video files found")

if __name__ == "__main__":
    check_videos()


