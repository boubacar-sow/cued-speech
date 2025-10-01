#!/usr/bin/env python3
"""
Sequential batch processing script to generate cued speech videos for all videos in a directory.
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cued_speech.generator import CuedSpeechGenerator

def process_video(video_path, output_dir, config=None):
    """Process a single video and generate cued speech."""
    try:
        # Get original filename
        original_filename = os.path.basename(video_path)
        name, ext = os.path.splitext(original_filename)
        
        # Create output path with same name
        output_path = os.path.join(output_dir, f"{name}{ext}")
        
        print(f"Processing: {original_filename}")
        start_time = time.time()
        
        # Create generator
        generator = CuedSpeechGenerator(config)
        
        # Generate cued speech video
        result = generator.generate_cue(
            text=None,  # Extract text from video using Whisper
            video_path=video_path,
            output_path=output_path
        )
        
        processing_time = time.time() - start_time
        print(f"✅ Completed: {original_filename} ({processing_time:.1f}s)")
        
        return {
            'video': original_filename,
            'status': 'success',
            'output': result,
            'time': processing_time
        }
        
    except Exception as e:
        print(f"❌ Failed: {original_filename} - {str(e)}")
        return {
            'video': original_filename,
            'status': 'failed',
            'error': str(e),
            'time': 0
        }

def main():
    """Process all videos in the specified directory sequentially."""
    
    # Configuration
    input_dir = "/pasteur/appa/homes/bsow/ACSR/data/generated_videos"
    output_dir = "/pasteur/appa/homes/bsow/ACSR/data/cued_speech_videos"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(input_dir).glob(f"*{ext}"))
        video_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Configuration for all videos
    config = {
        "easing_function": "ease_in_out_cubic",
        "enable_morphing": True,
        "enable_transparency": True,
        "enable_curving": True,
    }
    
    # Process videos sequentially
    results = []
    total_start_time = time.time()
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")
        
        result = process_video(str(video_path), output_dir, config)
        results.append(result)
        
        # Small delay between videos to avoid overwhelming the system
        time.sleep(1)
    
    # Summary
    total_time = time.time() - total_start_time
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total videos: {len(video_files)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per video: {total_time/len(video_files):.1f}s")
    
    if successful:
        avg_success_time = sum(r['time'] for r in successful) / len(successful)
        print(f"Average success time: {avg_success_time:.1f}s")
    
    if failed:
        print(f"\nFailed videos:")
        for result in failed:
            print(f"  • {result['video']}: {result['error']}")
    
    print(f"\nOutput directory: {output_dir}")
    print("All intermediate files have been cleaned up automatically.")

if __name__ == "__main__":
    main()
