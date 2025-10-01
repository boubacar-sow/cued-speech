#!/usr/bin/env python3
"""
Batch processing script for cued speech generation.
Processes all videos in /pasteur/appa/homes/bsow/ACSR/data/generated_videos/
with specified parameters and removes the _cued suffix from output filenames.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

# Add the src directory to the path so we can import the generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cued_speech.generator import CuedSpeechGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_processing.log')
    ]
)
logger = logging.getLogger(__name__)

def find_video_files(directory: str) -> List[str]:
    """Find all video files in the specified directory."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    video_files = []
    
    directory_path = Path(directory)
    if not directory_path.exists():
        logger.error(f"Directory does not exist: {directory}")
        return video_files
    
    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(str(file_path))
    
    return sorted(video_files)

def remove_cued_suffix(file_path: str) -> str:
    """Remove the _cued suffix from filename if present."""
    path = Path(file_path)
    name = path.stem
    suffix = path.suffix
    
    # Remove _cued suffix if present
    if name.endswith('_cued'):
        name = name[:-5]  # Remove '_cued'
    
    return str(path.parent / f"{name}{suffix}")

def process_video(generator: CuedSpeechGenerator, input_path: str, output_dir: str) -> bool:
    """Process a single video with the specified parameters."""
    try:
        # Get the base filename without extension
        input_filename = Path(input_path).stem
        output_filename = f"{input_filename}.mp4"  # Always use .mp4 for output
        output_path = os.path.join(output_dir, output_filename)
        
        logger.info(f"🎬 Processing: {input_path}")
        logger.info(f"📁 Output: {output_path}")
        
        # Generate cued speech video with specified parameters
        result_path = generator.generate_cue(
            text=None,  # Extract text from video using Whisper
            video_path=input_path,
            output_path=output_path,
            audio_path=None,
            # Parameters as requested
            enable_morphing=False,
            enable_transparency=False,
            enable_curving=False,
            easing_function="linear",
            language="french"
        )
        
        # The generator creates files with _cued suffix, so we need to rename
        if result_path.endswith('_cued.mp4'):
            final_path = remove_cued_suffix(result_path)
            if os.path.exists(result_path):
                os.rename(result_path, final_path)
                logger.info(f"✅ Renamed: {result_path} -> {final_path}")
                return True
            else:
                logger.error(f"❌ Generated file not found: {result_path}")
                return False
        else:
            logger.info(f"✅ Processing completed: {result_path}")
            return True
            
    except Exception as e:
        logger.error(f"💥 Error processing {input_path}: {e}")
        return False

def main():
    """Main function to process all videos."""
    # Configuration
    input_directory = "/pasteur/appa/homes/bsow/ACSR/data/generated_videos/"
    output_directory = "output/generator"
    
    logger.info("🚀 Starting batch video processing...")
    logger.info(f"📂 Input directory: {input_directory}")
    logger.info(f"📁 Output directory: {output_directory}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Find all video files
    video_files = find_video_files(input_directory)
    
    if not video_files:
        logger.warning(f"⚠️ No video files found in {input_directory}")
        return
    
    logger.info(f"📹 Found {len(video_files)} video files to process")
    
    # Initialize the generator
    generator = CuedSpeechGenerator()
    
    # Process each video
    successful = 0
    failed = 0
    
    for i, video_path in enumerate(video_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"🎬 Processing video {i}/{len(video_files)}")
        logger.info(f"{'='*60}")
        
        if process_video(generator, video_path, output_directory):
            successful += 1
            logger.info(f"✅ Successfully processed: {Path(video_path).name}")
        else:
            failed += 1
            logger.error(f"❌ Failed to process: {Path(video_path).name}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 BATCH PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Successfully processed: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📹 Total videos: {len(video_files)}")
    logger.info(f"📈 Success rate: {(successful/len(video_files)*100):.1f}%")
    
    if failed > 0:
        logger.warning(f"⚠️ {failed} videos failed to process. Check the logs for details.")
    else:
        logger.info("🎉 All videos processed successfully!")

if __name__ == "__main__":
    main()


