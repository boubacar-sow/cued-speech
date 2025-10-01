#!/usr/bin/env python3
"""
Test script to verify that the generator saves files with original filename.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cued_speech.generator import CuedSpeechGenerator

def test_filename_output():
    """Test that the generator saves files with the original filename."""
    print("Testing filename output...")
    
    # Test video path (you'll need to provide your own test video)
    video_path = "download/test_generate.mp4"
    
    if not os.path.exists(video_path):
        print(f"Test video not found: {video_path}")
        print("Please provide a test video file.")
        return False
    
    # Test text
    test_text = "bonjour"
    
    # Create output directory
    output_dir = "output/filename_test"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create generator
        config = {
            "easing_function": "linear",
            "enable_morphing": False,
            "enable_transparency": False,
            "enable_curving": False
        }
        generator = CuedSpeechGenerator(config)
        
        # Generate video
        result = generator.generate_cue(
            text=test_text,
            video_path=video_path,
            output_path=os.path.join(output_dir, "test_output.mp4")
        )
        
        print(f"✅ Generation completed: {result}")
        
        # Check the filename
        expected_filename = "test_generate_cued.mp4"
        actual_filename = os.path.basename(result)
        
        if actual_filename == expected_filename:
            print(f"✅ Correct filename: {actual_filename}")
            return True
        else:
            print(f"❌ Wrong filename. Expected: {expected_filename}, Got: {actual_filename}")
            return False
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return False

def test_no_intermediate_files():
    """Test that no intermediate files are left behind."""
    print("\nTesting for intermediate files...")
    
    output_dir = "output/filename_test"
    
    # Check for intermediate files that should not exist
    intermediate_files = [
        "rendered_video.mp4",
        "temp_video.mp4",
        "temp_rendered_video.mp4",
        "audio.wav"
    ]
    
    found_intermediate = []
    for filename in intermediate_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            found_intermediate.append(filename)
    
    if found_intermediate:
        print(f"❌ Found intermediate files: {found_intermediate}")
        return False
    else:
        print("✅ No intermediate files found")
        return True

def main():
    """Run all tests."""
    print("Filename Output Tests")
    print("=" * 50)
    
    success = True
    
    # Test filename output
    if not test_filename_output():
        success = False
    
    # Test for intermediate files
    if not test_no_intermediate_files():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed!")
        print("\nThe generator now:")
        print("  • Saves files with original filename + '_cued' suffix")
        print("  • Doesn't leave intermediate files behind")
        print("  • Cleans up temporary files automatically")
    else:
        print("❌ Some tests failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
