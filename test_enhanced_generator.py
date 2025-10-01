#!/usr/bin/env python3
"""
Test script for enhanced cued speech generator with new features:
- Easing functions
- Hand shape morphing
- Transparency effects
- Curved trajectories
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cued_speech.generator import CuedSpeechGenerator

def test_enhanced_features():
    """Test different combinations of enhanced features."""
    
    # Test video path (you'll need to provide your own test video)
    video_path = "download/test_generate.mp4"
    
    if not os.path.exists(video_path):
        print(f"Test video not found: {video_path}")
        print("Please provide a test video file.")
        return
    
    # Test text
    test_text = "bonjour comment allez-vous"
    
    # Test different configurations
    test_configs = [
        {
            "name": "Default (All features enabled)",
            "easing_function": "ease_in_out_cubic",
            "enable_morphing": True,
            "enable_transparency": True,
            "enable_curving": True,
        },
        {
            "name": "Linear easing only",
            "easing_function": "linear",
            "enable_morphing": True,
            "enable_transparency": True,
            "enable_curving": True,
        },
        {
            "name": "Elastic easing",
            "easing_function": "ease_out_elastic",
            "enable_morphing": True,
            "enable_transparency": True,
            "enable_curving": True,
        },
        {
            "name": "No morphing",
            "easing_function": "ease_in_out_cubic",
            "enable_morphing": False,
            "enable_transparency": True,
            "enable_curving": True,
        },
        {
            "name": "No transparency",
            "easing_function": "ease_in_out_cubic",
            "enable_morphing": True,
            "enable_transparency": False,
            "enable_curving": True,
        },
        {
            "name": "No curving",
            "easing_function": "ease_in_out_cubic",
            "enable_morphing": True,
            "enable_transparency": True,
            "enable_curving": False,
        },
        {
            "name": "Minimal features",
            "easing_function": "linear",
            "enable_morphing": False,
            "enable_transparency": False,
            "enable_curving": False,
        },
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {config['name']}")
        print(f"{'='*60}")
        
        # Create output directory for this test
        output_dir = f"output/enhanced_test_{i+1}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "enhanced_video.mp4")
        
        try:
            # Create generator with current configuration
            generator_config = {
                "easing_function": config["easing_function"],
                "enable_morphing": config["enable_morphing"],
                "enable_transparency": config["enable_transparency"],
                "enable_curving": config["enable_curving"],
            }
            generator = CuedSpeechGenerator(generator_config)
            
            # Generate video with current configuration
            result = generator.generate_cue(
                text=test_text,
                video_path=video_path,
                output_path=output_path
            )
            
            print(f"✅ Success: {result}")
            print(f"   Easing: {config['easing_function']}")
            print(f"   Morphing: {config['enable_morphing']}")
            print(f"   Transparency: {config['enable_transparency']}")
            print(f"   Curving: {config['enable_curving']}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            print(f"   Configuration: {config}")

def test_easing_functions():
    """Test all available easing functions."""
    print("\n" + "="*60)
    print("Testing Easing Functions")
    print("="*60)
    
    from cued_speech.generator import get_easing_function, linear_easing, ease_in_out_cubic, ease_out_elastic, ease_in_out_back
    
    easing_functions = {
        "linear": linear_easing,
        "ease_in_out_cubic": ease_in_out_cubic,
        "ease_out_elastic": ease_out_elastic,
        "ease_in_out_back": ease_in_out_back,
    }
    
    for name, func in easing_functions.items():
        print(f"\n{name}:")
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = func(t)
            print(f"  t={t:.2f} -> {result:.4f}")

if __name__ == "__main__":
    print("Enhanced Cued Speech Generator Test")
    print("="*60)
    
    # Test easing functions
    test_easing_functions()
    
    # Test enhanced features
    test_enhanced_features()
    
    print("\n" + "="*60)
    print("Test completed!")
    print("Check the output/enhanced_test_* directories for generated videos.")
    print("="*60)
