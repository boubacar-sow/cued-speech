#!/usr/bin/env python3
"""
Example usage of enhanced cued speech generator features.

This script demonstrates how to use the new features:
- Easing functions for smooth transitions
- Hand shape morphing for natural shape changes
- Transparency effects during transitions
- Curved trajectories for obstacle avoidance
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cued_speech.generator import CuedSpeechGenerator

def example_basic_usage():
    """Basic usage with default settings (all features enabled)."""
    print("Example 1: Basic usage with all features enabled")
    print("-" * 50)
    
    generator = CuedSpeechGenerator()
    result = generator.generate_cue(
        text="bonjour comment allez-vous",
        video_path="download/test_generate.mp4",
        output_path="output/example_basic.mp4"
    )
    print(f"Generated: {result}")
    print("Features used: cubic easing, morphing, transparency, curving")

def example_custom_easing():
    """Example with different easing functions."""
    print("\nExample 2: Custom easing functions")
    print("-" * 50)
    
    easing_functions = ["linear", "ease_in_out_cubic", "ease_out_elastic", "ease_in_out_back"]
    
    for easing in easing_functions:
        config = {"easing_function": easing}
        generator = CuedSpeechGenerator(config)
        result = generator.generate_cue(
            text="merci beaucoup",
            video_path="download/test_generate.mp4",
            output_path=f"output/example_easing_{easing}.mp4"
        )
        print(f"Generated with {easing} easing: {result}")

def example_feature_comparison():
    """Compare different feature combinations."""
    print("\nExample 3: Feature comparison")
    print("-" * 50)
    
    # Test with morphing enabled vs disabled
    config_with_morphing = {"enable_morphing": True}
    generator_with_morphing = CuedSpeechGenerator(config_with_morphing)
    result_with_morphing = generator_with_morphing.generate_cue(
        text="au revoir",
        video_path="download/test_generate.mp4",
        output_path="output/example_with_morphing.mp4"
    )
    print(f"With morphing: {result_with_morphing}")
    
    config_without_morphing = {"enable_morphing": False}
    generator_without_morphing = CuedSpeechGenerator(config_without_morphing)
    result_without_morphing = generator_without_morphing.generate_cue(
        text="au revoir",
        video_path="download/test_generate.mp4",
        output_path="output/example_without_morphing.mp4"
    )
    print(f"Without morphing: {result_without_morphing}")

def example_transparency_effects():
    """Example with transparency effects."""
    print("\nExample 4: Transparency effects")
    print("-" * 50)
    
    # With transparency
    config_with_transparency = {"enable_transparency": True}
    generator_with_transparency = CuedSpeechGenerator(config_with_transparency)
    result_with_transparency = generator_with_transparency.generate_cue(
        text="salut",
        video_path="download/test_generate.mp4",
        output_path="output/example_with_transparency.mp4"
    )
    print(f"With transparency: {result_with_transparency}")
    
    # Without transparency
    config_without_transparency = {"enable_transparency": False}
    generator_without_transparency = CuedSpeechGenerator(config_without_transparency)
    result_without_transparency = generator_without_transparency.generate_cue(
        text="salut",
        video_path="download/test_generate.mp4",
        output_path="output/example_without_transparency.mp4"
    )
    print(f"Without transparency: {result_without_transparency}")

def example_curved_trajectories():
    """Example with curved trajectories."""
    print("\nExample 5: Curved trajectories")
    print("-" * 50)
    
    # With curving
    config_with_curving = {"enable_curving": True}
    generator_with_curving = CuedSpeechGenerator(config_with_curving)
    result_with_curving = generator_with_curving.generate_cue(
        text="comment allez-vous",
        video_path="download/test_generate.mp4",
        output_path="output/example_with_curving.mp4"
    )
    print(f"With curving: {result_with_curving}")
    
    # Without curving (straight lines only)
    config_without_curving = {"enable_curving": False}
    generator_without_curving = CuedSpeechGenerator(config_without_curving)
    result_without_curving = generator_without_curving.generate_cue(
        text="comment allez-vous",
        video_path="download/test_generate.mp4",
        output_path="output/example_without_curving.mp4"
    )
    print(f"Without curving: {result_without_curving}")

def example_custom_configuration():
    """Example with custom configuration."""
    print("\nExample 6: Custom configuration")
    print("-" * 50)
    
    # Create a custom configuration
    custom_config = {
        "easing_function": "ease_out_elastic",
        "enable_morphing": True,
        "enable_transparency": False,
        "enable_curving": True,
    }
    
    generator = CuedSpeechGenerator(custom_config)
    result = generator.generate_cue(
        text="bonne journée",
        video_path="download/test_generate.mp4",
        output_path="output/example_custom_config.mp4"
    )
    print(f"Custom config result: {result}")
    print(f"Config used: {custom_config}")

def example_minimal_features():
    """Example with minimal features (like original version)."""
    print("\nExample 7: Minimal features (original behavior)")
    print("-" * 50)
    
    minimal_config = {
        "easing_function": "linear",
        "enable_morphing": False,
        "enable_transparency": False,
        "enable_curving": False
    }
    generator = CuedSpeechGenerator(minimal_config)
    result = generator.generate_cue(
        text="merci",
        video_path="download/test_generate.mp4",
        output_path="output/example_minimal.mp4"
    )
    print(f"Minimal features result: {result}")
    print("This should behave like the original generator")

if __name__ == "__main__":
    print("Enhanced Cued Speech Generator - Usage Examples")
    print("=" * 60)
    
    # Check if test video exists
    if not os.path.exists("download/test_generate.mp4"):
        print("❌ Test video not found: download/test_generate.mp4")
        print("Please provide a test video file to run these examples.")
        print("\nYou can still see the API usage patterns above.")
        sys.exit(1)
    
    try:
        example_basic_usage()
        example_custom_easing()
        example_feature_comparison()
        example_transparency_effects()
        example_curved_trajectories()
        example_custom_configuration()
        example_minimal_features()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("Check the output/ directory for generated videos.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have the required dependencies installed.")
