#!/usr/bin/env python3
"""Test the fixed cued speech generator."""

import sys
import os
import tempfile
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_generator_import():
    """Test that we can import the generator module."""
    try:
        from cued_speech.generator import CuedSpeechGenerator, generate_cue
        print("✓ Successfully imported CuedSpeechGenerator and generate_cue")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False

def test_generator_initialization():
    """Test that the generator can be initialized."""
    try:
        from cued_speech.generator import CuedSpeechGenerator
        generator = CuedSpeechGenerator()
        print("✓ Successfully initialized CuedSpeechGenerator")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize generator: {e}")
        return False

def test_syllabification_logic():
    """Test the syllabification logic with sample data."""
    try:
        from cued_speech.generator import CuedSpeechGenerator
        generator = CuedSpeechGenerator()
        
        # Test the syllabification with a simple text
        test_text = "merci beaucoup"
        test_audio_path = "/tmp/test_audio.wav"  # This won't exist, but we're testing the logic
        
        # Test the IPA splitting function
        test_ipa = "məʁsiboku"
        syllables = generator._split_ipa_into_syllables(test_ipa)
        print(f"✓ IPA splitting test: '{test_ipa}' -> {syllables}")
        
        # Test basic syllable timing creation
        enhanced_syllables = generator._create_basic_syllable_timing(test_text, test_audio_path)
        print(f"✓ Basic syllable timing created: {len(enhanced_syllables)} syllables")
        
        for i, syl in enumerate(enhanced_syllables[:3]):  # Show first 3
            print(f"  Syllable {i}: {syl['syllable']} ({syl['type']}) - a1:{syl['a1']:.2f}, a3:{syl['a3']:.2f}, m1:{syl['m1']:.2f}, m2:{syl['m2']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test syllabification: {e}")
        return False

def test_mediapipe_initialization():
    """Test that MediaPipe can be initialized."""
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        print("✓ MediaPipe FaceMesh initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize MediaPipe: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Fixed Cued Speech Generator...")
    print("=" * 50)
    
    tests = [
        test_generator_import,
        test_generator_initialization,
        test_syllabification_logic,
        test_mediapipe_initialization,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 