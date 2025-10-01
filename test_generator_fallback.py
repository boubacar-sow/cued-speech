#!/usr/bin/env python3
"""Test the generator with fallback syllabification."""

import sys
import os
import tempfile
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_generator_fallback():
    """Test that the generator works with fallback syllabification."""
    try:
        from cued_speech.generator import CuedSpeechGenerator
        
        # Create a temporary config for testing
        config = {
            "video_path": "download/test_generate.mp4",  # This might not exist, but we're testing the logic
            "output_dir": "output/test",
            "handshapes_dir": "download/handshapes/coordinates",
            "language": "french",
            "reference_face_size": 0.3,
            "hand_scale_factor": 0.75,
            "mfa_args": ["--beam", "200", "--retry_beam", "400", "--fine_tune"],
            "video_codec": "libx265",
            "audio_codec": "aac"
        }
        
        generator = CuedSpeechGenerator(config)
        print("✓ Generator initialized successfully")
        
        # Test the fallback syllabification logic
        test_text = "Merci à tous pour votre attention. Vos questions sont les bienvenues."
        test_audio_path = "/tmp/test_audio.wav"  # This won't exist, but we're testing the logic
        
        # Test the fallback method directly
        enhanced_syllables = generator._create_basic_syllable_timing(test_text, test_audio_path)
        print(f"✓ Fallback syllabification created {len(enhanced_syllables)} syllables")
        
        # Show the first few syllables
        for i, syl in enumerate(enhanced_syllables[:5]):
            print(f"  Syllable {i}: '{syl['syllable']}' ({syl['type']}) - a1:{syl['a1']:.2f}, a3:{syl['a3']:.2f}, m1:{syl['m1']:.2f}, m2:{syl['m2']:.2f}")
        
        # Verify the structure matches the reference
        required_keys = ['syllable', 'a1', 'a3', 'm1', 'm2', 'type']
        for syl in enhanced_syllables:
            for key in required_keys:
                if key not in syl:
                    print(f"✗ Missing key '{key}' in syllable: {syl}")
                    return False
        
        print("✓ All syllables have required structure")
        return True
        
    except Exception as e:
        print(f"✗ Failed to test generator fallback: {e}")
        return False

def test_syllabification_consistency():
    """Test that syllabification is consistent with the reference."""
    try:
        from cued_speech.generator import CuedSpeechGenerator
        generator = CuedSpeechGenerator()
        
        # Test with the same text as the reference
        test_ipa = "məʁki a toys poyʁ votʁə attɑ̃tiɔ̃ vos kəstiɔ̃s sɔ̃t ləs biɑ̃vɑ̃yəs"
        syllables = generator._split_ipa_into_syllables(test_ipa)
        
        # Expected syllables from the reference
        expected_syllables = ['məʁ', 'ki', 'a', 'toys', 'poyʁ', 'vot', 'ʁə', 'at', 'tɑ̃tiɔ̃', 'vos', 'kəs', 'tiɔ̃s', 'sɔ̃t', 'ləs', 'biɑ̃vɑ̃yəs']
        
        if syllables == expected_syllables:
            print("✓ Syllabification matches reference exactly")
            return True
        else:
            print(f"✗ Syllabification mismatch:")
            print(f"  Expected: {expected_syllables}")
            print(f"  Got:      {syllables}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to test syllabification consistency: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Generator with Fallback Syllabification...")
    print("=" * 60)
    
    tests = [
        test_generator_fallback,
        test_syllabification_consistency,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The generator should work correctly with fallback syllabification.")
        print("\nNote: MFA has dependency conflicts, but the fallback syllabification logic")
        print("matches the reference code exactly and should produce correct results.")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 