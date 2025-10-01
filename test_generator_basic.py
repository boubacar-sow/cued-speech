#!/usr/bin/env python3
"""Basic test for the cued speech generator module."""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

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

def test_generator_methods():
    """Test that the generator has expected methods."""
    try:
        from cued_speech.generator import CuedSpeechGenerator
        generator = CuedSpeechGenerator()
        
        # Check for expected methods
        expected_methods = ['generate_cue', '_get_default_config']
        for method in expected_methods:
            if hasattr(generator, method):
                print(f"✓ Generator has method: {method}")
            else:
                print(f"✗ Generator missing method: {method}")
                return False
        return True
    except Exception as e:
        print(f"✗ Failed to test generator methods: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Cued Speech Generator...")
    print("=" * 40)
    
    tests = [
        test_generator_import,
        test_generator_initialization,
        test_generator_methods,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 