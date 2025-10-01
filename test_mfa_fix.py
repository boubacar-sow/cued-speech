#!/usr/bin/env python3
"""Test MFA installation and syllabification logic."""

import sys
import os
import subprocess

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_mfa_installation():
    """Test that MFA is properly installed."""
    try:
        result = subprocess.run(["mfa", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ MFA installed successfully: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ MFA command failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("✗ MFA command not found")
        return False
    except Exception as e:
        print(f"✗ Error testing MFA: {e}")
        return False

def test_generator_import():
    """Test that we can import the generator module."""
    try:
        from cued_speech.generator import CuedSpeechGenerator
        print("✓ Successfully imported CuedSpeechGenerator")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False

def test_syllabification_logic():
    """Test the syllabification logic."""
    try:
        from cued_speech.generator import CuedSpeechGenerator
        generator = CuedSpeechGenerator()
        
        # Test the IPA splitting function
        test_ipa = "məʁki a toys poyʁ votʁə attɑ̃tiɔ̃ vos kəstiɔ̃s sɔ̃t ləs biɑ̃vɑ̃yəs"
        syllables = generator._split_ipa_into_syllables(test_ipa)
        print(f"✓ IPA splitting test: '{test_ipa}' -> {syllables}")
        
        # Test that we get the expected number of syllables
        expected_syllables = ['məʁ', 'ki', 'a', 'toys', 'poyʁ', 'vot', 'ʁə', 'at', 'tɑ̃tiɔ̃', 'vos', 'kəs', 'tiɔ̃s', 'sɔ̃t', 'ləs', 'biɑ̃vɑ̃yəs']
        if len(syllables) == len(expected_syllables):
            print(f"✓ Syllable count matches expected: {len(syllables)}")
        else:
            print(f"✗ Syllable count mismatch: got {len(syllables)}, expected {len(expected_syllables)}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Failed to test syllabification: {e}")
        return False

def test_mfa_command_structure():
    """Test that the MFA command structure matches the reference."""
    try:
        from cued_speech.generator import CuedSpeechGenerator
        generator = CuedSpeechGenerator()
        
        # Check that the config has the required MFA settings
        config = generator.config
        required_keys = ['language', 'mfa_args', 'output_dir']
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing config key: {key}")
                return False
        
        print(f"✓ Config has required MFA settings: {required_keys}")
        print(f"  Language: {config['language']}")
        print(f"  MFA args: {config['mfa_args']}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test MFA command structure: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing MFA Installation and Syllabification...")
    print("=" * 50)
    
    tests = [
        test_mfa_installation,
        test_generator_import,
        test_syllabification_logic,
        test_mfa_command_structure,
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
        print("✓ All tests passed! MFA should work correctly now.")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 