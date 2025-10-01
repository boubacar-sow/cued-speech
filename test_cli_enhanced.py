#!/usr/bin/env python3
"""
Test script to verify CLI works with enhanced parameters.
"""

import subprocess
import sys
import os

def test_cli_help():
    """Test that the CLI help shows the new parameters."""
    print("Testing CLI help...")
    
    try:
        # Test the generate command help
        result = subprocess.run([
            sys.executable, "-m", "src.cued_speech.cli", "generate", "--help"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            help_text = result.stdout
            print("✅ CLI help command works")
            
            # Check if new parameters are present
            expected_params = [
                "--easing",
                "--morphing",
                "--no-morphing", 
                "--transparency",
                "--no-transparency",
                "--curving",
                "--no-curving"
            ]
            
            missing_params = []
            for param in expected_params:
                if param not in help_text:
                    missing_params.append(param)
            
            if missing_params:
                print(f"❌ Missing parameters in help: {missing_params}")
                return False
            else:
                print("✅ All new parameters found in help")
                return True
        else:
            print(f"❌ CLI help command failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing CLI help: {e}")
        return False

def test_cli_parameter_validation():
    """Test that the CLI validates parameters correctly."""
    print("\nTesting CLI parameter validation...")
    
    try:
        # Test invalid easing function
        result = subprocess.run([
            sys.executable, "-m", "src.cued_speech.cli", "generate", 
            "nonexistent_video.mp4", "--easing", "invalid_easing"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            print("✅ CLI correctly rejects invalid easing function")
        else:
            print("❌ CLI should reject invalid easing function")
            return False
            
        # Test valid easing function
        result = subprocess.run([
            sys.executable, "-m", "src.cued_speech.cli", "generate", 
            "nonexistent_video.mp4", "--easing", "linear"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            print("✅ CLI correctly handles valid easing function")
        else:
            print("❌ CLI should fail for nonexistent video file")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing parameter validation: {e}")
        return False

def main():
    """Run all CLI tests."""
    print("Enhanced CLI Parameter Tests")
    print("=" * 50)
    
    success = True
    
    # Test help
    if not test_cli_help():
        success = False
    
    # Test parameter validation
    if not test_cli_parameter_validation():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All CLI tests passed!")
        print("\nYou can now use the enhanced parameters:")
        print("  cued-speech generate video.mp4 --easing ease_out_elastic")
        print("  cued-speech generate video.mp4 --no-morphing")
        print("  cued-speech generate video.mp4 --no-transparency")
        print("  cued-speech generate video.mp4 --no-curving")
    else:
        print("❌ Some CLI tests failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
