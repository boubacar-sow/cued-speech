#!/usr/bin/env python3
"""
Test script for the data manager functionality.

This script tests the automatic data downloading and management features.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cued_speech.data_manager import (
    get_data_dir,
    ensure_data_files,
    get_default_paths,
    list_data_files,
    get_data_file_path
)


def test_data_directory():
    """Test data directory functionality."""
    print("🧪 Testing data directory functionality...")
    
    data_dir = get_data_dir()
    print(f"  Data directory: {data_dir}")
    print(f"  Directory exists: {data_dir.exists()}")
    
    return data_dir


def test_default_paths():
    """Test default paths functionality."""
    print("\n🧪 Testing default paths...")
    
    paths = get_default_paths()
    print(f"  Number of expected files: {len(paths)}")
    
    for file_type, path in paths.items():
        print(f"  - {file_type}: {Path(path).name}")
    
    return paths


def test_data_file_paths():
    """Test individual file path resolution."""
    print("\n🧪 Testing individual file paths...")
    
    expected_files = ["model", "vocab", "lexicon", "kenlm_fr", "homophones", "kenlm_ipa"]
    
    for file_type in expected_files:
        path = get_data_file_path(file_type)
        if path:
            print(f"  ✅ {file_type}: {path.name}")
        else:
            print(f"  ❌ {file_type}: Not found")


def test_data_availability():
    """Test data availability checking."""
    print("\n🧪 Testing data availability...")
    
    try:
        files = ensure_data_files()
        print(f"  Available files: {len(files)}")
        
        for file_type, file_path in files.items():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  - {file_type}: {file_path.name} ({size_mb:.1f} MB)")
            
    except Exception as e:
        print(f"  ❌ Error checking data availability: {e}")


def main():
    """Run all tests."""
    print("🚀 Testing Cued Speech Data Manager")
    print("=" * 50)
    
    # Test basic functionality
    test_data_directory()
    test_default_paths()
    test_data_file_paths()
    
    # Test data availability (this might trigger download)
    print("\n⚠️  This test might download data files if they're missing...")
    response = input("Continue? (y/N): ")
    
    if response.lower() == 'y':
        test_data_availability()
    else:
        print("Skipping data availability test.")
    
    print("\n✅ Testing completed!")


if __name__ == "__main__":
    main() 