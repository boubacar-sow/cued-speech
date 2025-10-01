#!/usr/bin/env python3
"""
Build script for Cued Speech package.

This script helps build the package without including the download folder
and data files that should be downloaded separately.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def clean_build_artifacts():
    """Clean previous build artifacts."""
    print("🧹 Cleaning build artifacts...")
    
    dirs_to_clean = ["dist", "build", "*.egg-info"]
    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"  Removed: {path}")
            elif path.is_file():
                path.unlink()
                print(f"  Removed: {path}")


def check_download_folder():
    """Check if download folder exists and warn user."""
    download_path = Path("download")
    if download_path.exists():
        print("⚠️  Warning: 'download' folder found in project root.")
        print("   This folder will be excluded from the package build.")
        print("   Users will download data files automatically when using the package.")
        
        # Show download folder contents
        print(f"   Download folder contains {len(list(download_path.iterdir()))} items")
        
        response = input("   Continue with build? (y/N): ")
        if response.lower() != 'y':
            print("Build cancelled.")
            sys.exit(0)
    else:
        print("✅ No download folder found - clean build environment.")


def build_package():
    """Build the package using setuptools."""
    print("🔨 Building package...")
    
    try:
        # Build wheel
        subprocess.run([sys.executable, "-m", "build", "--wheel"], check=True)
        print("✅ Wheel build successful!")
        
        # Build source distribution
        subprocess.run([sys.executable, "-m", "build", "--sdist"], check=True)
        print("✅ Source distribution build successful!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)


def show_build_results():
    """Show the results of the build."""
    print("\n📦 Build Results:")
    
    dist_dir = Path("dist")
    if dist_dir.exists():
        for file_path in dist_dir.iterdir():
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  - {file_path.name} ({size_mb:.1f} MB)")
    else:
        print("  No build artifacts found.")


def main():
    """Main build process."""
    print("🚀 Cued Speech Package Builder")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Clean previous builds
    clean_build_artifacts()
    
    # Check download folder
    check_download_folder()
    
    # Build package
    build_package()
    
    # Show results
    show_build_results()
    
    print("\n✅ Build completed successfully!")
    print("\n📋 Next steps:")
    print("  1. Test the package: pip install dist/*.whl")
    print("  2. Upload to PyPI: python -m twine upload dist/*")
    print("  3. Users can install with: pip install cued-speech")


if __name__ == "__main__":
    main() 