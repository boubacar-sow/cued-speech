#!/usr/bin/env python3
"""
Script to prepare model files for GitHub release.

This script helps you organize and prepare your model files for uploading to GitHub releases.
"""

import os
import shutil
import json
from pathlib import Path
from cued_speech.data import get_default_model_paths, create_model_files_manifest

def prepare_model_files_for_release(output_dir: str = "model_files_for_release"):
    """Prepare model files for GitHub release.
    
    Args:
        output_dir: Directory to copy model files to
    """
    print("Preparing model files for GitHub release...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get default model paths
    default_paths = get_default_model_paths()
    
    # Create manifest
    manifest = create_model_files_manifest()
    
    copied_files = []
    missing_files = []
    
    print(f"Copying files to {output_dir}...")
    
    for file_type, source_path in default_paths.items():
        if os.path.exists(source_path):
            filename = os.path.basename(source_path)
            dest_path = os.path.join(output_dir, filename)
            
            try:
                shutil.copy2(source_path, dest_path)
                copied_files.append(filename)
                print(f"✓ Copied: {filename}")
            except Exception as e:
                print(f"✗ Failed to copy {filename}: {e}")
                missing_files.append(file_type)
        else:
            print(f"✗ Source file not found: {source_path}")
            missing_files.append(file_type)
    
    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Create README for the release
    readme_content = f"""# Cued Speech Model Files v1.0.0

This release contains the model files required for the cued-speech Python package.

## Files Included

"""
    
    for file_type, filename in manifest.items():
        status = "✓" if file_type not in missing_files else "✗"
        readme_content += f"- {status} `{filename}` - {file_type}\n"
    
    readme_content += f"""
## Installation

These files are automatically downloaded when you install and use the cued-speech package:

```python
from cued_speech.data import setup_model_files

# This will download all required model files
setup_model_files(download_missing=True)
```

## Manual Installation

If you prefer to download manually, you can use the URLs in the manifest.json file.

## File Descriptions

- `acsr_ctc_tiret_2_0.001_128_2.pt` - Trained CTC model for phoneme recognition
- `phonelist.csv` - Phoneme vocabulary mapping
- `lexicon.txt` - French lexicon for language modeling
- `language_model_fr_2.1.bin` - KenLM language model for French
- `french_homophones_updated_.jsonl` - Homophone mappings for French
- `french_ipa.binary` - French IPA language model

## Version

This release corresponds to cued-speech package version 0.1.2.
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\n✓ Preparation complete!")
    print(f"✓ Files copied to: {output_dir}")
    print(f"✓ Manifest saved to: {manifest_path}")
    print(f"✓ README saved to: {readme_path}")
    
    if copied_files:
        print(f"\nCopied files ({len(copied_files)}):")
        for filename in copied_files:
            print(f"  - {filename}")
    
    if missing_files:
        print(f"\nMissing files ({len(missing_files)}):")
        for file_type in missing_files:
            print(f"  - {file_type}")
    
    return copied_files, missing_files

def create_github_release_script():
    """Create a script with GitHub CLI commands to create a release."""
    script_content = """#!/bin/bash
# GitHub Release Script for Cued Speech Models

# Make sure you have GitHub CLI installed and authenticated
# Install with: brew install gh (macOS) or apt install gh (Ubuntu)
# Authenticate with: gh auth login

REPO="bsow/cued-speech-models"
VERSION="v1.0.0"
RELEASE_DIR="model_files_for_release"

echo "Creating GitHub release for $REPO version $VERSION..."

# Create the release
gh release create $VERSION $RELEASE_DIR/*.pt $RELEASE_DIR/*.csv $RELEASE_DIR/*.txt $RELEASE_DIR/*.bin $RELEASE_DIR/*.jsonl $RELEASE_DIR/*.binary \\
    --repo $REPO \\
    --title "Cued Speech Model Files $VERSION" \\
    --notes-file $RELEASE_DIR/README.md

echo "Release created successfully!"
echo "You can now update the GITHUB_RELEASE_VERSION in download_models.py to $VERSION"
"""
    
    with open("create_github_release.sh", 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod("create_github_release.sh", 0o755)
    print("✓ Created create_github_release.sh script")

if __name__ == "__main__":
    print("Cued Speech Model Files Preparation Script")
    print("=" * 50)
    
    # Prepare files
    copied, missing = prepare_model_files_for_release()
    
    # Create GitHub release script
    create_github_release_script()
    
    print("\nNext steps:")
    print("1. Review the files in 'model_files_for_release/' directory")
    print("2. Create a new GitHub repository: bsow/cued-speech-models")
    print("3. Upload the files to the repository")
    print("4. Create a release using the create_github_release.sh script")
    print("5. Update GITHUB_RELEASE_VERSION in download_models.py if needed")
    
    if missing:
        print(f"\n⚠️  Warning: {len(missing)} files are missing. Please ensure all model files are available before creating the release.") 