# Building the Package Without Data Files

This document explains how to build the Cued Speech package without including the large data files, which are instead downloaded automatically by users.

## Overview

The package is designed to separate the code from the data files for several reasons:

1. **Smaller Package Size**: The data files are large (~200MB) and would make the package unnecessarily large
2. **Automatic Updates**: Users can get the latest data files without reinstalling the package
3. **Flexible Distribution**: Data can be updated independently of code releases
4. **GitHub Integration**: Data is hosted on GitHub releases for reliable distribution

## Data Management Strategy

### What's Excluded from the Package

The following files and directories are excluded from the package build:

- `download/` - Contains all model files and data
- `*.zip` - Any zip files
- `*.tar.gz` - Any compressed archives
- `dist/` - Build artifacts
- `build/` - Build artifacts
- `*.egg-info/` - Package metadata

### What's Included in the Package

- Source code (`src/cued_speech/`)
- Tests (`tests/`)
- Documentation (`README.md`, etc.)
- Configuration files (`pyproject.toml`, `MANIFEST.in`)

## Building the Package

### Method 1: Using the Build Script (Recommended)

We provide a convenient build script that handles everything:

```bash
python build_package.py
```

This script will:
1. Clean previous build artifacts
2. Check for the download folder and warn you
3. Build both wheel and source distributions
4. Show you the results

### Method 2: Manual Build

If you prefer to build manually:

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build wheel distribution
python -m build --wheel

# Build source distribution
python -m build --sdist
```

### Method 3: Using pip

```bash
# Build wheel only
pip wheel . --no-deps

# Build source distribution
python setup.py sdist
```

## Configuration Files

### MANIFEST.in

The `MANIFEST.in` file controls what gets included in the source distribution:

```
include README.md
include LICENSE
include pyproject.toml
include GENERATION_IMPLEMENTATION.md
include PROJECT_SUMMARY.md

# Include source code
recursive-include src/cued_speech *.py
recursive-include src/cued_speech/data *.py *.json *.csv *.txt

# Include tests
recursive-include tests *.py

# Exclude unwanted files
exclude .coverage
exclude .gitignore
exclude .gitattributes
exclude pixi.toml
exclude pixi.lock
exclude merci.mp4
recursive-exclude . __pycache__
recursive-exclude . *.py[co]
recursive-exclude . *.so
recursive-exclude . .DS_Store
recursive-exclude output *
recursive-exclude .pixi *
recursive-exclude .pytest_cache *
recursive-exclude htmlcov *
recursive-exclude dist *
recursive-exclude build *
recursive-exclude download *
recursive-exclude . *.zip
recursive-exclude . *.tar.gz
```

### .gitignore

The `.gitignore` file excludes data files from version control:

```
# pixi environments
.pixi/*
!.pixi/config.toml

# Data files (downloaded automatically)
download/
*.zip
*.tar.gz

# Build artifacts
dist/
build/
*.egg-info/

# Python cache
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
.venv/
venv/
env/

# IDE files
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db
```

## Data Distribution Strategy

### GitHub Release

Data files are distributed via GitHub releases:

- **Repository**: `boubacar-sow/CuedSpeechRecognition`
- **Release Tag**: `cuedspeech`
- **Download URL**: `https://github.com/boubacar-sow/CuedSpeechRecognition/releases/download/cuedspeech/download.zip`

### Automatic Download

When users run the package, it automatically:

1. Checks if data files exist locally
2. Downloads the zip file from GitHub if needed
3. Extracts the files to a `download/` directory
4. Uses the extracted files for processing

### Manual Data Management

Users can also manage data files manually:

```bash
# Download data files
cued-speech download-data

# List available files
cued-speech list-data

# Clean up data files
cued-speech cleanup-data --confirm
```

## Package Structure After Build

After building, your package will have this structure:

```
cued_speech-0.1.3/
├── cued_speech/
│   ├── __init__.py
│   ├── cli.py
│   ├── decoder.py
│   ├── generator.py
│   ├── data_manager.py
│   └── data/
│       └── download_models.py
├── tests/
├── README.md
├── LICENSE
├── pyproject.toml
└── MANIFEST.in
```

## Testing the Build

### Test Installation

After building, test the package:

```bash
# Install the built wheel
pip install dist/cued_speech-*.whl

# Test the CLI
cued-speech --help

# Test data download (this will download the data)
cued-speech download-data
```

### Test Data Download

The package should automatically download data when needed:

```bash
# This should trigger data download
cued-speech decode
```

## Troubleshooting

### Build Issues

1. **Download folder in build**: Make sure `download/` is in `.gitignore` and `MANIFEST.in`
2. **Large package size**: Check that data files are excluded
3. **Missing files**: Verify `MANIFEST.in` includes necessary files

### Data Download Issues

1. **Network errors**: Check internet connection
2. **GitHub rate limits**: Wait and retry
3. **Corrupted downloads**: Use `cued-speech cleanup-data` and retry

### Package Installation Issues

1. **Missing dependencies**: Check `pyproject.toml` dependencies
2. **Import errors**: Verify all modules are included in the build
3. **CLI not found**: Check entry point configuration in `pyproject.toml`

## Best Practices

1. **Always test builds**: Install and test the built package
2. **Keep data separate**: Never include large data files in the package
3. **Version data releases**: Update the release tag when data changes
4. **Document changes**: Update this file when build process changes
5. **Automate builds**: Use CI/CD to automate the build process

## Future Improvements

- [ ] Add data file checksums for verification
- [ ] Implement incremental downloads
- [ ] Add data file compression
- [ ] Support multiple data sources
- [ ] Add data file caching 