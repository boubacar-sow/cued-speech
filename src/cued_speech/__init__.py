"""Cued Speech Decoder Package.

A Python package for decoding cued speech videos with real-time subtitle generation.
"""

__version__ = "0.1.95"
__author__ = "Boubacar Sow"
__email__ = "boubasow.pro@gmail.com"

# Suppress protobuf deprecation warnings globally
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# Handle NumPy/PyTorch compatibility issues
def _check_numpy_compatibility():
    """Check and fix NumPy/PyTorch compatibility issues."""
    try:
        import numpy as np
        import torch
        
        # Check if we have NumPy 2.x with PyTorch that expects NumPy 1.x
        if np.__version__.startswith('2.') and torch.__version__:
            print("⚠️  Detected NumPy 2.x with PyTorch - this may cause compatibility issues.")
            print("💡 To fix this, run: pip install 'numpy>=1.24,<2.0'")
            print("   Or upgrade PyTorch to a version that supports NumPy 2.x")
            return False
        return True
    except ImportError:
        return True

# Check compatibility before importing modules
_check_numpy_compatibility()

# Make decoder import optional to avoid dependency issues
try:
    from .decoder import decode_video
    DECODER_AVAILABLE = True
except ImportError as e:
    decode_video = None
    DECODER_AVAILABLE = False
    print(f"Warning: Decoder not available: {e}")
except Exception as e:
    decode_video = None
    DECODER_AVAILABLE = False
    if "NumPy" in str(e) or "numpy" in str(e).lower():
        print("❌ NumPy/PyTorch compatibility issue detected!")
        print("💡 Please run: pip install 'numpy>=1.24,<2.0'")
        print("   Then reinstall: pip install --upgrade cued-speech")
    else:
        print(f"Warning: Decoder not available: {e}")

from .generator import generate_cue
from .data_manager import ensure_data_files, download_and_extract_data, get_default_paths

__all__ = [
    "decode_video", 
    "generate_cue", 
    "DECODER_AVAILABLE",
    "ensure_data_files",
    "download_and_extract_data", 
    "get_default_paths"
]
