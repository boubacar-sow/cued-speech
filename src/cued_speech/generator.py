"""Cued Speech Generation Module.

This module provides functionality for generating cued speech videos from text input.
It follows the exact workflow from the reference file: Whisper transcription + MFA alignment.
"""

import json
import logging
import os
import subprocess
import tempfile
import ssl
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from bisect import bisect_left

# Suppress protobuf deprecation warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import whisper
from moviepy.editor import AudioFileClip, VideoFileClip
from praatio import textgrid as tgio

from .data.cue_mappings import (
    CONSONANTS,
    VOWELS,
    map_syllable_to_cue,
)
from .data_manager import get_data_dir

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MediaPipe FaceMesh and FaceDetection
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# FaceMesh with improved settings for better detection
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.3,  # Reduced from 0.5 for more sensitive detection
    min_tracking_confidence=0.3
)

# FaceDetection as fallback with full-range model
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  # Full-range model for faces at various distances
    min_detection_confidence=0.3
)

# Log MediaPipe configuration
logger.info("🔧 MediaPipe Configuration:")
logger.info(f"   FaceMesh: static_image_mode=False, max_num_faces=1, min_detection_confidence=0.3, min_tracking_confidence=0.3")
logger.info(f"   FaceDetection: model_selection=1 (full-range), min_detection_confidence=0.3")

# Configure SSL context for Whisper downloads
def _configure_ssl_for_whisper():
    """Configure SSL context to handle certificate issues for Whisper downloads."""
    try:
        # Create SSL context that ignores certificate verification
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Monkey patch urllib to use our SSL context
        def urlretrieve_with_ssl(url, filename, reporthook=None, data=None):
            return urllib.request.urlretrieve(url, filename, reporthook, data, context=ssl_context)
        
        urllib.request.urlretrieve = urlretrieve_with_ssl
        
        # Also patch requests to disable SSL verification
        import requests
        original_get = requests.get
        def get_with_ssl_verify(*args, **kwargs):
            kwargs['verify'] = False
            return original_get(*args, **kwargs)
        requests.get = get_with_ssl_verify
        
        # Patch urllib.request.urlopen
        original_urlopen = urllib.request.urlopen
        def urlopen_with_ssl(*args, **kwargs):
            kwargs['context'] = ssl_context
            return original_urlopen(*args, **kwargs)
        urllib.request.urlopen = urlopen_with_ssl
        
        logger.info("SSL context configured for Whisper downloads")
    except Exception as e:
        logger.warning(f"Failed to configure SSL context: {e}")

# Configure SSL on module import
_configure_ssl_for_whisper()


def calculate_face_scale(face_landmarks, reference_face_bbox):
    """
    Calculate the scale factor based on the face bounding box.
    Args:
        face_landmarks: MediaPipe face landmarks.
    Returns:
        float: Scale factor for the hand.
    """
    # Extract all x and y coordinates of the face landmarks
    x_coords = [landmark.x for landmark in face_landmarks.landmark]
    y_coords = [landmark.y for landmark in face_landmarks.landmark]

    # Calculate the width and height of the face bounding box
    face_width = max(x_coords) - min(x_coords)
    face_height = max(y_coords) - min(y_coords)

    # Use the average of width and height as the face size
    face_size = (face_width + face_height) / 2

    # Define a reference face size (empirically determined for a "normal" face size)
    ref_face_width = reference_face_bbox["x_max"] - reference_face_bbox["x_min"]
    ref_face_height = reference_face_bbox["y_max"] - reference_face_bbox["y_min"]
    reference_face_size = (ref_face_width + ref_face_height) / 2

    # Calculate the scale factor
    scale_factor = face_size / reference_face_size
    return scale_factor


# Easing functions for smooth transitions
def linear_easing(t):
    """Linear interpolation (constant speed)."""
    return t

def ease_in_out_cubic(t):
    """Cubic ease-in-out (slow start and end, fast middle)."""
    return t * t * (3.0 - 2.0 * t) if t < 1 else 1

def ease_out_elastic(t):
    """Elastic ease-out (overshoot then settle like a spring)."""
    if t == 0: 
        return 0
    if t == 1: 
        return 1
    return pow(2, -10 * t) * np.sin((t - 0.075) * (2 * np.pi) / 0.3) + 1

def ease_in_out_back(t):
    """Back ease-in-out (slight overshoot at start and end)."""
    c1 = 1.70158
    c2 = c1 * 1.525
    if t < 0.5:
        return (pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2
    else:
        return (pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2

def get_easing_function(easing_name):
    """Get easing function by name."""
    easing_functions = {
        "linear": linear_easing,
        "ease_in_out_cubic": ease_in_out_cubic,
        "ease_out_elastic": ease_out_elastic,
        "ease_in_out_back": ease_in_out_back
    }
    return easing_functions.get(easing_name, ease_in_out_cubic)


class CuedSpeechGenerator:
    """Main class for generating cued speech videos from text."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the cued speech generator."""
        self.config = config or self._get_default_config()
        self.syllable_map = []
        self.current_video_frame = None
        self.current_hand_pos = None
        self.target_hand_pos = None
        self.active_transition = None
        self.last_active_syllable = None
        self.syllable_times = []
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for generation."""
        return {
            "video_path": "download/test_generate.mp4",
            "output_dir": "output/generator",
            "handshapes_dir": "download/handshapes/coordinates",
            "language": "french",
            "reference_face_size": 0.3,  # Normalized reference face size
            "hand_scale_factor": 0.75,
            "mfa_args": ["--beam", "200", "--retry_beam", "400", "--fine_tune"],
            "video_codec": "libx265",
            "audio_codec": "aac",
            # New parameters for enhanced gesture generation
            "easing_function": "ease_in_out_cubic",  # Options: linear, ease_in_out_cubic, ease_out_elastic, ease_in_out_back
            "enable_morphing": True,  # Enable hand shape morphing
            "enable_transparency": True,  # Enable transparency effects during transitions
            "enable_curving": True,  # Enable curved trajectories for specific position pairs
        }
    
    def _validate_paths(self):
        """Ensure required directories and files exist."""
        if not os.path.exists(self.config["video_path"]):
            raise FileNotFoundError(f"Video file not found: {self.config['video_path']}")
        os.makedirs(self.config["output_dir"], exist_ok=True)
    
    def _should_use_curved_trajectory(self, from_pos, to_pos):
        """Determine if trajectory should be curved based on position pairs."""
        if not self.config.get("enable_curving", True):
            return False
            
        # Straight lines for most movements
        if from_pos in [1] or (from_pos == 5 and to_pos == 4):
            return False
        
        # Curved trajectories to avoid obstacles
        curved_pairs = [
            (5, 3),  # Throat to mouth - curve around chin
            (5, 2),  # Throat to cheek - curve around chin
            (4, 2),  # Chin to cheek - curve around mouth corner
        ]
        
        return (from_pos, to_pos) in curved_pairs
    
    def _calculate_curved_trajectory(self, start_pos, end_pos, progress):
        """Calculate subtle curved trajectory to avoid obstacles."""
        try:
            # Small control point offset for subtle curve
            if (start_pos, end_pos) == (5, 3):  # Throat to mouth
                control_point = (
                    (start_pos[0] + end_pos[0]) / 2,
                    (start_pos[1] + end_pos[1]) / 2 - 15  # Slight upward curve
                )
            elif (start_pos, end_pos) == (5, 2):  # Throat to cheek
                control_point = (
                    (start_pos[0] + end_pos[0]) / 2 + 10,  # Slight rightward curve
                    (start_pos[1] + end_pos[1]) / 2 - 10
                )
            elif (start_pos, end_pos) == (4, 2):  # Chin to cheek
                control_point = (
                    (start_pos[0] + end_pos[0]) / 2 + 8,   # Slight rightward curve
                    (start_pos[1] + end_pos[1]) / 2 + 5
                )
            else:
                # Fallback to linear interpolation
                return (
                    int(start_pos[0] + (end_pos[0] - start_pos[0]) * progress),
                    int(start_pos[1] + (end_pos[1] - start_pos[1]) * progress)
                )
            
            # Quadratic Bezier curve for smooth path
            t = progress
            x = (1-t)**2 * start_pos[0] + 2*(1-t)*t * control_point[0] + t**2 * end_pos[0]
            y = (1-t)**2 * start_pos[1] + 2*(1-t)*t * control_point[1] + t**2 * end_pos[1]
            
            return (int(x), int(y))
            
        except Exception as e:
            logger.warning(f"Curved trajectory calculation failed: {e}")
            # Fallback to linear interpolation
            return (
                int(start_pos[0] + (end_pos[0] - start_pos[0]) * progress),
                int(start_pos[1] + (end_pos[1] - start_pos[1]) * progress)
            )
    
    def _morph_hand_shapes(self, shape1, shape2, progress):
        """Gradually blend between two hand shapes."""
        if not self.config.get("enable_morphing", True):
            # If morphing is disabled, return the target shape
            return self._load_hand_image(shape2)
        
        try:
            # Load both hand images
            img1 = self._load_hand_image(shape1)
            img2 = self._load_hand_image(shape2)
            
            # Ensure both images have alpha channel
            if img1.shape[2] == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
            if img2.shape[2] == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
            
            # Resize images to the same size (use the larger size to avoid cropping)
            height1, width1 = img1.shape[:2]
            height2, width2 = img2.shape[:2]
            
            target_height = max(height1, height2)
            target_width = max(width1, width2)
            
            # Resize both images to the target size
            if img1.shape[:2] != (target_height, target_width):
                img1 = cv2.resize(img1, (target_width, target_height))
            if img2.shape[:2] != (target_height, target_width):
                img2 = cv2.resize(img2, (target_width, target_height))
            
            # Blend images based on progress
            alpha = progress
            morphed = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)
            
            return morphed
            
        except Exception as e:
            logger.warning(f"Morphing failed between shapes {shape1} and {shape2}: {e}")
            # Fallback to target shape if morphing fails
            return self._load_hand_image(shape2)
    
    def _apply_transparency_effect(self, hand_image, progress, is_transitioning):
        """Apply transparency effect during transitions."""
        if not self.config.get("enable_transparency", True):
            return hand_image
        
        try:
            if is_transitioning:
                # During transition: fade out current hand
                alpha = 1.0 - progress
                return self._apply_alpha(hand_image, alpha)
            else:
                # Stable position: full opacity
                return self._apply_alpha(hand_image, 1.0)
        except Exception as e:
            logger.warning(f"Transparency effect failed: {e}")
            # Return original image if transparency fails
            return hand_image
    
    def _apply_alpha(self, image, alpha):
        """Apply transparency to image."""
        try:
            if image.shape[2] == 4:  # Already has alpha channel
                image_copy = image.copy()
                image_copy[:, :, 3] = image_copy[:, :, 3] * alpha
                return image_copy
            else:
                # Convert to RGBA and apply alpha
                rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = rgba[:, :, 3] * alpha
                return rgba
        except Exception as e:
            logger.warning(f"Alpha application failed: {e}")
            # Return original image if alpha application fails
            return image
    
    def _get_current_position_code(self):
        """Get the current position code from the last active syllable."""
        if self.last_active_syllable is None:
            return None
        
        target_shape, hand_pos_code = map_syllable_to_cue(self.last_active_syllable['syllable'])
        return hand_pos_code
    
    def _test_mediapipe_face_detection(self, video_path: str) -> None:
        """Test MediaPipe face detection on the video to check detection rate."""
        logger.info("🔍 Testing MediaPipe face detection on video...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"❌ Could not open video: {video_path}")
            return
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"📹 Video info: {frame_count} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        frames_with_face = 0
        frames_processed = 0
        test_frames = min(100, frame_count)  # Test first 100 frames or all frames if less
        
        logger.info(f"🧪 Testing face detection on {test_frames} frames...")
        
        for i in range(test_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = rgb_frame.astype(np.uint8)
            
            # Test FaceMesh detection
            results = face_mesh.process(rgb_frame)
            face_detected = False
            
            if results.multi_face_landmarks:
                face_detected = True
            else:
                # Test fallback FaceDetection
                detection_results = face_detection.process(rgb_frame)
                if detection_results.detections:
                    face_detected = True
            
            if face_detected:
                frames_with_face += 1
            
            frames_processed += 1
            
            # Log progress every 20 frames
            if (i + 1) % 20 == 0:
                current_rate = (frames_with_face / frames_processed) * 100
                logger.info(f"   Progress: {i+1}/{test_frames} frames, detection rate: {current_rate:.1f}%")
        
        cap.release()
        
        # Calculate final detection rate
        detection_rate = (frames_with_face / frames_processed) * 100 if frames_processed > 0 else 0
        
        logger.info(f"📊 MediaPipe Face Detection Test Results:")
        logger.info(f"   Frames tested: {frames_processed}")
        logger.info(f"   Frames with face detected: {frames_with_face}")
        logger.info(f"   Detection rate: {detection_rate:.1f}%")
        
        if detection_rate < 30:
            logger.warning(f"⚠️ Very low face detection rate ({detection_rate:.1f}%) - video may have issues")
        elif detection_rate < 60:
            logger.warning(f"⚠️ Low face detection rate ({detection_rate:.1f}%) - some frames may be skipped")
        elif detection_rate > 90:
            logger.info(f"🎉 Excellent face detection rate ({detection_rate:.1f}%)")
        else:
            logger.info(f"✅ Good face detection rate ({detection_rate:.1f}%)")
        
        logger.info("✅ MediaPipe face detection test completed")

    def generate_cue(
        self,
        text: Optional[str],
        video_path: str,
        output_path: str,
        audio_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate cued speech video from text input using Whisper + MFA workflow."""
        try:
            # Update config with provided paths while preserving existing config
            if not hasattr(self, 'config') or self.config is None:
                self.config = self._get_default_config()
            
            # Ensure mfa_args is preserved
            if "mfa_args" not in self.config:
                self.config["mfa_args"] = ["--beam", "200", "--retry_beam", "400", "--fine_tune"]
            
            # Update config with any new parameters from kwargs
            for key, value in kwargs.items():
                if key in ["easing_function", "enable_morphing", "enable_transparency", "enable_curving"]:
                    self.config[key] = value
            
            self.config["video_path"] = video_path
            self.config["output_dir"] = os.path.dirname(output_path)
            self._validate_paths()
            
            # Test MediaPipe face detection first
            self._test_mediapipe_face_detection(video_path)
            
            # Step 1: Extract or use provided audio
            if audio_path is None:
                audio_path = self._extract_audio()
            
            # Step 2: Get text - either from parameter or from Whisper transcription
            if text is None and not self.config.get("skip_whisper", False):
                logger.info("No text provided, extracting from video using Whisper...")
                transcription = self._transcribe_audio(audio_path)
                logger.info(f"Whisper transcription: {transcription}")
                text = transcription
            elif self.config.get("skip_whisper", False):
                logger.info("Whisper skipped, using provided text for alignment")
                transcription = text  # Use provided text for alignment
            else:
                logger.info(f"Using provided text: '{text}'")
                # Still transcribe for alignment purposes
                transcription = self._transcribe_audio(audio_path)
                logger.info(f"Whisper transcription for alignment: {transcription}")
            
            # Step 3: Use MFA to align the transcription with the audio and get phoneme timing
            logger.info("🎯 Step 3: Starting MFA alignment and syllable building...")
            self._align_and_build_syllables(audio_path, transcription)
            logger.info("✅ Step 3 completed: MFA alignment and syllable building finished")
            
            # Step 4: Render video with hand cues directly to final output
            logger.info("🎬 Step 4: Starting video rendering with hand cues...")
            final_output = self._render_video_with_audio(audio_path)
            logger.info("✅ Step 4 completed: Video rendering finished")
            
            # Clean up temporary directory and all intermediate files
            temp_dir = os.path.join(self.config["output_dir"], "temp")
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary files")
            
            logger.info(f"Cued speech generation complete: {final_output}")
            return final_output
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def _extract_audio(self) -> str:
        """Extract audio from video file."""
        # Use temporary directory for intermediate files
        temp_dir = os.path.join(self.config["output_dir"], "temp")
        os.makedirs(temp_dir, exist_ok=True)
        audio_path = os.path.join(temp_dir, "audio.wav")
        with VideoFileClip(self.config["video_path"]) as video:
            video.audio.write_audiofile(audio_path, codec="pcm_s16le")
        logger.info(f"Audio extracted to temporary location")
        return audio_path
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model("medium", device=device)
            result = model.transcribe(audio_path, language=self.config["language"])
            logger.info("Audio transcription completed")
            return result["text"]
        except Exception as e:
            if "SSL" in str(e) or "certificate" in str(e).lower():
                logger.warning("SSL error during Whisper download, trying alternative approach...")
                return self._transcribe_audio_fallback(audio_path)
            else:
                raise e
    
    def _transcribe_audio_fallback(self, audio_path: str) -> str:
        """Fallback transcription method that handles SSL issues."""
        try:
            # Try with a smaller model that might already be cached
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model("tiny", device=device)
            result = model.transcribe(audio_path, language=self.config["language"])
            logger.info("Audio transcription completed with fallback model")
            return result["text"]
        except Exception as e:
            logger.error(f"Fallback transcription also failed: {e}")
            # Return a placeholder text if all else fails
            return "transcription failed"
    
    def _align_and_build_syllables(self, audio_path: str, text: str) -> None:
        """Align text and build syllable timeline using MFA."""
        logger.info("🔄 Starting MFA alignment and syllable building...")
        
        # Run MFA alignment
        logger.info("📝 Running MFA alignment...")
        textgrid_path = self._run_mfa_alignment(audio_path, text)
        logger.info(f"✅ MFA alignment completed, TextGrid saved to: {textgrid_path}")
        
        # Parse TextGrid
        logger.info("📊 Parsing TextGrid to build syllable map...")
        self.syllable_map = self._parse_textgrid(textgrid_path)
        logger.info(f"✅ TextGrid parsing completed, found {len(self.syllable_map)} syllables")
        
        # Sort by start time using 'a1' key instead of tuple index
        logger.info("🔄 Sorting syllables by start time...")
        self.syllable_map.sort(key=lambda x: x['a1'])
        
        # Create syllable times list using dictionary keys
        logger.info("🔄 Creating syllable times list...")
        self.syllable_times = [item['a1'] for item in self.syllable_map]
        
        # Debug logging
        logger.info(f"📋 Created syllable map with {len(self.syllable_map)} syllables:")
        for i, syl in enumerate(self.syllable_map):
            logger.info(f"  {i}: '{syl['syllable']}' ({syl['type']}) - a1:{syl['a1']:.3f}, a3:{syl['a3']:.3f}, m1:{syl['m1']:.3f}, m2:{syl['m2']:.3f}")
        
        logger.info("✅ Syllable alignment and building completed successfully!")
        print(self.syllable_map)
    
    def _run_mfa_alignment(self, audio_path: str, text: str) -> str:
        """Run Montreal Forced Aligner"""
        # Check if MFA is available - try multiple locations
        mfa_path = self._find_mfa_executable()
        if not mfa_path:
            logger.error("Montreal Forced Aligner (MFA) is not installed or not found in PATH")
            logger.error("")
            logger.error("📋 INSTALLATION INSTRUCTIONS:")
            logger.error("")
            logger.error("1. If you don't have conda/miniconda installed:")
            logger.error("   - Install Miniconda: https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions")
            logger.error("")
            logger.error("2. Install MFA using conda (strongly recommended to avoid _kalpy issues):")
            logger.error("   conda install -c conda-forge montreal-forced-aligner")
            logger.error("")
            logger.error("3. For detailed MFA installation instructions:")
            logger.error("   https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html")
            logger.error("")
            logger.error("4. Alternative: Install with package then conda:")
            logger.error("   pip install cued-speech[mfa]")
            logger.error("   conda install -c conda-forge montreal-forced-aligner")
            logger.error("")
            logger.error("5. If using Pixi environment:")
            logger.error("   - Make sure you're in the pixi shell: pixi shell")
            logger.error("   - Or run with pixi: pixi run cued-speech generate ...")
            logger.error("   - Or activate pixi environment manually")
            logger.error("")
            logger.error("⚠️  Note: Installing MFA via pip may cause _kalpy module errors. Use conda installation.")
            raise RuntimeError("MFA not found. Please install Montreal Forced Aligner first. Use conda installation to avoid _kalpy issues.")
        
        # Create a temporary directory for MFA input
        temp_dir = os.path.join(self.config['output_dir'], "temp")
        mfa_input_dir = os.path.join(temp_dir, "mfa_input")
        os.makedirs(mfa_input_dir, exist_ok=True)
        audio_filename = os.path.basename(audio_path)
        mfa_audio_path = os.path.join(mfa_input_dir, audio_filename)
        os.system(f"cp {audio_path} {mfa_audio_path}")
        text_filename = os.path.splitext(audio_filename)[0] + ".lab"
        text_path = os.path.join(mfa_input_dir, text_filename)
        with open(text_path, "w") as f:
            f.write(text)
        # Build MFA command using the found path
        cmd = [mfa_path, "align", mfa_input_dir, f"{self.config['language']}_mfa",
            f"{self.config['language']}_mfa", temp_dir, "--clean"
        ] + self.config["mfa_args"]

        logger.info(f"Running MFA command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"MFA alignment successful: {result.stdout}")
        return os.path.join(temp_dir, f"{os.path.splitext(audio_filename)[0]}.TextGrid")
    
    def _find_mfa_executable(self) -> Optional[str]:
        """Find the MFA executable in various possible locations."""
        import subprocess
        import os
        from pathlib import Path
        
        # First, try the standard PATH
        try:
            result = subprocess.run(["mfa", "--version"], capture_output=True, text=True, check=True)
            logger.info(f"Found MFA in PATH: {result.stdout.strip()}")
            return "mfa"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Check common pixi environment paths
        pixi_paths = [
            ".pixi/envs/default/bin/mfa",
            ".pixi/envs/dev/bin/mfa", 
            ".pixi/envs/docs/bin/mfa",
            "~/.pixi/envs/default/bin/mfa",
            "~/.pixi/envs/dev/bin/mfa",
            "~/.pixi/envs/docs/bin/mfa"
        ]
        
        for pixi_path in pixi_paths:
            # Expand user path for ~
            expanded_path = os.path.expanduser(pixi_path)
            if os.path.exists(expanded_path):
                logger.info(f"Found MFA in pixi environment: {expanded_path}")
                return expanded_path
        
        # Check conda environments
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_mfa_path = os.path.join(conda_prefix, "bin", "mfa")
            if os.path.exists(conda_mfa_path):
                logger.info(f"Found MFA in conda environment: {conda_mfa_path}")
                return conda_mfa_path
        
        # Check for pixi environment variable
        pixi_env = os.environ.get('PIXI_ENVIRONMENT')
        if pixi_env:
            pixi_env_path = f".pixi/envs/{pixi_env}/bin/mfa"
            if os.path.exists(pixi_env_path):
                logger.info(f"Found MFA in pixi environment {pixi_env}: {pixi_env_path}")
                return pixi_env_path
        
        # Check for pixi in PATH and try to find the environment
        try:
            result = subprocess.run(["pixi", "info"], capture_output=True, text=True)
            if result.returncode == 0:
                # Pixi is available, try to find the environment
                project_root = Path.cwd()
                while project_root != project_root.parent:
                    pixi_env_dir = project_root / ".pixi" / "envs"
                    if pixi_env_dir.exists():
                        for env_dir in pixi_env_dir.iterdir():
                            if env_dir.is_dir():
                                mfa_path = env_dir / "bin" / "mfa"
                                if mfa_path.exists():
                                    logger.info(f"Found MFA in pixi project: {mfa_path}")
                                    return str(mfa_path)
                    project_root = project_root.parent
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Check system-wide conda installations and common paths
        conda_locations = [
            os.path.expanduser("~/miniconda3/bin/mfa"),
            os.path.expanduser("~/anaconda3/bin/mfa"),
            "/opt/conda/bin/mfa",
            "/usr/local/conda/bin/mfa",
            # Add specific paths that might be used
            os.path.expanduser("~/cued_speech/.pixi/envs/default/bin/mfa"),
            os.path.expanduser("~/cued_speech/.pixi/envs/dev/bin/mfa"),
            os.path.expanduser("~/cued_speech/.pixi/envs/docs/bin/mfa")
        ]
        
        for conda_path in conda_locations:
            if os.path.exists(conda_path):
                logger.info(f"Found MFA in system conda: {conda_path}")
                return conda_path
        
        return None
    
    def _parse_textgrid(self, textgrid_path: str) -> List[Dict]:
        """
        Parse TextGrid into syllable timeline using manual syllable construction.
        Args:
            textgrid_path (str): Path to the TextGrid file.
        Returns:
            list: A list of tuples mapping syllables to their intervals [(syllable, start, end)].
        """
        logger.info(f"📊 Parsing TextGrid file: {textgrid_path}")
        
        consonants = "ptkbdgmnlrsfvzʃʒɡʁjwŋtrɥgʀcɲ"
        vowels = "aeɛioɔuøœyəɑ̃ɛ̃ɔ̃œ̃ɑ̃ɔ̃ɑ̃ɔ̃"
        
        logger.info("📖 Opening TextGrid file...")
        tg = tgio.openTextgrid(textgrid_path, includeEmptyIntervals=False)
        phone_tier = tg.getTier("phones")
        logger.info(f"📋 Found {len(phone_tier.entries)} phone entries in TextGrid")
        
        syllables = []
        i = 0
        max_iterations = len(phone_tier.entries) * 2  # Safety limit to prevent infinite loops
        iteration_count = 0
        
        logger.info("🔄 Processing phone entries to build syllables...")
        while i < len(phone_tier.entries) and iteration_count < max_iterations:
            iteration_count += 1
            
            # Log progress every 5 iterations for better debugging
            if iteration_count % 5 == 0:
                logger.info(f"   Processing phone {i}/{len(phone_tier.entries)} (iteration {iteration_count})")
            
            start, end, phone = phone_tier.entries[i]
            phone_str = ''.join(phone) if isinstance(phone, list) else str(phone)
            phone = list(phone)
            
            logger.info(f"   Phone {i}: '{phone_str}' ({start:.3f}-{end:.3f})")
            
            try:
                if len(phone) == 2:
                    if phone[0] in vowels and phone[1] == "̃":
                        syllable = phone[0] + phone[1]
                        syllables.append((syllable, start, end))
                        logger.info(f"     -> Created nasal vowel syllable: '{syllable}'")
                        i += 1
                    else:
                        # Handle other 2-character phones
                        syllable = phone[0]
                        syllables.append((syllable, start, end))
                        logger.info(f"     -> Created single char syllable: '{syllable}'")
                        i += 1
                else:
                    if phone[0] in vowels:
                        syllables.append((phone[0], start, end))
                        logger.info(f"     -> Created vowel syllable: '{phone[0]}'")
                        i += 1
                    elif phone[0] in consonants:
                        if i + 1 < len(phone_tier.entries):
                            next_start, next_end, next_phone = phone_tier.entries[i + 1]
                            next_phone_str = ''.join(next_phone) if isinstance(next_phone, list) else str(next_phone)
                            next_phone = list(next_phone)
                            
                            # Check if we can combine consonant with next vowel
                            can_combine = False
                            if len(next_phone) == 2:
                                if next_phone[0] in vowels and abs(end - next_start) < 0.01 and next_phone[1] == "̃":
                                    syllable = phone[0] + next_phone[0] + next_phone[1]
                                    syllables.append((syllable, start, next_end))
                                    logger.info(f"     -> Created CV syllable (nasal): '{syllable}' from '{phone_str}' + '{next_phone_str}'")
                                    i += 2
                                    can_combine = True
                            else:
                                if next_phone[0] in vowels and abs(end - next_start) < 0.01:
                                    syllable = phone[0] + next_phone[0]
                                    syllables.append((syllable, start, next_end))
                                    logger.info(f"     -> Created CV syllable: '{syllable}' from '{phone_str}' + '{next_phone_str}'")
                                    i += 2
                                    can_combine = True
                            
                            if not can_combine:
                                syllables.append((phone[0], start, end))
                                logger.info(f"     -> Created consonant syllable: '{phone[0]}'")
                                i += 1
                        else:
                            syllables.append((phone[0], start, end))
                            logger.info(f"     -> Created final consonant syllable: '{phone[0]}'")
                            i += 1
                    else:
                        # Handle other characters (spaces, punctuation, etc.)
                        if phone[0] not in [' ', '_', '']:  # Skip empty/space entries
                            syllables.append((phone[0], start, end))
                            logger.info(f"     -> Created other syllable: '{phone[0]}'")
                        else:
                            logger.info(f"     -> Skipping space/empty phone: '{phone[0]}'")
                        i += 1
            except Exception as e:
                logger.warning(f"Error processing phone {i} ('{phone_str}'): {e}")
                i += 1  # Skip this phone and continue
        
        if iteration_count >= max_iterations:
            logger.error(f"⚠️ TextGrid parsing hit safety limit ({max_iterations} iterations). This may indicate an infinite loop.")
            logger.error(f"   Processed {i}/{len(phone_tier.entries)} phones, created {len(syllables)} syllables")
        
        # Check if all phones were processed
        if i < len(phone_tier.entries):
            logger.warning(f"⚠️ Not all phones were processed: {i}/{len(phone_tier.entries)} phones processed")
            logger.warning(f"   Remaining phones: {[phone_tier.entries[j][2] for j in range(i, min(i+5, len(phone_tier.entries)))]}")
        
        logger.info(f"✅ Phone processing completed: {len(syllables)} syllables created from {len(phone_tier.entries)} phones")
        
        # Log all created syllables for verification
        logger.info("📋 Created syllables:")
        for idx, (syllable, start, end) in enumerate(syllables):
            logger.info(f"   {idx}: '{syllable}' ({start:.3f}-{end:.3f})")
        enhanced_syllables = []
        prev_syllable_end = 0
        for i, (syllable, start, end) in enumerate(syllables):
            logger.info(f"{syllable} {start} {end}")
            # Determine syllable type
            if len(syllable) == 1:
                syl_type = 'C' if syllable in consonants else 'V'
            else:
                syl_type = 'CV'
            
            # Calculate A1A3 duration in seconds
            a1a3_duration = end - start
            
            # Determine context
            from_neutral = (i == 0 or (start - prev_syllable_end) > 0.5)  # If pause >500ms
            to_neutral = False  # Implement similar logic for end of utterance
            
            # Calculate M1 and M2 based on WP3 algorithm from auto-cuedspeech.org
            # Determine if this is the first syllable (from_neutral) or last syllable (to_neutral)
            from_neutral = (i == 0)  # First syllable
            to_neutral = (i == len(syllables) - 1)  # Last syllable
            
            if from_neutral:
                m1 = start - (a1a3_duration * 1.60)
                m2 = start - (a1a3_duration * 0.10)
            elif to_neutral:
                m1 = start - 0.03
                m2 = m1 + 0.37
            else:
                if syl_type == 'C':
                    m1 = start - (a1a3_duration * 1.60)
                    m2 = start - (a1a3_duration * 0.30)
                elif syl_type == 'V':
                    m1 = start - (a1a3_duration * 2.40)
                    m2 = start - (a1a3_duration * 0.60)
                else:  # CV
                    m1 = start - (a1a3_duration * 0.80)
                    # Check if this is the second key (2nd syllable)
                    if i == 1:  # Second syllable
                        m2 = start
                    else:
                        m2 = start + (a1a3_duration * 0.11)
            
            enhanced_syllables.append({
                'syllable': syllable,
                'a1': start,
                'a3': end,
                'm1': m1,
                'm2': m2,
                'type': syl_type
            })
            prev_syllable_end = end
        
        logger.info(f"✅ TextGrid parsing completed: {len(enhanced_syllables)} syllables created")
        return enhanced_syllables
    

    
    def _split_ipa_into_syllables(self, ipa_text: str) -> List[str]:
        """Split IPA text into syllables."""
        consonants = "ptkbdgmnlrsfvzʃʒɡʁjwŋtrɥgʀcɲ"
        vowels = "aeɛioɔuøœyəɑ̃ɛ̃ɔ̃œ̃ɑ̃ɔ̃ɑ̃ɔ̃"
        
        syllables = []
        current_syllable = ""
        
        for char in ipa_text:
            if char in vowels:
                # Vowel starts a new syllable or continues current one
                current_syllable += char
            elif char in consonants:
                # Consonant can be part of current syllable or start new one
                if current_syllable and current_syllable[-1] in vowels:
                    # Add consonant to current syllable
                    current_syllable += char
                else:
                    # Start new syllable with consonant
                    if current_syllable:
                        syllables.append(current_syllable)
                    current_syllable = char
            else:
                # Other characters (spaces, etc.)
                if current_syllable:
                    syllables.append(current_syllable)
                    current_syllable = ""
        
        # Add final syllable
        if current_syllable:
            syllables.append(current_syllable)
        
        return syllables if syllables else ["a"]
    
    def _render_video_with_audio(self, audio_path: str) -> str:
        """Render video with hand cues and audio directly to final output."""
        logger.info("🎬 Starting video rendering with audio...")
        
        # Get original video filename
        original_filename = os.path.basename(self.config["video_path"])
        name, ext = os.path.splitext(original_filename)
        
        # Create final output path with original filename
        final_output_path = os.path.join(self.config["output_dir"], f"{name}_cued{ext}")
        logger.info(f"📁 Final output path: {final_output_path}")
        
        # Create temporary path for video without audio
        temp_dir = os.path.join(self.config["output_dir"], "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
        logger.info(f"📁 Temporary video path: {temp_video_path}")
        
        try:
            # Render video with hand cues (without audio)
            logger.info("🎥 Rendering video with hand cues (without audio)...")
            self._render_video_to_path(temp_video_path)
            logger.info("✅ Video rendering completed")
            
            # Add audio to create final video
            logger.info("🔊 Adding audio to create final video...")
            final_output = self._add_audio(temp_video_path, audio_path, final_output_path)
            logger.info("✅ Audio addition completed")
            
            # Clean up temporary file (will be handled by main cleanup)
            pass
            
            return final_output
            
        except Exception as e:
            logger.error(f"💥 Error in video rendering: {e}")
            # Clean up temporary directory on error
            temp_dir = os.path.join(self.config["output_dir"], "temp")
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
            raise e
    
    def _render_video_to_path(self, output_path: str) -> None:
        """Render video with hand cues to specified path."""
        logger.info("🎬 Starting video rendering with hand cues...")
        
        input_video = cv2.VideoCapture(self.config["video_path"])
        frame_info = self._get_video_properties(input_video)
        
        logger.info(f"📹 Video properties: {frame_info['width']}x{frame_info['height']}, {frame_info['fps']:.2f} FPS, {frame_info['frame_count']} frames")
        logger.info(f"⏱️ Total duration: {frame_info['frame_count'] / frame_info['fps']:.2f} seconds")
        
        video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            frame_info["fps"],
            (frame_info["width"], frame_info["height"])
        )

        frames_processed = 0
        frames_with_face = 0
        frames_without_face = 0
        
        for frame_idx in range(int(frame_info["frame_count"])):
            success, frame = input_video.read()
            if not success:
                logger.warning(f"⚠️ Failed to read frame {frame_idx}")
                break
                
            self.current_video_frame = frame
            current_time = frame_idx / frame_info["fps"]
            
            # Log progress every 30 frames (about 1 second at 30fps)
            if frame_idx % 30 == 0:
                progress = (frame_idx / frame_info["frame_count"]) * 100
                logger.info(f"📊 Rendering progress: {progress:.1f}% ({frame_idx}/{frame_info['frame_count']} frames)")
            
            # Process frame and track face detection
            face_detected = self._process_frame_with_tracking(current_time)
            
            if face_detected:
                frames_with_face += 1
            else:
                frames_without_face += 1
            
            frames_processed += 1
            video_writer.write(frame)

        input_video.release()
        video_writer.release()
        
        logger.info(f"✅ Video rendering complete!")
        logger.info(f"📊 Final stats: {frames_processed} frames processed")
        logger.info(f"👤 Frames with face detected: {frames_with_face}")
        logger.info(f"🚫 Frames without face: {frames_without_face}")
        
        # Calculate detection rate
        if frames_processed > 0:
            detection_rate = (frames_with_face / frames_processed) * 100
            logger.info(f"📈 Face detection rate: {detection_rate:.1f}%")
            
            if detection_rate < 50:
                logger.warning(f"⚠️ Low face detection rate ({detection_rate:.1f}%) - consider checking video quality or lighting")
            elif detection_rate > 90:
                logger.info(f"🎉 Excellent face detection rate ({detection_rate:.1f}%)")
            else:
                logger.info(f"✅ Good face detection rate ({detection_rate:.1f}%)")
        
        logger.info(f"📁 Output saved to: {output_path}")
    
    def _render_video(self) -> str:
        """Render video with hand cues (legacy method for backward compatibility)."""
        temp_path = os.path.join(self.config["output_dir"], "temp_rendered_video.mp4")
        self._render_video_to_path(temp_path)
        return temp_path
    
    def _process_frame_with_tracking(self, current_time: float) -> bool:
        """Process a single frame and add hand cues, returning whether face was detected."""
        try:
            # Ensure frame is valid
            if self.current_video_frame is None:
                logger.warning("No video frame available")
                return False
                
            # Convert to RGB and ensure proper data type
            rgb_frame = cv2.cvtColor(self.current_video_frame, cv2.COLOR_BGR2RGB)
            rgb_frame = rgb_frame.astype(np.uint8)
            
            # DEBUG: Log frame processing start
            logger.info(f"🔍 Processing frame at time {current_time:.3f}s - Frame shape: {rgb_frame.shape}")
            
            # Process with MediaPipe FaceMesh first
            results = face_mesh.process(rgb_frame)
            face_landmarks = None
            
            # Primary detection: FaceMesh
            if results.multi_face_landmarks:
                try:
                    face_landmarks = results.multi_face_landmarks[0]
                    logger.info(f"✅ FaceMesh detection successful at time {current_time:.3f}s - Found {len(results.multi_face_landmarks)} face(s)")
                except Exception as e:
                    logger.warning(f"❌ Error accessing FaceMesh landmarks at time {current_time:.3f}s: {e}")
                    face_landmarks = None
            else:
                logger.warning(f"⚠️ FaceMesh found no faces at time {current_time:.3f}s")
            
            # Fallback detection: FaceDetection + FaceMesh static mode
            if face_landmarks is None:
                logger.info(f"🔄 FaceMesh failed at time {current_time:.3f}s, trying fallback detection...")
                try:
                    # First, try FaceDetection to confirm a face exists
                    detection_results = face_detection.process(rgb_frame)
                    if detection_results.detections:
                        logger.info(f"✅ FaceDetection confirmed face presence at time {current_time:.3f}s - Found {len(detection_results.detections)} detection(s)")
                        
                        # Try FaceMesh in static mode (more robust for per-frame detection)
                        static_face_mesh = mp_face_mesh.FaceMesh(
                            static_image_mode=True,  # Static mode for better per-frame detection
                            max_num_faces=1,
                            min_detection_confidence=0.3,
                            min_tracking_confidence=0.3
                        )
                        static_results = static_face_mesh.process(rgb_frame)
                        
                        if static_results.multi_face_landmarks:
                            face_landmarks = static_results.multi_face_landmarks[0]
                            logger.info(f"✅ Fallback FaceMesh detection successful at time {current_time:.3f}s")
                        else:
                            logger.warning(f"❌ Fallback FaceMesh also failed at time {current_time:.3f}s")
                    else:
                        logger.warning(f"❌ FaceDetection found no faces at time {current_time:.3f}s")
                        
                except Exception as e:
                    logger.error(f"💥 Error in fallback detection at time {current_time:.3f}s: {e}")
            
            # If still no face detected, skip this frame
            if face_landmarks is None:
                logger.warning(f"🚫 No face detected in frame at time {current_time:.3f}s - SKIPPING FRAME")
                return False
            else:
                logger.info(f"🎯 Face landmarks successfully obtained at time {current_time:.3f}s")
                
        except Exception as e:
            logger.error(f"Error processing frame with MediaPipe: {e}")
            return False
        
        # Find active transition for the current time
        self.active_transition = None
        for syl in self.syllable_map:
            if syl['m1'] <= current_time <= syl['m2']:
                self.active_transition = syl
                break
        
        # Debug logging for syllable mapping
        if self.active_transition:
            current_syllable = self.active_transition['syllable']
            hand_shape, hand_pos = map_syllable_to_cue(current_syllable)
            logger.info(f"syllable: {current_syllable} mapped to hand_shape: {hand_shape} hand_position: {hand_pos}")
        else:
            # Log when no syllable is found for current time
            logger.debug(f"No active syllable found at time {current_time:.3f}s")
            # Check if we're within any syllable's a1-a3 window as fallback
            for syl in self.syllable_map:
                if syl['a1'] <= current_time <= syl['a3']:
                    logger.debug(f"Found syllable '{syl['syllable']}' in a1-a3 window at time {current_time:.3f}s")
                    break
        
        if self.active_transition:
            progress = (current_time - self.active_transition['m1']) / (self.active_transition['m2'] - self.active_transition['m1'])
            self._render_hand_transition(face_landmarks, progress)
        else:
            # If no new gesture is active, persist the last hand position 
            # as long as the last syllable is not the final syllable of the sentence.
            if self.current_hand_pos is not None and \
            self.last_active_syllable is not None and \
            self.last_active_syllable != self.syllable_map[-1]:
                hand_shape, hand_pos_code = map_syllable_to_cue(self.last_active_syllable['syllable'])
                hand_image = self._load_hand_image(hand_shape)
                scale_factor = calculate_face_scale(face_landmarks, {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1})
                self.current_video_frame = self._overlay_hand_image(
                    self.current_video_frame,
                    hand_image,
                    self.current_hand_pos[0],
                    self.current_hand_pos[1],
                    scale_factor,
                    hand_shape
                )
        
        return True  # Face was successfully detected and processed

    def _process_frame(self, current_time: float) -> None:
        """Process a single frame and add hand cues (legacy method)."""
        # Use the new tracking method but ignore the return value
        self._process_frame_with_tracking(current_time)
    
    def _render_hand_transition(self, face_landmarks, progress: float) -> None:
        """Render hand gesture transition with enhanced features."""
        progress = max(0.0, min(1.0, progress))
        target_shape, hand_pos_code = map_syllable_to_cue(self.active_transition['syllable'])
        final_target = self._get_target_position(face_landmarks, hand_pos_code)
        
        if self.current_hand_pos is None:
            self.current_hand_pos = final_target
            self.last_active_syllable = self.active_transition
            return
        
        # Get current position code for trajectory determination
        current_pos_code = self._get_current_position_code()
        
        # Apply easing function
        easing_func = get_easing_function(self.config.get("easing_function", "ease_in_out_cubic"))
        eased_progress = easing_func(progress)
        
        # Calculate trajectory (curved or linear)
        if current_pos_code is not None and self._should_use_curved_trajectory(current_pos_code, hand_pos_code):
            intermediate_pos = self._calculate_curved_trajectory(
                self.current_hand_pos, final_target, eased_progress
            )
        else:
            # Linear interpolation
            new_x = self.current_hand_pos[0] + (final_target[0] - self.current_hand_pos[0]) * eased_progress
            new_y = self.current_hand_pos[1] + (final_target[1] - self.current_hand_pos[1]) * eased_progress
            intermediate_pos = (int(new_x), int(new_y))
        
        # Update current position
        if progress < 0.95:
            self.current_hand_pos = intermediate_pos
        else:
            self.current_hand_pos = final_target
        
        # Handle hand shape morphing
        if self.last_active_syllable is not None:
            last_shape, _ = map_syllable_to_cue(self.last_active_syllable['syllable'])
            if last_shape != target_shape:
                # Morph between shapes
                hand_image = self._morph_hand_shapes(last_shape, target_shape, eased_progress)
            else:
                hand_image = self._load_hand_image(target_shape)
        else:
            hand_image = self._load_hand_image(target_shape)
        
        # Apply transparency effect
        is_transitioning = progress < 0.95
        hand_image = self._apply_transparency_effect(hand_image, eased_progress, is_transitioning)
        
        # Render to frame
        scale_factor = calculate_face_scale(face_landmarks, {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1})
        self.current_video_frame = self._overlay_hand_image(
            self.current_video_frame,
            hand_image,
            intermediate_pos[0],
            intermediate_pos[1],
            scale_factor,
            target_shape
        )
        
        # Store the syllable that is currently active
        self.last_active_syllable = self.active_transition
    
    def _get_current_syllable(self, current_time: float) -> Optional[str]:
        """Find current syllable based on timing."""
        # Find the syllable that is currently active (within m1-m2 window)
        for syllable in self.syllable_map:
            if syllable['m1'] <= current_time <= syllable['m2']:
                return syllable['syllable']
        
        # If no syllable is in transition window, find the closest one
        for syllable in self.syllable_map:
            if syllable['a1'] <= current_time <= syllable['a3']:
                return syllable['syllable']
        
        return None
    
    def _load_hand_image(self, hand_shape: int) -> np.ndarray:
        """
        Load the preprocessed hand image for the specified hand shape.
        Args:
            hand_shape (int): Hand shape number (1 to 8).
        Returns:
            np.ndarray: Loaded hand image with transparency (RGBA).
        """
        # Try to load from download directory first
        data_dir = get_data_dir()
        hand_image_path = os.path.join(
            data_dir, "rotated_images", f"rotated_handshape_{hand_shape}.png"
        )
        
        # Fallback to hardcoded path if not found in download directory
        if not os.path.exists(hand_image_path):
            hand_image_path = os.path.join(
                "download/rotated_images",
                f"rotated_handshape_{hand_shape}.png"
            )
        
        if not os.path.exists(hand_image_path):
            raise FileNotFoundError(f"Hand image {hand_shape} not found: {hand_image_path}")
        return cv2.imread(hand_image_path, cv2.IMREAD_UNCHANGED)
    
    def _overlay_hand_image(self, frame: np.ndarray, hand_image: np.ndarray, 
                           target_x: int, target_y: int, scale_factor: float, 
                           hand_shape: int) -> np.ndarray:
        """
        Overlay the hand image on the current frame at the specified position and scale.
        Args:
            frame: Current video frame.
            hand_image: Preprocessed hand image with transparency.
            target_x, target_y: Target position for the reference finger.
            scale_factor: Scale factor for the hand image.
            hand_shape (int): The hand shape number (1 to 8).
        Returns:
            np.ndarray: Updated video frame with the hand image overlaid.
        """
        h, w = hand_image.shape[:2]
        scaled_width = int(w * scale_factor * self.config["hand_scale_factor"])
        scaled_height = int(h * scale_factor * self.config["hand_scale_factor"])
        resized_hand = cv2.resize(hand_image, (scaled_width, scaled_height))

        # Try to load from download directory first
        data_dir = get_data_dir()
        csv_path = os.path.join(data_dir, "yellow_pixels.csv")
        
        # Fallback to hardcoded path if not found in download directory
        if not os.path.exists(csv_path):
            csv_path = "download/yellow_pixels.csv"
        ref_finger_data = pd.read_csv(csv_path)
        hand_row = ref_finger_data[ref_finger_data["image_name"] == f"handshape_{hand_shape}.png"]
        if hand_row.empty:
            raise ValueError(f"No reference finger data found for hand shape {hand_shape}")

        ref_finger_x = hand_row["yellow_pixel_x"].values[0]
        ref_finger_y = hand_row["yellow_pixel_y"].values[0]
        ref_finger_x_scaled = ref_finger_x * scale_factor * self.config["hand_scale_factor"]
        ref_finger_y_scaled = ref_finger_y * scale_factor * self.config["hand_scale_factor"]

        x_offset = int(target_x - ref_finger_x_scaled)
        y_offset = int(target_y - ref_finger_y_scaled)

        # Ensure the hand image stays within the frame boundaries
        x_offset = max(0, min(x_offset, frame.shape[1] - scaled_width))
        y_offset = max(0, min(y_offset, frame.shape[0] - scaled_height))

        if resized_hand.shape[2] == 4:  # Check if the image has an alpha channel
            alpha_hand = resized_hand[:, :, 3] / 255.0
            alpha_frame = 1.0 - alpha_hand
            for c in range(3):
                frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width, c] = (
                    alpha_hand * resized_hand[:, :, c] +
                    alpha_frame * frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width, c]
                )
        else:
            frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = resized_hand
        return frame
    
    def _get_target_position(self, face_landmarks, hand_pos: Union[int, str]) -> Tuple[int, int]:
        """
        Get the target position for the reference landmark on the face.
        Args:
            face_landmarks: MediaPipe face landmarks.
            hand_pos (int): Target position index or special case (-1, -2).
        Returns:
            tuple: (target_x, target_y) in pixel coordinates.
        """
        frame_height, frame_width = self.current_video_frame.shape[:2]
        
        if hand_pos == -1:
            # Position 1: Right side of mouth - use relative distance based on nose landmarks
            # Calculate the distance between nose landmarks 50 and 4 for relative positioning
            nose_landmark_50 = face_landmarks.landmark[50]  # Nose tip
            nose_landmark_4 = face_landmarks.landmark[4]    # Nose bridge
            
            # Calculate the distance between nose landmarks
            nose_distance_x = abs(nose_landmark_50.x - nose_landmark_4.x) * frame_width
            nose_distance_y = abs(nose_landmark_50.y - nose_landmark_4.y) * frame_height
            
            # Use this distance to position the hand relative to the mouth corner (landmark 57)
            mouth_corner = face_landmarks.landmark[57]
            target_x = mouth_corner.x * frame_width - nose_distance_x
            target_y = mouth_corner.y * frame_height
            
        elif hand_pos == -2:
            # Position 5: Throat/below chin - use relative distance based on nose landmarks
            # Calculate the distance between nose landmarks 50 and 4 for relative positioning
            nose_landmark_50 = face_landmarks.landmark[50]  # Nose tip
            nose_landmark_4 = face_landmarks.landmark[4]    # Nose bridge
            
            # Calculate the distance between nose landmarks
            nose_distance_x = abs(nose_landmark_50.x - nose_landmark_4.x) * frame_width
            nose_distance_y = abs(nose_landmark_50.y - nose_landmark_4.y) * frame_height
            
            # Use this distance to position the hand relative to the chin (landmark 152)
            chin = face_landmarks.landmark[152]
            target_x = chin.x * frame_width
            target_y = chin.y * frame_height + nose_distance_y
            
        else:
            # Direct landmark positioning for other positions
            target_x = face_landmarks.landmark[hand_pos].x * frame_width
            target_y = face_landmarks.landmark[hand_pos].y * frame_height
            
        return int(target_x), int(target_y)
    
    def _get_video_properties(self, cap) -> Dict:
        """Get essential video properties."""
        return {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
    
    def _add_audio(self, video_path: str, audio_path: str, output_path: str = None) -> str:
        """
        Add the original audio to the rendered video with robust error handling.
        """
        if output_path is None:
            output_path = os.path.join(self.config["output_dir"], f"final_{os.path.basename(video_path)}")
        
        logger.info(f"🔊 Adding audio to video...")
        logger.info(f"   Video path: {video_path}")
        logger.info(f"   Audio path: {audio_path}")
        logger.info(f"   Output path: {output_path}")
        
        try:
            logger.info("📹 Loading video and audio clips...")
            with VideoFileClip(video_path) as video_clip, AudioFileClip(audio_path) as audio_clip:
                logger.info(f"   Video duration: {video_clip.duration:.2f}s")
                logger.info(f"   Audio duration: {audio_clip.duration:.2f}s")
                
                if abs(video_clip.duration - audio_clip.duration) > 0.1:
                    logger.info("⚠️ Duration mismatch detected, trimming audio to match video")
                    audio_clip = audio_clip.subclip(0, video_clip.duration)
                
                logger.info("🎬 Combining video and audio...")
                final_clip = video_clip.set_audio(audio_clip)
                
                logger.info("💾 Writing final video file...")
                final_clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    preset="fast",
                    ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                    threads=4,
                    logger=None
                )
                
                if not os.path.exists(output_path):
                    raise RuntimeError(f"Output file not created: {output_path}")
                
                logger.info(f"✅ Final video created successfully: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"💥 Error adding audio: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            raise RuntimeError(f"Failed to add audio: {str(e)}") from e


# Public API function for backward compatibility
def generate_cue(
    text: Optional[str],
    video_path: str,
    output_path: str,
    audio_path: Optional[str] = None,
    config: Optional[Dict] = None,
    **kwargs
) -> str:
    """
    Generate cued speech video from text input or extract text from video using Whisper.
    
    This is a convenience function that creates a CuedSpeechGenerator instance.
    For more control, use CuedSpeechGenerator directly.
    """
    generator = CuedSpeechGenerator(config)
    return generator.generate_cue(text, video_path, output_path, audio_path, **kwargs)
