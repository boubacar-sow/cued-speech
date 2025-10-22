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
import time
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


# calculate_face_scale function removed - now using face width for dynamic scaling


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
        # Always start from defaults, then selectively update with user-provided config
        defaults = self._get_default_config()
        self.config = dict(defaults)
        if config:
            for key, value in config.items():
                if key in defaults:
                    self.config[key] = value
                else:
                    logger.warning(f"Unknown config key '{key}' ignored. Valid keys: {list(defaults.keys())}")
        self.syllable_map = []
        self.current_video_frame = None
        self.current_hand_pos = None
        self.target_hand_pos = None
        self.active_transition = None
        self.last_active_syllable = None
        self.syllable_times = []
        self._timings: Dict[str, float] = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for generation."""
        return {
            "video_path": "download/test_generate.mp4",
            "output_dir": "output/generator",
            "handshapes_dir": "download/handshapes/coordinates",
            "language": "french",
            "mfa_args": ["--beam", "200", "--retry_beam", "400", "--fine_tune"],
            "video_codec": "libx265",
            "audio_codec": "aac",
            # New parameters for enhanced gesture generation
            "easing_function": "linear",  # Options: linear, ease_in_out_cubic, ease_out_elastic, ease_in_out_back
            "enable_morphing": False,  # Enable hand shape morphing
            "enable_transparency": False,  # Enable transparency effects during transitions
            "enable_curving": True,  # Enable curved trajectories for specific position pairs
            "enable_debug_landmarks": False,  # Enable debug visualization of face landmarks
            # Control whether to skip Whisper when text is provided
            "skip_whisper": False,
            "model": None,
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
    ) -> str:
        """Generate cued speech video from text input using Whisper + MFA workflow."""
        try:
            # Initialize config from defaults, then merge existing config
            defaults = self._get_default_config()
            base_config = self.config if hasattr(self, 'config') and self.config is not None else {}
            self.config = {**defaults, **base_config}
            
            self.config["video_path"] = video_path
            self.config["output_dir"] = os.path.dirname(output_path)
            self._validate_paths()
            
            # Test MediaPipe face detection first
            t0 = time.perf_counter()
            self._test_mediapipe_face_detection(video_path)
            self._timings["face_test"] = time.perf_counter() - t0
            
            # Step 1: Extract or use provided audio
            if audio_path is None:
                t0 = time.perf_counter()
                audio_path = self._extract_audio()
                self._timings["extract_audio"] = time.perf_counter() - t0
            
            # Step 2: Get text - either from parameter or from Whisper transcription
            if text is None and not self.config.get("skip_whisper", False):
                logger.info("No text provided, extracting from video using Whisper...")
                t0 = time.perf_counter()
                transcription = self._transcribe_audio(audio_path)
                self._timings["whisper"] = time.perf_counter() - t0
                logger.info(f"Whisper transcription: {transcription}")
                text = transcription
            elif self.config.get("skip_whisper", False) and text is not None:
                logger.info("Whisper skipped, using provided text for alignment")
                transcription = text  # Use provided text for alignment
            else:
                logger.info(f"Using provided text: '{text}'")
                # Skip extra Whisper run when text is provided
                transcription = text
                self._timings["whisper"] = 0.0
            
            # Step 3: Use MFA to align the transcription with the audio and get phoneme timing
            logger.info("🎯 Step 3: Starting MFA alignment and syllable building...")
            t0 = time.perf_counter()
            self._align_and_build_syllables(audio_path, transcription)
            self._timings["mfa_total"] = time.perf_counter() - t0
            logger.info("✅ Step 3 completed: MFA alignment and syllable building finished")
            
            # Step 4: Render video with hand cues directly to final output
            logger.info("🎬 Step 4: Starting video rendering with hand cues...")
            t0 = time.perf_counter()
            final_output = self._render_video_with_audio(audio_path)
            self._timings["render_total"] = time.perf_counter() - t0
            logger.info("✅ Step 4 completed: Video rendering finished")
            
            # Clean up temporary directory and all intermediate files
            temp_dir = os.path.join(self.config["output_dir"], "temp")
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary files")
            
            # Log timing summary
            total_time = sum(self._timings.values()) if self._timings else 0.0
            logger.info("⏱️ Timing summary (seconds):")
            for k in [
                "face_test",
                "extract_audio",
                "whisper",
                "mfa_align",
                "textgrid_parse",
                "syllable_build",
                "mfa_total",
                "render_video",
                "add_audio",
                "render_total",
            ]:
                if k in self._timings:
                    logger.info(f"   {k}: {self._timings[k]:.3f}")
            logger.info(f"   total_recorded: {total_time:.3f}")

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
            t0 = time.perf_counter()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Use provided model object if supplied; else default to "medium"
            model_obj = self.config.get("model")
            if model_obj is None:
                logger.info("Model was not provided, downloading whisper model")
                model = whisper.load_model("medium", device=device)
            else:
                logger.info("Using provided whisper model")
                model = model_obj
            result = model.transcribe(audio_path, language=self.config["language"])
            logger.info("Audio transcription completed")
            # Note: high-level whisper timing is recorded in caller
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
            model_obj = self.config.get("model")
            if model_obj is None:
                model = whisper.load_model("tiny", device=device)
            else:
                model = model_obj
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
        t_align = time.perf_counter()
        textgrid_path = self._run_mfa_alignment(audio_path, text)
        self._timings["mfa_align"] = time.perf_counter() - t_align
        logger.info(f"✅ MFA alignment completed, TextGrid saved to: {textgrid_path}")
        
        # Parse TextGrid
        logger.info("📊 Parsing TextGrid to build syllable map...")
        t_parse = time.perf_counter()
        self.syllable_map = self._parse_textgrid(textgrid_path)
        self._timings["textgrid_parse"] = time.perf_counter() - t_parse
        logger.info(f"✅ TextGrid parsing completed, found {len(self.syllable_map)} syllables")
        
        # Sort by start time using 'a1' key instead of tuple index
        logger.info("🔄 Sorting syllables by start time...")
        t_build = time.perf_counter()
        self.syllable_map.sort(key=lambda x: x['a1'])
        
        # Create syllable times list using dictionary keys
        logger.info("🔄 Creating syllable times list...")
        self.syllable_times = [item['a1'] for item in self.syllable_map]
        self._timings["syllable_build"] = time.perf_counter() - t_build
        
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
            #logger.info(f"🔍 Processing frame at time {current_time:.3f}s - Frame shape: {rgb_frame.shape}")
            
            # Process with MediaPipe FaceMesh first
            results = face_mesh.process(rgb_frame)
            face_landmarks = None
            
            # Primary detection: FaceMesh
            if results.multi_face_landmarks:
                try:
                    face_landmarks = results.multi_face_landmarks[0]
                    #logger.info(f"✅ FaceMesh detection successful at time {current_time:.3f}s - Found {len(results.multi_face_landmarks)} face(s)")
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
            #if face_landmarks is None:
            #    logger.warning(f"🚫 No face detected in frame at time {current_time:.3f}s - SKIPPING FRAME")
            #    return False
            #else:
            #    logger.info(f"🎯 Face landmarks successfully obtained at time {current_time:.3f}s")
                
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
            #logger.info(f"syllable: {current_syllable} mapped to hand_shape: {hand_shape} hand_position: {hand_pos}")
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
                # scale_factor no longer needed - using face width for dynamic scaling
                self.current_video_frame = self._overlay_hand_image(
                    self.current_video_frame,
                    hand_image,
                    self.current_hand_pos[0],
                    self.current_hand_pos[1],
                    hand_shape,
                    face_landmarks
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
        # scale_factor no longer needed - using face width for dynamic scaling
        self.current_video_frame = self._overlay_hand_image(
            self.current_video_frame,
            hand_image,
            intermediate_pos[0],
            intermediate_pos[1],
            target_shape,
            face_landmarks
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
                           target_x: int, target_y: int, hand_shape: int, 
                           face_landmarks) -> np.ndarray:
        """
        Overlay the hand image on the current frame at the specified position and scale.
        Args:
            frame: Current video frame.
            hand_image: Preprocessed hand image with transparency.
            target_x, target_y: Target position for the reference finger.
            hand_shape (int): The hand shape number (1 to 8).
            face_landmarks: MediaPipe face landmarks for scaling calculations.
        Returns:
            np.ndarray: Updated video frame with the hand image overlaid.
        """
        # Get original hand image dimensions
        original_h, original_w = hand_image.shape[:2]
        
        # Calculate face width for dynamic scaling
        # Use landmarks 234 (left cheek) and 454 (right cheek) to get face width
        left_cheek = face_landmarks.landmark[234]
        right_cheek = face_landmarks.landmark[454]
        face_width = abs(right_cheek.x - left_cheek.x) * frame.shape[1]
        
        # Detect reference finger in original hand image to get the distance from landmark 0
        ref_finger_x_orig, ref_finger_y_orig = self._detect_reference_finger(hand_image, hand_shape)
        
        # Get hand landmark 0 (wrist) position in original image
        # We need to detect this using MediaPipe on the original hand image
        hand_landmark_0_x, hand_landmark_0_y = self._get_hand_landmark_0(hand_image)
        
        # Calculate the distance from landmark 0 to reference finger in original image
        hand_span_orig = ((ref_finger_x_orig - hand_landmark_0_x)**2 + (ref_finger_y_orig - hand_landmark_0_y)**2)**0.5
        
        # Calculate scale factor to make hand span equal to face width
        scale_factor_hand = face_width / hand_span_orig if hand_span_orig > 0 else 1.0
        
        # Resize hand image with calculated scale
        original_h, original_w = hand_image.shape[:2]
        scaled_width = int(original_w * scale_factor_hand)
        scaled_height = int(original_h * scale_factor_hand)
        resized_hand = cv2.resize(hand_image, (scaled_width, scaled_height))

        logger.info(f"Face width: {face_width:.1f}px, Hand span: {hand_span_orig:.1f}px, Scale factor: {scale_factor_hand:.3f}")
        logger.info(f"Resized hand to {scaled_width}x{scaled_height}")
        
        # Detect reference finger in the RESIZED image using MediaPipe
        # This gives us the exact position in the resized image coordinate system
        ref_finger_x_scaled, ref_finger_y_scaled = self._detect_reference_finger(resized_hand, hand_shape)
        
        # Debug logging
        logger.info(f"Hand shape {hand_shape}: target=({target_x}, {target_y}), ref_finger=({ref_finger_x_scaled}, {ref_finger_y_scaled}), resized_size=({scaled_width}x{scaled_height})")

        # Since both hand and video have the same resolution, positioning is simple:
        # Place the hand so the reference finger aligns with the target
        x_offset = int(target_x - ref_finger_x_scaled)
        y_offset = int(target_y - ref_finger_y_scaled)

        logger.info(f"Simple positioning: offset=({x_offset}, {y_offset}), target=({target_x}, {target_y}), ref_finger=({ref_finger_x_scaled}, {ref_finger_y_scaled})")
        
        # Handle negative offsets by cropping the hand image appropriately
        if x_offset < 0:
            # Crop from left side of hand image
            crop_x = -x_offset
            resized_hand = resized_hand[:, crop_x:]
            x_offset = 0
            ref_finger_x_scaled = ref_finger_x_scaled - crop_x
            logger.info(f"Cropped left side: crop_x={crop_x}, new ref_finger_x={ref_finger_x_scaled}")
            
        if y_offset < 0:
            # Crop from top side of hand image
            crop_y = -y_offset
            resized_hand = resized_hand[crop_y:, :]
            y_offset = 0
            ref_finger_y_scaled = ref_finger_y_scaled - crop_y
            logger.info(f"Cropped top side: crop_y={crop_y}, new ref_finger_y={ref_finger_y_scaled}")
            
        # Handle right/bottom overflow by cropping
        if x_offset + resized_hand.shape[1] > frame.shape[1]:
            crop_width = frame.shape[1] - x_offset
            resized_hand = resized_hand[:, :crop_width]
            logger.info(f"Cropped right side: new width={crop_width}")
            
        if y_offset + resized_hand.shape[0] > frame.shape[0]:
            crop_height = frame.shape[0] - y_offset
            resized_hand = resized_hand[:crop_height, :]
            logger.info(f"Cropped bottom side: new height={crop_height}")

        # Get the actual dimensions of the cropped hand image
        hand_h, hand_w = resized_hand.shape[:2]
        
        # Ensure we don't go beyond frame boundaries
        end_x = min(x_offset + hand_w, frame.shape[1])
        end_y = min(y_offset + hand_h, frame.shape[0])
        
        # Calculate the actual region to overlay
        actual_w = end_x - x_offset
        actual_h = end_y - y_offset
        
        if actual_w <= 0 or actual_h <= 0:
            logger.warning(f"Hand image completely outside frame boundaries")
            return frame
            
        # Crop the hand image to fit the actual overlay region
        hand_cropped = resized_hand[:actual_h, :actual_w]

        if resized_hand.shape[2] == 4:  # Check if the image has an alpha channel
            alpha_hand = hand_cropped[:, :, 3] / 255.0
            alpha_frame = 1.0 - alpha_hand
            for c in range(3):
                frame[y_offset:end_y, x_offset:end_x, c] = (
                    alpha_hand * hand_cropped[:, :, c] +
                    alpha_frame * frame[y_offset:end_y, x_offset:end_x, c]
                )
        else:
            frame[y_offset:end_y, x_offset:end_x] = hand_cropped
        
        # Add debug visualization for hand reference finger position
        if self.config.get("enable_debug_landmarks", True):
            # Draw where we intended to place the reference finger (target position)
            cv2.circle(frame, (target_x, target_y), 8, (0, 255, 255), -1)  # Cyan dot for target
            cv2.circle(frame, (target_x, target_y), 12, (0, 0, 0), 2)  # Black border
            cv2.putText(frame, "TARGET", (target_x + 15, target_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw where the reference finger actually ended up after applying the offset
            actual_ref_x = int(x_offset + ref_finger_x_scaled)
            actual_ref_y = int(y_offset + ref_finger_y_scaled)
            cv2.circle(frame, (actual_ref_x, actual_ref_y), 8, (255, 0, 255), -1)  # Magenta dot for actual
            cv2.circle(frame, (actual_ref_x, actual_ref_y), 12, (255, 255, 255), 2)  # White border
            cv2.putText(frame, "REF", (actual_ref_x + 15, actual_ref_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Add debug landmarks visualization
        frame = self._add_debug_landmarks(frame)
        
        return frame
    
    def _detect_reference_finger(self, hand_image: np.ndarray, hand_shape: int) -> Tuple[int, int]:
        """
        Detect the reference finger tip in the hand image using MediaPipe.
        - Handshapes 1 and 6: Use index finger (landmark 8)
        - Other handshapes: Use middle finger (landmark 12)
        
        Args:
            hand_image: Hand image (already resized) with RGBA channels
            hand_shape: Hand shape number (1-8)
            
        Returns:
            Tuple of (x, y) coordinates of the reference finger tip
        """
        try:
            # Initialize MediaPipe Hands
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.3
            )
            
            # Convert to RGB for MediaPipe
            if hand_image.shape[2] == 4:
                rgb_image = cv2.cvtColor(hand_image, cv2.COLOR_BGRA2RGB)
            else:
                rgb_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
            
            # Process image to detect hand
            results = hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                h, w = hand_image.shape[:2]
                
                # Choose reference finger based on hand shape
                # Handshapes 1 and 6 use index finger (landmark 8)
                # Other handshapes use middle finger (landmark 12)
                if hand_shape in [1, 6]:
                    ref_landmark_idx = 8  # Index finger tip
                else:
                    ref_landmark_idx = 12  # Middle finger tip
                
                ref_landmark = hand_landmarks.landmark[ref_landmark_idx]
                ref_x = int(ref_landmark.x * w)
                ref_y = int(ref_landmark.y * h)
                
                logger.info(f"MediaPipe detected reference finger for shape {hand_shape}: landmark {ref_landmark_idx} at ({ref_x}, {ref_y}) in {w}x{h} image")
                
                hands.close()
                return ref_x, ref_y
            else:
                logger.warning(f"MediaPipe could not detect hand in hand image for shape {hand_shape}")
                hands.close()
                # Try yellow pixel detection as fallback
                return self._detect_yellow_pixel_fallback(hand_image)
                
        except Exception as e:
            logger.error(f"Error detecting reference finger with MediaPipe: {e}")
            # Fallback to yellow pixel detection
            return self._detect_yellow_pixel_fallback(hand_image)
    
    def _get_hand_landmark_0(self, hand_image: np.ndarray) -> Tuple[int, int]:
        """
        Get the position of hand landmark 0 (wrist) in the hand image using MediaPipe.
        
        Args:
            hand_image: Hand image with RGBA channels
            
        Returns:
            Tuple of (x, y) coordinates of landmark 0 (wrist)
        """
        try:
            # Initialize MediaPipe Hands
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.3
            )
            
            # Convert to RGB for MediaPipe
            if hand_image.shape[2] == 4:
                rgb_image = cv2.cvtColor(hand_image, cv2.COLOR_BGRA2RGB)
            else:
                rgb_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
            
            # Process image to detect hand
            results = hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                h, w = hand_image.shape[:2]
                
                # Get landmark 0 (wrist)
                wrist_landmark = hand_landmarks.landmark[0]
                wrist_x = int(wrist_landmark.x * w)
                wrist_y = int(wrist_landmark.y * h)
                
                logger.info(f"MediaPipe detected wrist (landmark 0) at ({wrist_x}, {wrist_y}) in {w}x{h} image")
                
                hands.close()
                return wrist_x, wrist_y
            else:
                logger.warning(f"MediaPipe could not detect hand landmarks in hand image")
                hands.close()
                # Fallback: use center of image as wrist position
                h, w = hand_image.shape[:2]
                return w // 2, h // 2
                
        except Exception as e:
            logger.error(f"Error detecting wrist with MediaPipe: {e}")
            # Fallback: use center of image as wrist position
            h, w = hand_image.shape[:2]
            return w // 2, h // 2
    
    def _detect_yellow_pixel_fallback(self, hand_image: np.ndarray) -> Tuple[int, int]:
        """
        Fallback method to detect yellow pixel if MediaPipe fails.
        
        Args:
            hand_image: Hand image (already resized) with RGBA channels
            
        Returns:
            Tuple of (x, y) coordinates of the yellow pixel or center
        """
        try:
            # Convert to RGB if needed
            if hand_image.shape[2] == 4:
                rgb_image = cv2.cvtColor(hand_image, cv2.COLOR_BGRA2BGR)
            else:
                rgb_image = hand_image.copy()
            
            # Convert to HSV for better yellow detection
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            
            # Define yellow color range in HSV - broader range for better detection
            lower_yellow = np.array([15, 150, 150])
            upper_yellow = np.array([40, 255, 255])
            
            # Create mask for yellow pixels
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Find yellow pixels
            yellow_pixels = np.where(yellow_mask > 0)
            
            if len(yellow_pixels[0]) > 0:
                # Get the centroid of yellow pixels
                y_coords = yellow_pixels[0]
                x_coords = yellow_pixels[1]
                
                ref_y = int(np.mean(y_coords))
                ref_x = int(np.mean(x_coords))
                
                logger.info(f"Yellow pixel detected at ({ref_x}, {ref_y})")
                return ref_x, ref_y
            else:
                # Ultimate fallback: use center of image
                logger.warning("Yellow pixel not found, using center as fallback")
                h, w = hand_image.shape[:2]
                return w // 2, h // 2
                
        except Exception as e:
            logger.error(f"Error in yellow pixel fallback: {e}")
            # Use center
            h, w = hand_image.shape[:2]
            return w // 2, h // 2
    
    def _add_debug_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        Add debug visualization dots for the 3 reference landmarks used in positioning.
        This helps visualize landmark detection accuracy, especially when the phone is not parallel to the face.
        """
        try:
            # Check if debug landmarks are enabled
            if not self.config.get("enable_debug_landmarks", True):
                return frame
                
            # Only add debug landmarks if we have face landmarks available
            if not hasattr(self, 'current_video_frame') or self.current_video_frame is None:
                return frame
                
            # Convert to RGB and process with MediaPipe to get current landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = rgb_frame.astype(np.uint8)
            
            # Process with MediaPipe FaceMesh
            results = face_mesh.process(rgb_frame)
            if not results.multi_face_landmarks:
                return frame
                
            face_landmarks = results.multi_face_landmarks[0]
            frame_height, frame_width = frame.shape[:2]
            
            # Define the key landmarks used in positioning
            debug_landmarks = [
                (50, "Nose Tip", (0, 255, 0)),      # Green - Nose tip (landmark 50)
                (4, "Nose Bridge", (255, 0, 0)),    # Blue - Nose bridge (landmark 4) 
                (57, "Mouth Corner", (0, 0, 255)),  # Red - Mouth corner (landmark 57)
                (152, "Chin", (255, 255, 0))        # Yellow - Chin (landmark 152)
            ]
            
            # Draw debug dots for each landmark
            for landmark_id, name, color in debug_landmarks:
                if landmark_id < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[landmark_id]
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)
                    
                    # Draw a large, visible dot
                    cv2.circle(frame, (x, y), 8, color, -1)  # Filled circle
                    cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)  # White border
                    
                    # Add text label
                    cv2.putText(frame, f"{landmark_id}", (x + 15, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"{landmark_id}", (x + 15, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Calculate and display positioning information
            nose_landmark_50 = face_landmarks.landmark[50]
            nose_landmark_4 = face_landmarks.landmark[4]
            mouth_corner = face_landmarks.landmark[57]
            chin = face_landmarks.landmark[152]
            
            # Calculate nose distance (used for positioning)
            nose_distance_x = abs(nose_landmark_50.x - nose_landmark_4.x) * frame_width
            nose_distance_y = abs(nose_landmark_50.y - nose_landmark_4.y) * frame_height
            
            # Add a legend in the top-left corner
            legend_y = 30
            cv2.putText(frame, "Debug Landmarks:", (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Debug Landmarks:", (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            legend_items = [
                ("50: Nose Tip", (0, 255, 0)),
                ("4: Nose Bridge", (255, 0, 0)), 
                ("57: Mouth Corner", (0, 0, 255)),
                ("152: Chin", (255, 255, 0))
            ]
            
            for i, (text, color) in enumerate(legend_items):
                y_pos = legend_y + 25 + (i * 20)
                cv2.circle(frame, (15, y_pos - 5), 6, color, -1)
                cv2.putText(frame, text, (30, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, text, (30, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Add positioning information
            info_y = legend_y + 25 + (len(legend_items) * 20) + 20
            cv2.putText(frame, f"Nose Distance: {nose_distance_x:.1f}x{nose_distance_y:.1f}px", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Nose Distance: {nose_distance_x:.1f}x{nose_distance_y:.1f}px", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Show calculated positions for positions 1 and 5
            pos1_x = int(mouth_corner.x * frame_width - nose_distance_x)
            pos1_y = int(mouth_corner.y * frame_height)
            pos5_x = int(chin.x * frame_width)
            pos5_y = int(chin.y * frame_height + nose_distance_y)
            
            cv2.putText(frame, f"Pos1: ({pos1_x},{pos1_y})", 
                       (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Pos1: ({pos1_x},{pos1_y})", 
                       (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            cv2.putText(frame, f"Pos5: ({pos5_x},{pos5_y})", 
                       (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Pos5: ({pos5_x},{pos5_y})", 
                       (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw calculated position markers
            cv2.circle(frame, (pos1_x, pos1_y), 6, (255, 0, 255), -1)  # Magenta for Pos1
            cv2.circle(frame, (pos1_x, pos1_y), 10, (255, 255, 255), 2)  # White border
            cv2.putText(frame, "P1", (pos1_x + 15, pos1_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.circle(frame, (pos5_x, pos5_y), 6, (0, 255, 255), -1)  # Cyan for Pos5
            cv2.circle(frame, (pos5_x, pos5_y), 10, (255, 255, 255), 2)  # White border
            cv2.putText(frame, "P5", (pos5_x + 15, pos5_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw position 2, 3, 4 markers as well
            pos2_x = int(face_landmarks.landmark[50].x * frame_width)
            pos2_y = int(face_landmarks.landmark[50].y * frame_height)
            cv2.circle(frame, (pos2_x, pos2_y), 6, (255, 128, 0), -1)  # Orange for Pos2
            cv2.circle(frame, (pos2_x, pos2_y), 10, (255, 255, 255), 2)
            cv2.putText(frame, "P2", (pos2_x + 15, pos2_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            pos3_x = int(face_landmarks.landmark[57].x * frame_width)
            pos3_y = int(face_landmarks.landmark[57].y * frame_height)
            cv2.circle(frame, (pos3_x, pos3_y), 6, (128, 0, 255), -1)  # Purple for Pos3
            cv2.circle(frame, (pos3_x, pos3_y), 10, (255, 255, 255), 2)
            cv2.putText(frame, "P3", (pos3_x + 15, pos3_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            pos4_x = int(face_landmarks.landmark[175].x * frame_width)
            pos4_y = int(face_landmarks.landmark[175].y * frame_height)
            cv2.circle(frame, (pos4_x, pos4_y), 6, (0, 128, 255), -1)  # Light blue for Pos4
            cv2.circle(frame, (pos4_x, pos4_y), 10, (255, 255, 255), 2)
            cv2.putText(frame, "P4", (pos4_x + 15, pos4_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        except Exception as e:
            logger.warning(f"Debug landmarks visualization failed: {e}")
            
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
            # Position 1: Side of mouth
            nose_landmark_50 = face_landmarks.landmark[50]  # Nose tip
            nose_landmark_4 = face_landmarks.landmark[4]    # Nose bridge
            
            # Calculate the distance between nose landmarks
            nose_distance_x = abs(nose_landmark_50.x - nose_landmark_4.x) * frame_width
            nose_distance_y = abs(nose_landmark_50.y - nose_landmark_4.y) * frame_height
            
            # Use landmark 57 (mouth corner)
            mouth_corner = face_landmarks.landmark[57]
            target_x = mouth_corner.x * frame_width - nose_distance_x
            target_y = mouth_corner.y * frame_height
            
        elif hand_pos == -2:
            # Position 5: Throat/below chin - use relative distance based on face height
            # Calculate face height for better scaling
            nose_landmark_50 = face_landmarks.landmark[50]  # Nose tip
            nose_landmark_4 = face_landmarks.landmark[4]    # Nose bridge
            chin = face_landmarks.landmark[152]
            
            # Calculate face height from nose bridge to chin
            face_height = abs(chin.y - nose_landmark_4.y) * frame_height
            
            # Position below chin at about 0.5x face height
            target_x = chin.x * frame_width
            target_y = chin.y * frame_height + (face_height * 0.5)
            
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
) -> str:
    """
    Generate cued speech video from text input or extract text from video using Whisper.
    
    This is a convenience function that creates a CuedSpeechGenerator instance.
    For more control, use CuedSpeechGenerator directly.
    """
    generator = CuedSpeechGenerator(config)
    return generator.generate_cue(text, video_path, output_path, audio_path)
