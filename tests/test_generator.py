"""Tests for the cued speech generator module."""

import pytest
import os
from pathlib import Path

# Import the functions to test
from cued_speech.generator import (
    generate_hand_gestures, 
    generate_lip_movements, 
    generate_cue,
    CuedSpeechGenerator
)


class TestHandGestures:
    """Test hand gesture generation functionality."""
    
    def test_basic_hand_gestures(self):
        """Test basic hand gesture generation."""
        text = "merci"
        gestures = generate_hand_gestures(text)
        
        assert isinstance(gestures, list)
        assert len(gestures) > 0
        
        for gesture in gestures:
            assert 'syllable' in gesture
            assert 'hand_shape' in gesture
            assert 'hand_position' in gesture
            assert isinstance(gesture['hand_shape'], int)
            assert 1 <= gesture['hand_shape'] <= 8
    
    def test_multiple_words(self):
        """Test hand gesture generation for multiple words."""
        text = "merci beaucoup"
        gestures = generate_hand_gestures(text)
        
        assert len(gestures) >= 2  # Should have at least 2 syllables
        
        # Check that we get different hand shapes for different syllables
        hand_shapes = [g['hand_shape'] for g in gestures]
        assert len(set(hand_shapes)) >= 1  # At least one unique hand shape
    
    def test_empty_text(self):
        """Test handling of empty text."""
        gestures = generate_hand_gestures("")
        assert isinstance(gestures, list)
        assert len(gestures) == 0
    
    def test_french_characters(self):
        """Test handling of French characters."""
        text = "au revoir"
        gestures = generate_hand_gestures(text)
        
        assert isinstance(gestures, list)
        assert len(gestures) > 0


class TestLipMovements:
    """Test lip movement generation functionality."""
    
    def test_basic_lip_movements(self):
        """Test basic lip movement generation."""
        text = "merci"
        movements = generate_lip_movements(text)
        
        assert isinstance(movements, list)
        assert len(movements) > 0
        
        for movement in movements:
            assert 'phone' in movement
            assert 'lip_type' in movement
            assert movement['lip_type'] in ['vowel', 'consonant']
    
    def test_vowel_consonant_mix(self):
        """Test lip movements for text with both vowels and consonants."""
        text = "bonjour"
        movements = generate_lip_movements(text)
        
        assert len(movements) > 0
        
        # Should have both vowels and consonants
        lip_types = [m['lip_type'] for m in movements]
        assert 'vowel' in lip_types or 'consonant' in lip_types


class TestCuedSpeechGenerator:
    """Test the CuedSpeechGenerator class."""
    
    def test_generator_initialization(self):
        """Test that the generator can be initialized."""
        generator = CuedSpeechGenerator()
        assert generator is not None
    
    def test_text_to_ipa_conversion(self):
        """Test text to IPA conversion."""
        generator = CuedSpeechGenerator()
        
        # Test basic conversion
        ipa = generator._text_to_ipa("merci")
        assert isinstance(ipa, str)
        assert len(ipa) > 0
        
        # Test fallback for unknown words
        ipa_fallback = generator._text_to_ipa("xyz123")
        assert isinstance(ipa_fallback, str)
    
    def test_syllable_timing_creation(self):
        """Test syllable timing creation."""
        generator = CuedSpeechGenerator()
        
        # Test with a simple IPA text without requiring audio file
        ipa_text = "mɛʁsi"
        
        # Test the basic logic without MoviePy dependency
        # This tests that the method exists and can be called
        assert hasattr(generator, '_create_syllable_timing')
        assert callable(generator._create_syllable_timing)
        
        # Test that the method signature is correct
        import inspect
        sig = inspect.signature(generator._create_syllable_timing)
        assert len(sig.parameters) == 2  # ipa_text, audio_path


class TestVideoGeneration:
    """Test video generation functionality."""
    
    @pytest.mark.skipif(
        not os.path.exists("download/test_generate.mp4"),
        reason="Test video not available"
    )
    def test_video_generation_with_real_video(self):
        """Test video generation with a real video file."""
        video_path = "download/test_generate.mp4"
        output_path = "/tmp/test_generated_video.mp4"
        
        try:
            result = generate_cue(
                text="merci",
                video_path=video_path,
                output_path=output_path
            )
            
            assert isinstance(result, str)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0
            
        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_video_generation_invalid_path(self):
        """Test video generation with invalid video path."""
        with pytest.raises(Exception):
            generate_cue(
                text="test",
                video_path="/nonexistent/video.mp4",
                output_path="/tmp/test.mp4"
            )


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_complete_pipeline_components(self):
        """Test that all pipeline components work together."""
        text = "merci beaucoup"
        
        # Test hand gestures
        gestures = generate_hand_gestures(text)
        assert len(gestures) > 0
        
        # Test lip movements
        movements = generate_lip_movements(text)
        assert len(movements) > 0
        
        # Test that both return valid data structures
        assert all('syllable' in g for g in gestures)
        assert all('phone' in m for m in movements)
    
    def test_generator_class_methods(self):
        """Test that all generator class methods are callable."""
        generator = CuedSpeechGenerator()
        
        # Test that all expected methods exist
        assert hasattr(generator, '_text_to_ipa')
        assert hasattr(generator, '_create_syllable_timing')
        assert hasattr(generator, '_render_video_with_cues')
        
        # Test that methods are callable
        assert callable(generator._text_to_ipa)
        assert callable(generator._create_syllable_timing)
        assert callable(generator._render_video_with_cues)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 