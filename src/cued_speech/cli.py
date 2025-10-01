"""Command Line Interface for Cued Speech Decoder."""

import click
import sys
from pathlib import Path

def check_numpy_compatibility():
    """Check for NumPy/PyTorch compatibility issues and provide solutions."""
    try:
        import numpy as np
        import torch
        
        if np.__version__.startswith('2.') and torch.__version__:
            click.echo("❌ NumPy/PyTorch compatibility issue detected!", err=True)
            click.echo("💡 This is a known issue with NumPy 2.x and older PyTorch versions.", err=True)
            click.echo("", err=True)
            click.echo("🔧 Quick fix:", err=True)
            click.echo("   pip install 'numpy>=1.24,<2.0'", err=True)
            click.echo("   pip install --upgrade cued-speech", err=True)
            click.echo("", err=True)
            click.echo("📖 For more details, see: https://github.com/bsow/cued-speech#troubleshooting", err=True)
            return False
        return True
    except ImportError:
        return True

from .generator import CuedSpeechGenerator
from .data_manager import ensure_data_files, get_default_paths, download_and_extract_data, cleanup_data_files


@click.group()
def cli():
    """Cued Speech Processing Tools."""
    # Check compatibility before running any commands
    if not check_numpy_compatibility():
        sys.exit(1)
    pass


@cli.command()
@click.option("--video_path", default="download/test_decode.mp4", help="Path to input cued-speech video")
@click.option("--right_speaker", default=True, type=bool,
              help="Left or right speaker")
@click.option("--model_path", default="download/cuedspeech-model.pt", help="Path to the pretrained model file")
@click.option("--output_path", default="output/decoder/decoded_video.mp4", help="Path to save subtitled video")
@click.option("--vocab_path", default="download/phonelist.csv", help="Path to vocabulary file")
@click.option("--lexicon_path", default="download/lexicon.txt", help="Path to lexicon file")
@click.option("--kenlm_fr", default="download/kenlm_fr.bin", help="Path to KenLM model file")
@click.option("--homophones_path", default="download/homophones_dico.jsonl", help="Path to homophones file")
@click.option("--kenlm_ipa", default="download/kenlm_ipa.binary", help="Path to language model file")
@click.option("--auto_download", default=True, type=bool, help="Automatically download missing data files")
def decode(video_path, right_speaker, model_path, output_path, vocab_path, 
          lexicon_path, kenlm_fr, homophones_path, kenlm_ipa, auto_download):
    """
    Decode a cued-speech video and produce a subtitled video (French sentences at bottom).
    """
    try:
        # Import decoder only when needed
        from .decoder import decode_video
        import os
        
        # Ensure data files are available if auto_download is enabled
        if auto_download:
            print("🔍 Checking for required data files...")
            data_files = ensure_data_files()
            print("✅ All data files are available!")
            
            # Update paths to use downloaded files if they exist
            default_paths = get_default_paths()
            if Path(video_path).exists():
                video_path = video_path
            else:
                video_path = str(default_paths["test_decode"])
                
            if Path(model_path).exists():
                model_path = model_path
            else:
                model_path = str(default_paths["model"])
                
            if Path(vocab_path).exists():
                vocab_path = vocab_path
            else:
                vocab_path = str(default_paths["vocab"])
                
            if Path(lexicon_path).exists():
                lexicon_path = lexicon_path
            else:
                lexicon_path = str(default_paths["lexicon"])
                
            if Path(kenlm_fr).exists():
                kenlm_fr = kenlm_fr
            else:
                kenlm_fr = str(default_paths["kenlm_fr"])
                
            if Path(homophones_path).exists():
                homophones_path = homophones_path
            else:
                homophones_path = str(default_paths["homophones"])
                
            if Path(kenlm_ipa).exists():
                kenlm_ipa = kenlm_ipa
            else:
                kenlm_ipa = str(default_paths["kenlm_ipa"])
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        decode_video(
            video_path=video_path,
            right_speaker=right_speaker,
            model_path=model_path,
            output_path=output_path,
            vocab_path=vocab_path,
            lexicon_path=lexicon_path,
            kenlm_model_path=kenlm_fr,
            homophones_path=homophones_path,
            lm_path=kenlm_ipa
        )
        click.echo(f"✅ Decoding complete! Output saved to: {output_path}")
    except ImportError as e:
        click.echo(f"❌ Decoder not available: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Error during decoding: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--text", default=None, help="French text to convert (optional, will be extracted from video using Whisper)")
@click.option("--output_path", default="output/generator/generated_cued_speech.mp4", help="Path to save generated cued speech video")
@click.option("--audio_path", default=None, help="Path to audio file (optional, will extract from video if not provided)")
@click.option("--language", default="french", help="Language for speech processing")
@click.option("--skip-whisper", is_flag=True, help="Skip Whisper transcription (requires --text to be provided)")
@click.option("--easing", default="ease_in_out_cubic", 
              type=click.Choice(["linear", "ease_in_out_cubic", "ease_out_elastic", "ease_in_out_back"]),
              help="Easing function for gesture transitions")
@click.option("--morphing/--no-morphing", default=True, help="Enable/disable hand shape morphing")
@click.option("--transparency/--no-transparency", default=True, help="Enable/disable transparency effects")
@click.option("--curving/--no-curving", default=True, help="Enable/disable curved trajectories")
def generate(video_path, text, output_path, audio_path, language, skip_whisper, 
            easing, morphing, transparency, curving):
    """
    Generate cued speech video from a video file.
    
    VIDEO_PATH: Path to the input video file
    
    The text will be automatically extracted from the video using Whisper.
    You can optionally provide text manually with --text option.
    Use --skip-whisper if you're having SSL issues with Whisper downloads.
    
    Enhanced Features:
    - Easing functions for smooth gesture transitions
    - Hand shape morphing for natural shape changes
    - Transparency effects during transitions
    - Curved trajectories for obstacle avoidance
    """
    try:
        if skip_whisper and not text:
            click.echo("❌ Error: --skip-whisper requires --text to be provided", err=True)
            click.echo("💡 Example: cued-speech generate video.mp4 --skip-whisper --text 'merci beaucoup'", err=True)
            raise click.Abort()
        
        if text:
            click.echo(f"🎬 Starting cued speech generation with provided text: '{text}'")
        elif skip_whisper:
            click.echo(f"🎬 Starting cued speech generation with manual text (Whisper skipped): '{text}'")
        else:
            click.echo(f"🎬 Starting cued speech generation - extracting text from video using Whisper...")
        
        # Show helpful information about data location
        from .data_manager import get_data_dir
        data_dir = get_data_dir()
        click.echo(f"📁 Data directory: {data_dir}")
        click.echo(f"📹 Video path: {video_path}")
        
        # Import here to avoid circular imports
        from .generator import CuedSpeechGenerator
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create generator with configuration
        config = {
            "language": language,
            "hand_scale_factor": 0.75,
            "video_codec": "libx264",
            "audio_codec": "aac",
            "min_display_duration": 0.4,
            "skip_whisper": skip_whisper,  # Pass the flag to the generator
            "easing_function": easing,
            "enable_morphing": morphing,
            "enable_transparency": transparency,
            "enable_curving": curving,
        }
        
        generator = CuedSpeechGenerator(config)
        
        # Generate the cued speech video
        result_path = generator.generate_cue(
            text=text,  # Will be None if not provided, and Whisper will extract it
            video_path=video_path,
            output_path=output_path,
            audio_path=audio_path
        )
        
        click.echo(f"✅ Cued speech generation complete!")
        click.echo(f"📁 Output saved to: {result_path}")
        if text:
            click.echo(f"🎯 Used provided text: '{text}'")
        elif skip_whisper:
            click.echo(f"🎯 Used manual text (Whisper skipped): '{text}'")
        else:
            click.echo(f"🎯 Text was automatically extracted from video using Whisper")
        
        # Show which features were used
        click.echo(f"🎨 Enhanced features used:")
        click.echo(f"   • Easing: {easing}")
        click.echo(f"   • Morphing: {'enabled' if morphing else 'disabled'}")
        click.echo(f"   • Transparency: {'enabled' if transparency else 'disabled'}")
        click.echo(f"   • Curving: {'enabled' if curving else 'disabled'}")
        
    except Exception as e:
        click.echo(f"❌ Error during generation: {str(e)}", err=True)
        if "SSL" in str(e) or "certificate" in str(e):
            click.echo("💡 SSL error detected. Try using --skip-whisper with manual text:", err=True)
            click.echo("   cued-speech generate video.mp4 --skip-whisper --text 'your text here'", err=True)
        raise click.Abort()


@cli.command()
@click.option("--force", is_flag=True, help="Force re-download even if files exist")
@click.option("--quiet", is_flag=True, help="Suppress progress output")
def download_data(force, quiet):
    """
    Download and extract required data files from GitHub release.
    """
    try:
        click.echo("📥 Downloading cued speech data files...")
        data_dir = download_and_extract_data(force_download=force, show_progress=not quiet)
        click.echo(f"✅ Data files downloaded to: {data_dir}")
        
        # List available files
        from .data_manager import list_data_files
        files = list_data_files()
        click.echo(f"📋 Available files: {len(files)}")
        for file_type, file_path in files.items():
            click.echo(f"  - {file_type}: {file_path.name}")
            
    except Exception as e:
        click.echo(f"❌ Error downloading data: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
def list_data():
    """
    List all available data files.
    """
    try:
        from .data_manager import list_data_files
        files = list_data_files()
        
        click.echo("📋 Available data files:")
        for file_type, file_path in files.items():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            click.echo(f"  - {file_type}: {file_path.name} ({size_mb:.1f} MB)")
            
    except Exception as e:
        click.echo(f"❌ Error listing data: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option("--confirm", is_flag=True, help="Confirm deletion without prompting")
def cleanup_data(confirm):
    """
    Remove all downloaded data files.
    """
    if not confirm:
        if not click.confirm("Are you sure you want to delete all data files?"):
            click.echo("Operation cancelled.")
            return
    
    try:
        cleanup_data_files()
        click.echo("✅ Data files cleaned up successfully!")
    except Exception as e:
        click.echo(f"❌ Error cleaning up data: {str(e)}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()
