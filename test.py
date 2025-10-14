import whisper
from cued_speech import generate_cue

model = whisper.load_model("medium", device="cpu", download_root="download")

result_path = generate_cue(
    text=None,  # Whisper will transcribe
    video_path="download/test_generate.mp4",
    output_path="output/generator/",
    audio_path=None,
    config={
        "language": "french",
        "hand_scale_factor": 0.75,
        "model": model,  # use preloaded whisper model
    }
)
print("Generated:", result_path)