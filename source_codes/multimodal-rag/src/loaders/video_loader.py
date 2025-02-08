# Standard library imports
import os, time
from typing import List, Tuple, Optional

# Third-party library imports
import cv2
import numpy as np
from PIL import Image
from moviepy import VideoFileClip
import whisper

# Local/application imports
from utils.config import (
    text_model,
    image_model
)
from utils.helpers import adjust_embedding_dimension

def extract_frames(
    video_path: str, frame_rate: float = 1.0
) -> List[Tuple[np.ndarray, float]]:
    """
    Extract frames from video at specified frame rate.
    Returns list of (frame, timestamp) tuples.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return frames

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frames.append((frame, timestamp))

        frame_count += 1

    cap.release()
    print(f"Extracted {len(frames)} frames from video at {frame_rate} fps")
    return frames


def extract_audio_and_transcribe(video_path: str, output_audio_path: str) -> str:
    """
    Extract audio from video and transcribe to text using Whisper.
    """
    try:
        # Get the video filename without extension
        video_filename = os.path.splitext(os.path.basename(video_path))[0]

        # Create audio filename with video name
        audio_filename = f"{video_filename}_audio.wav"

        # Combine with output directory
        final_audio_path = os.path.join(os.path.dirname(output_audio_path), audio_filename)

        print("Extracting audio from video...")
        video = VideoFileClip(video_path)

        # Check if video has audio
        if video.audio is None:
            print("No audio found in video file")
            return ""

        # Write audio to temp file
        video.audio.write_audiofile(final_audio_path, logger=None)

        # Load Whisper model and transcribe
        print("Transcribing audio...")
        model = whisper.load_model("medium.en", device="cpu")

        # Set torch dtype to float32 for CPU
        options = dict(fp16=False)
        result = model.transcribe(final_audio_path, **options)

        # Add debug logging
        print(f"Transcription completed. Text length: {len(result['text'])}")
        print(f"First 100 characters of transcription: {result['text'][:100]}...")

        # Cleanup
        video.close()


        return result["text"]

    except Exception as e:
        print(f"Error in audio transcription: {str(e)}")
        import traceback

        print(traceback.format_exc())
        return ""


def embed_frame(frame: np.ndarray) -> np.ndarray:
    """
    Generate embedding for a video frame using the image model.
    """
    try:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # Generate embedding using the image model
        embedding = image_model.encode(pil_image)

        return embedding

    except Exception as e:
        print(f"Error embedding frame: {e}")
        return None


def embed_text(text: str) -> Optional[np.ndarray]:
    """
    Generate embedding for text using the text model.

    Methods available for text_model:
    - encode(): Convert text to embeddings
    - encode_multi_process(): Parallel processing for multiple texts
    - start_multi_process_pool(): Start a pool for parallel processing
    - stop_multi_process_pool(): Stop the parallel processing pool
    """
    try:
        if not text or text.isspace():
            return None

        embedding = text_model.encode(text)
        return embedding

    except Exception as e:
        print(f"Error embedding text: {e}")
        return None

def process_video(video_folder, output_dir, output_audio_path):
    """Process a video file by extracting transcript."""
    try:
       for video in os.listdir(video_folder):
            if video.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more video formats if needed
                video_path = os.path.join(video_folder, video)
                video_name = os.path.splitext(video)[0]
                
                print(f"\nProcessing video: {video_name}")
                
                # Extract and transcribe audio
                transcript = extract_audio_and_transcribe(video_path, output_audio_path)
                if not transcript:
                    print("No transcript generated")
                    return False
                    
                # Save text with video-specific filename
                text_filename = f"{video_name}_transcript.txt"
                text_path = os.path.join(output_dir, text_filename)
                
                with open(text_path, "w") as file:
                    file.write(transcript)
                print(f"Text data saved to {text_path}")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return False
