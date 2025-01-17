# Standard library imports
import os
from typing import List, Tuple, Optional

# Third-party library imports
import cv2
import numpy as np
from PIL import Image
from moviepy import VideoFileClip
import whisper

# Local/application imports
from utils.config import (
    collection,
    text_splitter,
    FIXED_DIMENSION,
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


def extract_audio_and_transcribe(video_path: str) -> str:
    """
    Extract audio from video and transcribe to text using Whisper.
    """
    try:
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Create temp audio file path with safe filename
        safe_filename = "".join(
            c for c in os.path.basename(video_path) if c.isalnum() or c in (" ._-")
        )
        temp_audio_path = os.path.join(temp_dir, f"temp_audio_{safe_filename}.mp3")

        print("Extracting audio from video...")
        video = VideoFileClip(video_path)

        # Check if video has audio
        if video.audio is None:
            print("No audio found in video file")
            return ""

        # Write audio to temp file
        video.audio.write_audiofile(temp_audio_path, logger=None)

        # Verify temp file exists
        if not os.path.exists(temp_audio_path):
            print(f"Failed to create temporary audio file at {temp_audio_path}")
            return ""

        # Load Whisper model and transcribe
        print("Transcribing audio...")
        model = whisper.load_model("base", device="cpu")

        # Set torch dtype to float32 for CPU
        options = dict(fp16=False)
        result = model.transcribe(temp_audio_path, **options)

        # Add debug logging
        print(f"Transcription completed. Text length: {len(result['text'])}")
        print(f"First 100 characters of transcription: {result['text'][:100]}...")

        # Cleanup
        video.close()
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

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

def process_video(file_path):
    """Process a video file by extracting frames and transcripts."""
    try:
        print(f"\n{'='*50}")
        print(f"Starting video processing for: {file_path}")
        
        # 1. Extract frames at a lower frame rate
        print("Step 1: Extracting frames...")
        frames = extract_frames(file_path, frame_rate=0.5)  # 1 frame every 2 seconds
        if not frames:
            print("Warning: No frames were extracted!")
            return False
        print(f"✓ Extracted {len(frames)} frames from video")
        
        # 2. Process frames in even smaller batches
        print("\nStep 2: Processing frames...")
        frame_count = 0
        batch_size = 3  # Further reduced batch size to 3 frames
        max_retries = 3  # Number of retries for failed batches
        
        for i in range(0, len(frames), batch_size):
            try:
                batch_frames = frames[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(frames) + batch_size - 1)//batch_size}")
                
                # Prepare batch data
                batch_embeddings = []
                batch_documents = []
                batch_metadatas = []
                batch_ids = []
                
                # Process each frame in the batch
                for j, (frame, timestamp) in enumerate(batch_frames):
                    try:
                        print(f"  Processing frame {i+j+1} at timestamp {timestamp:.2f}s", end='\r')
                        embedding = embed_frame(frame)
                        if embedding is not None:
                            batch_embeddings.append(adjust_embedding_dimension(embedding, FIXED_DIMENSION))
                            batch_documents.append(f"Frame at {timestamp:.2f}s")
                            batch_metadatas.append({
                                "type": "frame",
                                "file_name": os.path.basename(file_path),
                                "timestamp": timestamp
                            })
                            batch_ids.append(f"{os.path.basename(file_path)}_frame_{timestamp}")
                    except Exception as e:
                        print(f"\n  Error processing frame {i+j+1}: {str(e)}")
                        continue
                
                # Add batch to ChromaDB if we have any valid embeddings
                if batch_embeddings:
                    for retry in range(max_retries):
                        try:
                            print(f"\n  Adding batch of {len(batch_embeddings)} embeddings to ChromaDB...")
                            collection.add(
                                embeddings=batch_embeddings,
                                documents=batch_documents,
                                metadatas=batch_metadatas,
                                ids=batch_ids
                            )
                            frame_count += len(batch_embeddings)
                            print(f"  ✓ Successfully added batch {i//batch_size + 1}")
                            
                            # Add a longer delay between batches
                            import time
                            time.sleep(1.0)  # 1 second delay between batches
                            break  # Success, exit retry loop
                            
                        except Exception as e:
                            print(f"\n  Error adding batch to ChromaDB (attempt {retry + 1}/{max_retries}): {str(e)}")
                            if retry < max_retries - 1:  # If not the last retry
                                print("  Retrying after delay...")
                                time.sleep(2.0)  # Wait 2 seconds before retry
                            else:
                                print("  Failed all retry attempts for this batch")
                
            except Exception as e:
                print(f"\nError processing batch {i//batch_size + 1}: {str(e)}")
                continue
        
        print(f"\n✓ Processed {frame_count} frames")
        
        # 3. Process audio transcript in smaller batches
        print("\nStep 3: Processing audio transcript...")
        transcript = extract_audio_and_transcribe(file_path)
        
        if transcript:
            chunks = text_splitter.split_text(transcript)
            chunk_count = 0
            batch_size = 2  # Reduced batch size for transcript chunks
            
            for i in range(0, len(chunks), batch_size):
                try:
                    batch_chunks = chunks[i:i + batch_size]
                    print(f"Processing transcript batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                    
                    # Prepare batch data
                    batch_embeddings = []
                    batch_documents = []
                    batch_metadatas = []
                    batch_ids = []
                    
                    # Process each chunk in the batch
                    for j, chunk in enumerate(batch_chunks):
                        try:
                            embedding = embed_text(chunk)
                            if embedding is not None:
                                batch_embeddings.append(adjust_embedding_dimension(embedding, FIXED_DIMENSION))
                                batch_documents.append(chunk)
                                batch_metadatas.append({
                                    "type": "transcript",
                                    "file_name": os.path.basename(file_path),
                                    "chunk_index": i + j
                                })
                                batch_ids.append(f"{os.path.basename(file_path)}_transcript_{i + j}")
                        except Exception as e:
                            print(f"  Error processing chunk {i+j+1}: {str(e)}")
                            continue
                    
                    # Add batch to ChromaDB if we have any valid embeddings
                    if batch_embeddings:
                        try:
                            collection.add(
                                embeddings=batch_embeddings,
                                documents=batch_documents,
                                metadatas=batch_metadatas,
                                ids=batch_ids
                            )
                            chunk_count += len(batch_embeddings)
                            print(f"  ✓ Added transcript batch {i//batch_size + 1}")
                            time.sleep(0.5)  # 500ms delay between batches
                            
                        except Exception as e:
                            print(f"  Error adding transcript batch: {str(e)}")
                            
                except Exception as e:
                    print(f"Error processing transcript batch {i//batch_size + 1}: {str(e)}")
                    continue
            
            print(f"✓ Processed {chunk_count} transcript chunks")
        
        print(f"\nCompleted processing video: {os.path.basename(file_path)}")
        print(f"{'='*50}")
        return True
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False
