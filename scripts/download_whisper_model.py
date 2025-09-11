#!/usr/bin/env python3
import sys


def main():
    """Download the Whisper tiny model."""
    
    print(f"Downloading Whisper tiny model.")
    
    try:
        import whisper
        
        print("Loading Whisper tiny model...")
        model = whisper.load_model("tiny")
        print("Whisper tiny model downloaded successfully!")
        
        print("Testing model with empty audio...")
        import numpy as np
        empty_audio = np.zeros(1600, dtype=np.float32)  # 0.1 seconds of silence
        result = model.transcribe(empty_audio, fp16=False)
        print(f"Model test completed: '{result.get('text', '').strip()}'")
        
        return 0
        
    except ImportError as e:
        print(f"Error: Could not import whisper: {e}")
        print("Make sure openai-whisper is installed")
        return 1
        
    except Exception as e:
        print(f"Error downloading Whisper model: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
