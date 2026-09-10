import os
import sys
from pathlib import Path

# Ensure the parent directory is in the path to import stage2_focus
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stage2_focus.anti_spoofing import check_audio_spoof

def main():
    print("==================================================")
    print("ANTI-SPOOFING TEST")
    print("==================================================")

    # Find an audio file to test
    raw_dir = Path("data/01_raw_noisy")
    test_files = list(raw_dir.glob("*.wav"))
    
    if not test_files:
        print(f"No .wav files found in {raw_dir}")
        print("Please ensure you have an audio file in the data/01_raw_noisy/ directory.")
        return

    # Just take the first available test file
    test_audio = test_files[0]
    print(f"Input:\n{test_audio}")
    
    # Run the anti-spoofing check
    result = check_audio_spoof(str(test_audio))
    
    if not result.get("success"):
        print("\nTest failed with error:")
        print(result.get("error"))
        return

    print("\nModel:")
    print(result.get("model"))
    
    print("\nClassification:")
    print(result.get("classification"))
    
    print("\nSpoof Probability:")
    print(f"{result.get('spoof_probability') * 100:.2f}%")
    
    print("\nBonafide Probability:")
    print(f"{result.get('bonafide_probability') * 100:.2f}%")
    
    print("\nConfidence:")
    print(f"{result.get('confidence') * 100:.2f}%")
    
    print("\nChunks:")
    print(result.get("num_chunks"))

    print("==================================================")
    
    print("\nNOTE: This test validates that the AST anti-spoofing model loads,")
    print("chunks the audio, runs inference, and returns a valid result.")
    print("To empirically validate the anti-spoofing *accuracy*, you must provide")
    print("real spoofed samples (e.g., replay recordings, TTS, or voice clones).")

if __name__ == "__main__":
    main()
