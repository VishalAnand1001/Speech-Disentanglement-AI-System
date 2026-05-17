import subprocess
import sys
from pathlib import Path

RAW_DIR = Path("data/01_raw_noisy")
CLEAN_DIR = Path("data/03_cleaned_audio")


def clean_audio():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    wav_files = list(RAW_DIR.glob("*.wav"))

    if not wav_files:
        print("No .wav files found")
        return

    print("Using Python:", sys.executable)

    for audio_file in wav_files:
        print(f"Cleaning: {audio_file.name}")

        command = [
            sys.executable,
            "-m",
            "df.enhance",
            "--pf",
            str(audio_file),
            "-o",
            str(CLEAN_DIR)
        ]

        print("Running command:", command)

        subprocess.run(command, check=True)

    print("Audio cleaning completed.")


if __name__ == "__main__":
    clean_audio()