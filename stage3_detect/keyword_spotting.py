from pathlib import Path
from faster_whisper import WhisperModel

ISOLATED_AUDIO = Path("data/04_isolated_audio/target_speaker_isolated.wav")

MODEL_SIZE = "small"


def detect_custom_word():
    if not ISOLATED_AUDIO.exists():
        print("Isolated audio not found:", ISOLATED_AUDIO)
        return

    keyword = input("Enter keyword/phrase to detect: ").strip().lower()

    if not keyword:
        print("Keyword cannot be empty.")
        return

    print("Loading Whisper model...")

    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8"
    )

    print("Transcribing isolated target speaker audio...")

    segments, info = model.transcribe(
        str(ISOLATED_AUDIO),
        beam_size=5,
        language="en",
        vad_filter=True
    )

    transcript = ""

    for segment in segments:
        transcript += segment.text + " "

    transcript = transcript.strip().lower()

    print("\nTranscript:")
    print(transcript)

    print("\nKeyword:")
    print(keyword)

    if keyword in transcript:
        print("\nYES - Target speaker said the keyword.")
    else:
        print("\nNO - Target speaker did not say the keyword.")


if __name__ == "__main__":
    detect_custom_word()