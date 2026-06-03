from pathlib import Path
from faster_whisper import WhisperModel

ISOLATED_AUDIO = Path("data/04_isolated_audio/target_speaker_isolated.wav")
MODEL_SIZE = "small"


def detect_custom_word(keyword):
    if not ISOLATED_AUDIO.exists():
        print("Isolated audio not found:", ISOLATED_AUDIO)
        return False

    keyword = keyword.strip().lower()

    if not keyword:
        print("Keyword cannot be empty.")
        return False

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
        vad_filter=True,
        word_timestamps=True
    )

    transcript = ""
    matches = []

    for segment in segments:
        transcript += segment.text + " "

        if segment.words:
            for word in segment.words:
                spoken_word = word.word.strip().lower().replace(".", "").replace(",", "")

                if spoken_word == keyword:
                    matches.append({
                        "word": spoken_word,
                        "start": word.start,
                        "end": word.end,
                        "segment_text": segment.text.strip()
                    })

    transcript = transcript.strip().lower()

    print("\nTranscript:")
    print(transcript)

    print("\nKeyword:")
    print(keyword)

    if matches:
        print("\nYES - Target speaker said the keyword.")
        print("\nTimestamps:")

        for match in matches:
            print(
                f"{match['word']} spoken from "
                f"{match['start']:.2f}s to {match['end']:.2f}s"
            )
            print("Context:", match["segment_text"])

        return True

    else:
        print("\nNO - Target speaker did not say the keyword.")
        return False


if __name__ == "__main__":
    user_keyword = input("Enter keyword/phrase to detect: ")
    detect_custom_word(user_keyword)