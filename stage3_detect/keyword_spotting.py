from pathlib import Path
from faster_whisper import WhisperModel
import difflib
import re

ISOLATED_AUDIO = Path("data/04_isolated_audio/target_speaker_isolated.wav")
MODEL_SIZE = "small"


def clean_word(word):
    word = word.lower().strip()
    word = re.sub(r"[^a-z0-9]", "", word)
    return word


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def detect_custom_word(keyword, audio_path=None):
    if audio_path is None:
        audio_path = ISOLATED_AUDIO
    else:
        audio_path = Path(audio_path)

    if not audio_path.exists():
        print("Isolated audio not found:", audio_path)

        return {
            "found": False,
            "matches": [],
            "error": "Isolated audio not found"
        }

    keyword = clean_word(keyword)

    if not keyword:
        print("Keyword cannot be empty.")

        return {
            "found": False,
            "matches": [],
            "error": "Keyword cannot be empty"
        }

    print("Loading Whisper model...")

    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8"
    )

    print("Transcribing isolated target speaker audio...")

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language="en",
        vad_filter=True,
        word_timestamps=True
    )

    transcript = ""
    matches = []

    for segment in segments:
        transcript += segment.text + " "

        if not segment.words:
            continue

        for word_info in segment.words:

            current_word = clean_word(word_info.word)

            score = similarity(current_word, keyword)

            if current_word == keyword or score >= 0.75:

                matches.append({
                    "detected_as": current_word,
                    "matched_keyword": keyword,
                    "start": word_info.start,
                    "end": word_info.end,
                    "confidence": score,
                    "segment_text": segment.text.strip()
                })

    transcript = normalize_text(transcript)

    print("\nTranscript:")
    print(transcript)

    print("\nKeyword:")
    print(keyword)

    # Remove Whisper duplicate detections

    matches = sorted(matches, key=lambda x: x["start"])

    filtered_matches = []

    for match in matches:

        if not filtered_matches:
            filtered_matches.append(match)
            continue

        previous = filtered_matches[-1]

        if (
            match["detected_as"] == previous["detected_as"]
            and abs(match["start"] - previous["start"]) < 5
        ):
            continue

        filtered_matches.append(match)

    matches = filtered_matches

    if matches:

        print("\nYES - Target speaker said the keyword.")
        print("\nTimestamps:")

        for match in matches:

            print(
                f"{match['detected_as']} matched '{match['matched_keyword']}' "
                f"from {match['start']:.2f}s to {match['end']:.2f}s "
                f"(score: {match['confidence']:.2f})"
            )

            print("Context:", match["segment_text"])

        return {
            "found": True,
            "matches": matches,
            "transcript": transcript
        }

    else:

        print("\nNO - Target speaker did not say the keyword.")

        return {
            "found": False,
            "matches": [],
            "transcript": transcript
        }


if __name__ == "__main__":
    user_keyword = input("Enter keyword/phrase to detect: ")

    result = detect_custom_word(user_keyword)

    print("\nReturned Result:")
    print(result)