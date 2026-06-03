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


def detect_custom_word(keyword):
    if not ISOLATED_AUDIO.exists():
        print("Isolated audio not found:", ISOLATED_AUDIO)
        return False

    keyword = clean_word(keyword)

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

        if not segment.words:
            continue

        words = segment.words

        for i in range(len(words)):
            current_word = clean_word(words[i].word)

            # Exact or fuzzy single-word match
            if current_word == keyword or similarity(current_word, keyword) >= 0.75:
                matches.append({
                    "detected_as": current_word,
                    "matched_keyword": keyword,
                    "start": words[i].start,
                    "end": words[i].end,
                    "confidence": similarity(current_word, keyword),
                    "segment_text": segment.text.strip()
                })

            # Two-word fuzzy match, example: "can nap" -> "kidnap"
            if i < len(words) - 1:
                word1 = clean_word(words[i].word)
                word2 = clean_word(words[i + 1].word)

                combined = word1 + word2
                score = similarity(combined, keyword)

                if score >= 0.60:
                    matches.append({
                        "detected_as": word1 + " " + word2,
                        "matched_keyword": keyword,
                        "start": words[i].start,
                        "end": words[i + 1].end,
                        "confidence": score,
                        "segment_text": segment.text.strip()
                    })

            # Three-word fuzzy match, just in case Whisper splits more badly
            if i < len(words) - 2:
                word1 = clean_word(words[i].word)
                word2 = clean_word(words[i + 1].word)
                word3 = clean_word(words[i + 2].word)

                combined = word1 + word2 + word3
                score = similarity(combined, keyword)

                if score >= 0.60:
                    matches.append({
                        "detected_as": word1 + " " + word2 + " " + word3,
                        "matched_keyword": keyword,
                        "start": words[i].start,
                        "end": words[i + 2].end,
                        "confidence": score,
                        "segment_text": segment.text.strip()
                    })

    transcript = normalize_text(transcript)

    print("\nTranscript:")
    print(transcript)

    print("\nKeyword:")
    print(keyword)

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

        return True

    else:
        print("\nNO - Target speaker did not say the keyword.")
        return False


if __name__ == "__main__":
    user_keyword = input("Enter keyword/phrase to detect: ")
    detect_custom_word(user_keyword)