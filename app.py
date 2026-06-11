from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path

from stage1_listen.noise_suppression import clean_audio
from stage2_focus.enrollment import extract_voiceprint
from stage2_focus.speaker_isolation import isolate_target_speaker
from stage3_detect.keyword_spotting import detect_custom_word

app = Flask(__name__)
CORS(app)

RAW_DIR = Path("data/01_raw_noisy")

INPUT1 = RAW_DIR / "input1.wav"
MIXED = RAW_DIR / "mixed_audio3.wav"


@app.route("/audio")
def get_audio():

    return send_file(
        "data/04_isolated_audio/target_speaker_isolated.wav",
        mimetype="audio/wav"
    )


@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        if "scammerAudio" not in request.files:
            return jsonify({
                "error": "Scammer audio missing"
            }), 400

        if "mixedAudio" not in request.files:
            return jsonify({
                "error": "Mixed audio missing"
            }), 400

        keyword = request.form.get("keyword", "").strip()

        if not keyword:
            return jsonify({
                "error": "Keyword missing"
            }), 400

        RAW_DIR.mkdir(parents=True, exist_ok=True)

        scammer_audio = request.files["scammerAudio"]
        mixed_audio = request.files["mixedAudio"]

        scammer_audio.save(INPUT1)
        mixed_audio.save(MIXED)

        print("\n===== STAGE 1 =====")
        clean_audio()

        print("\n===== STAGE 2A =====")
        extract_voiceprint()

        print("\n===== STAGE 2B =====")
        isolate_target_speaker()

        print("\n===== STAGE 3 =====")
        result = detect_custom_word(keyword)

        return jsonify({
            "success": True,
            "found": result["found"],
            "matches": result["matches"],
            "transcript": result.get("transcript", ""),
            "audio_file": "target_speaker_isolated.wav"
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)