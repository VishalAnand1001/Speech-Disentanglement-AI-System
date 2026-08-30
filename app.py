import os
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path

from stage1_listen.noise_suppression import clean_audio
from stage2_focus.enrollment import extract_voiceprint
from stage2_focus.speaker_isolation import isolate_target_speaker
from stage2_focus.audio_enhancement import enhance_target_audio
from stage2_focus.confidence_gate import evaluate_isolation_confidence
from stage3_detect.keyword_spotting import detect_custom_word
from stage4_analyze.keyword_emotion import analyze_keyword_emotion

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

@app.route("/clean_mixed_audio")
def get_clean_mixed_audio():
    return send_file(
        "data/03_cleaned_audio/mixed_audio3_DeepFilterNet3_pf.wav",
        mimetype="audio/wav"
    )

@app.route("/enhanced_audio")
def get_enhanced_audio():
    return send_file(
        "data/04_isolated_audio/target_speaker_enhanced.wav",
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
        isolation_result = isolate_target_speaker()

        if isolation_result is None or not isolation_result.get("success"):
            return jsonify({
                "success": False,
                "error": isolation_result.get("error", "Isolation failed.") if isolation_result else "Isolation failed."
            }), 500

        print("\n===== STAGE 2C: TARGET AUDIO ENHANCEMENT =====")
        enhanced_path = "data/04_isolated_audio/target_speaker_enhanced.wav"
        enhancement_success = enhance_target_audio(
            input_path=isolation_result["output_path"],
            output_path=enhanced_path
        )
        final_target_audio = enhanced_path if enhancement_success else isolation_result["output_path"]

        print("\n===== CONFIDENCE GATE =====")
        confidence_result = evaluate_isolation_confidence(
            isolated_audio_path=isolation_result["output_path"],
            target_similarity_score=isolation_result["best_similarity"]
        )

        if not confidence_result["passed"]:
            return jsonify({
                "success": True,
                "confidence_gate": confidence_result,
                "found": False,
                "matches": [],
                "transcript": "",
                "warning": "Isolated speech did not pass the confidence gate. ASR was not executed."
            })

        print("\n===== STAGE 3 =====")
        result = detect_custom_word(keyword, audio_path=final_target_audio)
        
        if result.get("found"):
            print("\n===== STAGE 4 (EMOTION) =====")
            result["matches"] = analyze_keyword_emotion(
                audio_path=final_target_audio,
                keyword_matches=result.get("matches", [])
            )

        return jsonify({
            "success": True,
            "confidence_gate": confidence_result,
            "found": result["found"],
            "matches": result["matches"],
            "transcript": result.get("transcript", ""),
            "audio_file": "target_speaker_isolated.wav",
            "enhanced_audio_file": "target_speaker_enhanced.wav" if enhancement_success else "target_speaker_isolated.wav"
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False
    )
