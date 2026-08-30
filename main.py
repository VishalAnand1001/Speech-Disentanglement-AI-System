from stage1_listen.noise_suppression import clean_audio
from stage2_focus.enrollment import extract_voiceprint
from stage2_focus.speaker_isolation import isolate_target_speaker
from stage2_focus.audio_enhancement import enhance_target_audio
from stage2_focus.confidence_gate import evaluate_isolation_confidence
from stage3_detect.keyword_spotting import detect_custom_word
from stage4_analyze.keyword_emotion import analyze_keyword_emotion

def main():

    keyword = input("Enter keyword: ")

    print("\n===== STAGE 1 =====")
    clean_audio()

    print("\n===== STAGE 2A =====")
    extract_voiceprint()

    print("\n===== STAGE 2B =====")
    isolation_result = isolate_target_speaker()

    if not isolation_result or not isolation_result.get("success"):
        print("\nPIPELINE ERROR: Speaker isolation failed.")
        return

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

    if not confidence_result.get("passed"):
        print(f"\n[!] ASR SKIPPED: Isolated speech did not pass the confidence gate.")
        print(f"[!] Reason: {confidence_result.get('reason')}")
        return

    print("\n===== STAGE 3 =====")
    keyword_result = detect_custom_word(keyword, audio_path=final_target_audio)
    
    if keyword_result and keyword_result.get("found"):
        print("\n===== STAGE 4 (EMOTION) =====")
        analyze_keyword_emotion(
            audio_path=final_target_audio,
            keyword_matches=keyword_result.get("matches", [])
        )

if __name__ == "__main__":
    main()