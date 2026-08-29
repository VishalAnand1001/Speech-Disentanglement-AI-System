from stage1_listen.noise_suppression import clean_audio
from stage2_focus.enrollment import extract_voiceprint
from stage2_focus.speaker_isolation import isolate_target_speaker
from stage2_focus.confidence_gate import evaluate_isolation_confidence
from stage3_detect.keyword_spotting import detect_custom_word

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
    detect_custom_word(keyword)

if __name__ == "__main__":
    main()