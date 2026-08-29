import os
import sys
import shutil
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Pipeline imports
from stage1_listen.noise_suppression import clean_audio
from stage2_focus.enrollment import extract_voiceprint
from stage2_focus.speaker_isolation import isolate_target_speaker
from stage2_focus.confidence_gate import evaluate_isolation_confidence
from stage3_detect.keyword_spotting import detect_custom_word

# Ensure we have a place for test files
TEST_DIR = Path("data/test_confidence")
TEST_DIR.mkdir(parents=True, exist_ok=True)

def run_test_a():
    print("\n==================================================")
    print("TEST A: NORMAL EXISTING AUDIO")
    print("==================================================")
    
    print("\n[OK] Running Stage 1: DeepFilterNet Noise Suppression...")
    clean_audio()
    
    print("\n[OK] Running Stage 2A: ECAPA-TDNN Enrollment...")
    extract_voiceprint()
    
    print("\n[OK] Running Stage 2B: SepFormer Speaker Isolation...")
    isolation_result = isolate_target_speaker()
    
    if not isolation_result or not isolation_result.get("success"):
        print("\nPIPELINE ERROR: Speaker isolation failed entirely.")
        print(isolation_result)
        return None
        
    print(f"[OK] Speaker separation completed. Best similarity: {isolation_result['best_similarity']:.4f}")
    
    print("\n[OK] Running Confidence Gate Evaluation...")
    confidence_result = evaluate_isolation_confidence(
        isolated_audio_path=isolation_result["output_path"],
        target_similarity_score=isolation_result["best_similarity"]
    )
    
    print("\n==================================================")
    print("CONFIDENCE GATE TEST (NORMAL AUDIO)")
    print("==================================================")
    print("Mixed audio:\n  data/01_raw_noisy/mixed_audio3.wav")
    print("Enrollment audio:\n  data/01_raw_noisy/input1.wav")
    print(f"Isolated audio:\n  {isolation_result['output_path']}")
    
    print("\n--------------------------------------------------")
    print("SIGNAL METRICS")
    print("--------------------------------------------------")
    print(f"ECAPA Similarity       : {confidence_result.get('ecapa_score', 0):.4f}")
    print(f"Voiced Ratio           : {confidence_result.get('voiced_ratio', 0):.2f}")
    print(f"Noise Quality          : {confidence_result.get('noise_estimate_score', 0):.2f}")
    print(f"Clipping Quality       : {confidence_result.get('clipping_score', 0):.2f}")
    print(f"Energy Quality         : {confidence_result.get('energy_score', 0):.2f}")
    
    print("\n--------------------------------------------------")
    print("COMPOSITE SCORE")
    print("--------------------------------------------------")
    print(f"Confidence Score       : {confidence_result.get('confidence_score', 0):.2f}")
    print(f"Threshold              : 0.45")
    
    print("\n--------------------------------------------------")
    print("DECISION")
    print("--------------------------------------------------")
    
    if confidence_result.get("passed"):
        print("CONFIDENCE GATE: PASSED\n")
        print("Running Faster-Whisper...")
        result = detect_custom_word("scam")
        print(f"Keyword 'scam' found: {result['found']}")
    else:
        print("CONFIDENCE GATE: FAILED\n")
        print(f"Reason:\n{confidence_result.get('reason')}\n")
        print("Faster-Whisper will NOT be executed.")
        print("Keyword detection will NOT be executed.")
        
    return isolation_result["output_path"], isolation_result["best_similarity"]

def run_test_b(original_isolated_path, ecapa_score):
    print("\n==================================================")
    print("TEST B: SYNTHETIC DEGRADED AUDIO")
    print("==================================================")
    
    degraded_path = TEST_DIR / "degraded_test.wav"
    
    # Create degraded audio (near silence to trigger confidence gate failure)
    try:
        sample_rate, signal = wavfile.read(original_isolated_path)
        # Multiply signal by 0.0001 to make it near silent, triggering voiced ratio and energy failure
        if np.issubdtype(signal.dtype, np.integer):
            signal = signal.astype(np.float32) / np.iinfo(signal.dtype).max
        degraded_signal = (signal * 0.00001).astype(np.float32)
        wavfile.write(str(degraded_path), sample_rate, degraded_signal)
    except Exception as e:
        print(f"PIPELINE ERROR: Failed to create synthetic degraded audio: {e}")
        return
        
    print("\n[OK] Created synthetically degraded isolated audio.")
    print(f"Path: {degraded_path}")
    print("\n[OK] Running Confidence Gate Evaluation on DEGRADED audio...")
    
    confidence_result = evaluate_isolation_confidence(
        isolated_audio_path=str(degraded_path),
        target_similarity_score=ecapa_score
    )
    
    if not confidence_result.get("passed"):
        print("\nCONFIDENCE GATE: FAILED (Expected behavior on degraded audio)")
        print(f"Confidence Score: {confidence_result.get('confidence_score', 0):.2f}")
        print(f"Threshold: 0.45")
        print(f"Reason:\n{confidence_result.get('reason')}\n")
        print("Faster-Whisper will NOT be executed.")
        print("Keyword detection will NOT be executed.")
    else:
        print("\n[WARNING] CONFIDENCE GATE: PASSED on degraded audio! This is a failure of the gate logic.")
        print(f"Confidence Score: {confidence_result.get('confidence_score', 0):.2f}")
        
    # Clean up test artifact
    try:
        if degraded_path.exists():
            os.remove(degraded_path)
    except Exception as e:
        pass


if __name__ == "__main__":
    test_result = run_test_a()
    if test_result:
        original_isolated_path, ecapa_score = test_result
        run_test_b(original_isolated_path, ecapa_score)
