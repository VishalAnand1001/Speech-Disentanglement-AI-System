import torch
import numpy as np
import scipy.io.wavfile as wavfile
import math
from pathlib import Path

# Configurable Weights
ECAPA_WEIGHT = 0.50
VOICED_WEIGHT = 0.20
NOISE_WEIGHT = 0.15
ENERGY_WEIGHT = 0.10
CLIPPING_WEIGHT = 0.05

CONFIDENCE_THRESHOLD = 0.45
ECAPA_MIN_SCORE = 0.40

def evaluate_isolation_confidence(isolated_audio_path: str, target_similarity_score: float, mixed_audio_path: str = None) -> dict:
    """
    Mathematically evaluate whether the separated audio is reliable enough to be trusted,
    using inference-time signal-processing metrics.
    """
    
    # Base failure response
    fail_res = {
        "passed": False,
        "confidence_score": 0.0,
        "ecapa_score": target_similarity_score,
        "voiced_ratio": 0.0,
        "noise_estimate_score": 0.0,
        "clipping_ratio": 0.0,
        "energy_score": 0.0,
        "reason": "Unknown failure",
        "metrics": {}
    }

    try:
        audio_path = Path(isolated_audio_path)
        if not audio_path.exists():
            fail_res["reason"] = f"File not found: {isolated_audio_path}"
            return fail_res

        sample_rate, signal_np = wavfile.read(str(audio_path))
        
        # Handle empty audio
        if signal_np.size == 0:
            fail_res["reason"] = "Audio file is empty."
            return fail_res
            
        # Normalize to float32 in [-1.0, 1.0] if it's integer
        if np.issubdtype(signal_np.dtype, np.integer):
            if signal_np.dtype == np.int16:
                signal_np = signal_np.astype(np.float32) / 32768.0
            elif signal_np.dtype == np.int32:
                signal_np = signal_np.astype(np.float32) / 2147483648.0
            else:
                max_val = float(np.iinfo(signal_np.dtype).max)
                signal_np = signal_np.astype(np.float32) / max_val
        else:
            signal_np = signal_np.astype(np.float32)

        # Convert to torch tensor. wavfile returns (frames, channels)
        # We need (channels, frames) to match torchaudio's output shape before conversion
        if signal_np.ndim == 1:
            signal = torch.from_numpy(signal_np).unsqueeze(0)
        else:
            signal = torch.from_numpy(signal_np).transpose(0, 1)
            
        # Convert to mono if necessary
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
            
        signal = signal.squeeze()
        
        # Handle very short audio (less than 0.1 seconds)
        if len(signal) < sample_rate * 0.1:
            fail_res["reason"] = "Audio file is too short to evaluate."
            return fail_res
            
        # 1. ECAPA Similarity Score
        # Clamp between 0 and 1. (Sometimes cosine similarity is slightly negative or > 1)
        ecapa_normalized = max(0.0, min(1.0, target_similarity_score))
        
        # 2. RMS / Energy Score
        # Calculate RMS of the entire signal
        rms = torch.sqrt(torch.mean(signal**2)).item()
        # Normalize RMS (assuming normal speech RMS is around 0.05 - 0.2)
        # We don't want to penalize quiet speech too much, so we clamp it generously.
        energy_score = min(1.0, rms / 0.05) if rms > 1e-5 else 0.0
        
        # 3. Voiced Ratio
        # Frame the audio (e.g., 20ms frames)
        frame_size = int(sample_rate * 0.02)
        if frame_size > 0 and len(signal) >= frame_size:
            # Unfold into frames
            frames = signal.unfold(0, frame_size, frame_size)
            # Calculate RMS per frame
            frame_rms = torch.sqrt(torch.mean(frames**2, dim=1))
            
            # Adaptive threshold: 10% of max frame RMS, or a small absolute floor
            max_rms = torch.max(frame_rms).item()
            threshold = max(max_rms * 0.1, 1e-4)
            
            voiced_frames = torch.sum(frame_rms > threshold).item()
            total_frames = frame_rms.shape[0]
            voiced_ratio = voiced_frames / total_frames if total_frames > 0 else 0.0
        else:
            voiced_ratio = 0.0
            
        # 4. Clipping Ratio
        # Proportion of samples near maximum absolute amplitude
        clipping_samples = torch.sum(torch.abs(signal) >= 0.99).item()
        clipping_ratio = clipping_samples / len(signal)
        clipping_score = max(0.0, 1.0 - (clipping_ratio * 10)) # Penalize heavily if > 10% clipping
        
        # 5. Residual-Noise Estimate
        # Using the frame_rms from above.
        if 'frame_rms' in locals() and len(frame_rms) >= 10:
            sorted_rms, _ = torch.sort(frame_rms)
            # Take lowest 20% frames as noise floor estimate
            bottom_20_count = max(1, int(len(sorted_rms) * 0.2))
            top_20_count = max(1, int(len(sorted_rms) * 0.2))
            
            noise_floor = torch.mean(sorted_rms[:bottom_20_count]).item()
            speech_level = torch.mean(sorted_rms[-top_20_count:]).item()
            
            if speech_level > 1e-6:
                noise_ratio = noise_floor / speech_level
                noise_estimate_score = max(0.0, 1.0 - min(1.0, noise_ratio * 2)) 
            else:
                noise_estimate_score = 0.0
        else:
            noise_estimate_score = 0.5 # Default if too short
            
        # Composite Score Calculation
        confidence_score = (
            ECAPA_WEIGHT * ecapa_normalized +
            VOICED_WEIGHT * voiced_ratio +
            NOISE_WEIGHT * noise_estimate_score +
            ENERGY_WEIGHT * energy_score +
            CLIPPING_WEIGHT * clipping_score
        )
        
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        # Gate Logic
        passed = False
        reason = ""
        
        if ecapa_normalized < ECAPA_MIN_SCORE:
            reason = f"Failed: ECAPA similarity ({ecapa_normalized:.3f}) is below hard minimum ({ECAPA_MIN_SCORE})."
        elif confidence_score >= CONFIDENCE_THRESHOLD:
            passed = True
            reason = "Isolation passed confidence gate."
        else:
            reason = f"Failed: Composite confidence ({confidence_score:.3f}) is below threshold ({CONFIDENCE_THRESHOLD})."
            
        # Logging
        print("\n===== CONFIDENCE GATE =====")
        print(f"ECAPA Similarity: {ecapa_normalized:.4f}")
        print(f"Voiced Ratio: {voiced_ratio:.2f}")
        print(f"Noise Quality: {noise_estimate_score:.2f}")
        print(f"Clipping Quality: {clipping_score:.2f}")
        print(f"Energy Quality: {energy_score:.2f}")
        print(f"\nComposite Confidence: {confidence_score:.2f}")
        print(f"Threshold: {CONFIDENCE_THRESHOLD:.2f}")
        print(f"\nCONFIDENCE GATE: {'PASSED' if passed else 'FAILED'}")
        if not passed:
            print(f"Reason: {reason}")
            
        return {
            "passed": passed,
            "confidence_score": confidence_score,
            "ecapa_score": ecapa_normalized,
            "voiced_ratio": voiced_ratio,
            "noise_estimate_score": noise_estimate_score,
            "clipping_ratio": clipping_ratio,
            "clipping_score": clipping_score,
            "energy_score": energy_score,
            "reason": reason,
            "metrics": {
                "rms": rms,
                "clipping_samples": clipping_samples
            }
        }

    except Exception as e:
        print(f"Confidence Gate Error: {str(e)}")
        fail_res["reason"] = f"Unable to evaluate isolated audio quality: {str(e)}"
        return fail_res
