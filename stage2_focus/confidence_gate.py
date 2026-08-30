import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import scipy.io.wavfile as wavfile
import math
from pathlib import Path

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

# Configurable Weights
ECAPA_WEIGHT = 0.50
VOICED_WEIGHT = 0.20
NOISE_WEIGHT = 0.15
ENERGY_WEIGHT = 0.10
CLIPPING_WEIGHT = 0.05

CONFIDENCE_THRESHOLD = 0.45
ECAPA_MIN_SCORE = 0.40

# Paths for ECAPA model caching
ECAPA_MODEL_DIR = Path("models/ecapa_tdnn_weights")
TARGET_VOICEPRINT = Path("models/ecapa_tdnn_weights/target_voiceprint.pt")

_ecapa_classifier = None

def get_ecapa_classifier():
    global _ecapa_classifier
    if _ecapa_classifier is None:
        ECAPA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _ecapa_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(ECAPA_MODEL_DIR),
            local_strategy=LocalStrategy.COPY
        )
    return _ecapa_classifier

def evaluate_isolation_confidence(isolated_audio_path: str, target_similarity_score: float, mixed_audio_path: str = None) -> dict:
    """
    Mathematically evaluate whether the separated audio is reliable enough to be trusted,
    using inference-time signal-processing metrics and segment-level ECAPA similarity.
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
            
        # 2. RMS / Energy Score
        rms = torch.sqrt(torch.mean(signal**2)).item()
        energy_score = min(1.0, rms / 0.05) if rms > 1e-5 else 0.0
        
        # 3. Voiced Ratio & Noise Estimate prep
        frame_size = int(sample_rate * 0.02)
        if frame_size > 0 and len(signal) >= frame_size:
            frames = signal.unfold(0, frame_size, frame_size)
            frame_rms = torch.sqrt(torch.mean(frames**2, dim=1))
            max_rms = torch.max(frame_rms).item()
            threshold = max(max_rms * 0.1, 1e-4)
            voiced_frames = torch.sum(frame_rms > threshold).item()
            total_frames = frame_rms.shape[0]
            voiced_ratio = voiced_frames / total_frames if total_frames > 0 else 0.0
        else:
            voiced_ratio = 0.0
            
        # 4. Clipping Ratio
        clipping_samples = torch.sum(torch.abs(signal) >= 0.99).item()
        clipping_ratio = clipping_samples / len(signal)
        clipping_score = max(0.0, 1.0 - (clipping_ratio * 10))
        
        # 5. Residual-Noise Estimate
        if 'frame_rms' in locals() and len(frame_rms) >= 10:
            sorted_rms, _ = torch.sort(frame_rms)
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
            noise_estimate_score = 0.5
            
        # --- NEW: SEGMENT-LEVEL ECAPA LOGIC ---
        segment_duration = 2.0
        overlap = 1.0
        step_samples = int((segment_duration - overlap) * sample_rate)
        window_samples = int(segment_duration * sample_rate)
        
        valid_scores = []
        best_segment_timestamps = None
        
        if TARGET_VOICEPRINT.exists():
            try:
                target_embedding = torch.load(TARGET_VOICEPRINT).squeeze()
                classifier = get_ecapa_classifier()
                
                # Resample signal to 16kHz for ECAPA if needed
                ecapa_signal = signal
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    ecapa_signal = resampler(signal.unsqueeze(0)).squeeze(0)
                    
                ecapa_sample_rate = 16000
                ecapa_window_samples = int(segment_duration * ecapa_sample_rate)
                ecapa_step_samples = int((segment_duration - overlap) * ecapa_sample_rate)
                
                total_len = len(ecapa_signal)
                
                # Fallback if audio is shorter than the window
                if total_len < ecapa_window_samples:
                    if total_len > ecapa_sample_rate * 0.5: # At least 0.5 seconds
                        # Check if it has any voice (RMS threshold)
                        seg_rms = torch.sqrt(torch.mean(ecapa_signal**2)).item()
                        # Use a very generous threshold for edge cases
                        if seg_rms > 1e-4:
                            emb = classifier.encode_batch(ecapa_signal.unsqueeze(0)).squeeze()
                            sim = F.cosine_similarity(target_embedding, emb, dim=0).item()
                            valid_scores.append(sim)
                            best_segment_timestamps = (0.0, total_len / ecapa_sample_rate)
                else:
                    for start_idx in range(0, total_len - ecapa_window_samples + 1, ecapa_step_samples):
                        end_idx = start_idx + ecapa_window_samples
                        segment = ecapa_signal[start_idx:end_idx]
                        
                        # Only run ECAPA on voiced segments (using RMS heuristic)
                        seg_rms = torch.sqrt(torch.mean(segment**2)).item()
                        if seg_rms > max(1e-4, threshold * 0.5):
                            emb = classifier.encode_batch(segment.unsqueeze(0)).squeeze()
                            sim = F.cosine_similarity(target_embedding, emb, dim=0).item()
                            valid_scores.append(sim)
                            if len(valid_scores) == 1 or sim > max(valid_scores[:-1]):
                                best_segment_timestamps = (start_idx / ecapa_sample_rate, end_idx / ecapa_sample_rate)

            except Exception as e:
                print(f"Warning: Segment-level ECAPA failed, using fallback. Error: {e}")
                valid_scores = []
        else:
            print(f"Warning: Target voiceprint not found at {TARGET_VOICEPRINT}, using global fallback score.")

        if valid_scores:
            valid_scores_t = torch.tensor(valid_scores)
            best_similarity = torch.max(valid_scores_t).item()
            median_similarity = torch.median(valid_scores_t).item()
            
            top_k = min(3, len(valid_scores))
            top_3_mean = torch.mean(torch.topk(valid_scores_t, top_k).values).item()
            
            ecapa_normalized = max(0.0, min(1.0, top_3_mean))
        else:
            # Fallback to the original global score
            best_similarity = target_similarity_score
            median_similarity = target_similarity_score
            top_3_mean = target_similarity_score
            ecapa_normalized = max(0.0, min(1.0, target_similarity_score))
            
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
            reason = f"Failed: ECAPA top-3 mean similarity ({ecapa_normalized:.3f}) is below hard minimum ({ECAPA_MIN_SCORE})."
        elif confidence_score >= CONFIDENCE_THRESHOLD:
            passed = True
            reason = "Isolation passed confidence gate."
        else:
            reason = f"Failed: Composite confidence ({confidence_score:.3f}) is below threshold ({CONFIDENCE_THRESHOLD})."
            
        # Logging
        print("\n===== CONFIDENCE GATE =====")
        print(f"Global ECAPA Similarity (Legacy): {target_similarity_score:.4f}")
        print("\nECAPA Segment Analysis")
        print("----------------------")
        print(f"Valid speech segments: {len(valid_scores)}")
        print(f"Best similarity: {best_similarity:.4f}")
        print(f"Top-3 mean similarity: {top_3_mean:.4f}")
        print(f"Median similarity: {median_similarity:.4f}")
        if best_segment_timestamps:
            print(f"Best segment: {best_segment_timestamps[0]:.2f}s - {best_segment_timestamps[1]:.2f}s")
        else:
            print("Best segment: N/A")
            
        print(f"\nVoiced Ratio: {voiced_ratio:.2f}")
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
