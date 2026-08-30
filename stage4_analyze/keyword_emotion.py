import os
import torchaudio
import torch
from pathlib import Path

try:
    from speechbrain.inference.interfaces import foreign_class
except ImportError:
    foreign_class = None

# Lazy loading of the emotion classifier to avoid startup delays
_emotion_classifier = None

def load_emotion_classifier():
    global _emotion_classifier
    if _emotion_classifier is None:
        if foreign_class is None:
            raise ImportError("speechbrain is not installed or missing components.")
        print("Loading SpeechBrain Emotion Classifier (Wav2Vec2-IEMOCAP)...")
        # Run on CPU as requested
        _emotion_classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": "cpu"}
        )
    return _emotion_classifier

def analyze_keyword_emotion(audio_path, keyword_matches, pre_context_seconds=1.0, post_context_seconds=1.0):
    """
    Analyzes the emotion of specific keyword occurrences in the audio.
    
    Args:
        audio_path (str): Path to the isolated target speaker audio.
        keyword_matches (list): List of dictionaries containing keyword match information.
        pre_context_seconds (float): Padding before the keyword start.
        post_context_seconds (float): Padding after the keyword end.
        
    Returns:
        list: The updated keyword_matches with added emotion information.
    """
    if not keyword_matches:
        return keyword_matches
        
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"[Emotion Analysis] Audio file not found: {audio_path}")
        for match in keyword_matches:
            match["emotion"] = None
            match["emotion_status"] = "error"
            match["emotion_error"] = "Audio file not found"
        return keyword_matches

    try:
        classifier = load_emotion_classifier()
    except Exception as e:
        print(f"[Emotion Analysis] Failed to load emotion model: {e}")
        for match in keyword_matches:
            match["emotion"] = None
            match["emotion_status"] = "error"
            match["emotion_error"] = f"Model load error: {str(e)}"
        return keyword_matches

    try:
        # Load audio (torchaudio returns channels, frames)
        signal, sample_rate = torchaudio.load(str(audio_path))
        
        # Resample to 16000Hz as Wav2Vec2 expects 16kHz
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            signal = resampler(signal)
            sample_rate = target_sample_rate
            
        # Convert to mono if necessary
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
            
        audio_duration = signal.shape[1] / sample_rate
        
    except Exception as e:
        print(f"[Emotion Analysis] Failed to process audio file: {e}")
        for match in keyword_matches:
            match["emotion"] = None
            match["emotion_status"] = "error"
            match["emotion_error"] = f"Audio processing error: {str(e)}"
        return keyword_matches

    for match in keyword_matches:
        keyword_start = match.get("start", 0)
        keyword_end = match.get("end", 0)
        
        # Calculate localized window
        window_start = max(0.0, keyword_start - pre_context_seconds)
        window_end = min(audio_duration, keyword_end + post_context_seconds)
        
        start_sample = int(window_start * sample_rate)
        end_sample = int(window_end * sample_rate)
        
        # Ensure we have a valid slice
        if end_sample <= start_sample:
            match["emotion"] = None
            match["emotion_status"] = "error"
            match["emotion_error"] = "Invalid audio segment slice"
            continue
            
        # Extract the segment
        segment = signal[:, start_sample:end_sample]
        
        try:
            # Run inference
            out_prob, score, index, text_lab = classifier.classify_batch(segment)
            
            # The label is a list of strings
            label = text_lab[0] if isinstance(text_lab, list) else text_lab
            label_upper = label.upper()
            
            emotion_map = {
                "ANG": "ANGRY",
                "NEU": "NEUTRAL",
                "HAP": "HAPPY",
                "SAD": "SAD",
                "EXC": "EXCITED",
                "FRU": "FRUSTRATED",
                "FEA": "FEARFUL",
                "SUR": "SURPRISED",
                "DIS": "DISGUSTED"
            }
            formatted_label = emotion_map.get(label_upper, label_upper)
            
            # The probability is a tensor
            confidence = score.item() if hasattr(score, 'item') else float(score[0])
            
            match["emotion"] = {
                "label": formatted_label,
                "confidence": confidence,
                "window_start": window_start,
                "window_end": window_end
            }
            match["emotion_status"] = "analyzed"
            
            print(f"[Emotion Analysis] Keyword '{match.get('detected_as')}' ({keyword_start:.2f}s - {keyword_end:.2f}s) -> {label.upper()} ({confidence:.2f})")
            
        except Exception as e:
            print(f"[Emotion Analysis] Classification failed for segment: {e}")
            match["emotion"] = None
            match["emotion_status"] = "error"
            match["emotion_error"] = f"Classification error: {str(e)}"
            
    return keyword_matches
