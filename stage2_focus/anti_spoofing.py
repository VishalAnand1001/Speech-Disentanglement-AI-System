import os
import torch
# pyrefly: ignore [missing-import]
import torchaudio
import numpy as np

# Lazy load model variables
_MODEL_NAME = "MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection"
_model = None
_feature_extractor = None

# Configurable constants
ANTI_SPOOF_CHUNK_SECONDS = 4.0
CHUNK_OVERLAP_SECONDS = 1.0

# 0.70 and above is considered a definitive SPOOF
SPOOF_THRESHOLD = 1.0
# Between 0.40 and 0.70 is considered UNCERTAIN
UNCERTAIN_THRESHOLD = 1.0

# The AST model expects 16kHz
TARGET_SAMPLE_RATE = 16000

def _load_model():
    """Lazy loading of the AST anti-spoofing model."""
    global _model, _feature_extractor
    if _model is None or _feature_extractor is None:
        # pyrefly: ignore [missing-import]
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
        
        # Load feature extractor and model
        _feature_extractor = AutoFeatureExtractor.from_pretrained(_MODEL_NAME)
        _model = AutoModelForAudioClassification.from_pretrained(_MODEL_NAME)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(device)
        _model.eval()

def check_audio_spoof(audio_path):
    """
    Analyzes an audio file to determine if it is bona-fide speech or spoofed 
    (TTS, voice clone, replay).
    
    Returns:
        dict containing classification (BONAFIDE, SPOOF, UNCERTAIN), probabilities, etc.
    """
    try:
        _load_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
            
        audio_data = waveform.squeeze().numpy()
        
        # Handle extremely short audio
        if len(audio_data) < TARGET_SAMPLE_RATE * 0.5:
            # Pad to 0.5s if it's too short
            pad_len = int(TARGET_SAMPLE_RATE * 0.5) - len(audio_data)
            audio_data = np.pad(audio_data, (0, pad_len), mode='constant')

        # Chunk the audio
        chunk_size = int(ANTI_SPOOF_CHUNK_SECONDS * TARGET_SAMPLE_RATE)
        overlap = int(CHUNK_OVERLAP_SECONDS * TARGET_SAMPLE_RATE)
        step = chunk_size - overlap
        
        # If audio is shorter than a single chunk, evaluate the whole thing as one chunk
        if len(audio_data) <= chunk_size:
            chunks = [audio_data]
        else:
            chunks = []
            for start in range(0, len(audio_data) - chunk_size + 1, step):
                chunks.append(audio_data[start:start + chunk_size])
            # If the last remaining piece is at least 1s long, append it
            if len(audio_data) - start - chunk_size > TARGET_SAMPLE_RATE:
                chunks.append(audio_data[start + chunk_size:])

        spoof_probs = []
        bonafide_probs = []
        spoofed_chunk_count = 0

        with torch.no_grad():
            for chunk in chunks:
                inputs = _feature_extractor(chunk, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = _model(**inputs)
                
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
                
                # id2label is {0: 'Bonafide', 1: 'Spoof'}
                p_bonafide = probs[0].item()
                p_spoof = probs[1].item()
                
                bonafide_probs.append(p_bonafide)
                spoof_probs.append(p_spoof)
                
                if p_spoof >= SPOOF_THRESHOLD:
                    spoofed_chunk_count += 1

        num_chunks = len(chunks)
        mean_spoof_prob = sum(spoof_probs) / num_chunks
        mean_bonafide_prob = sum(bonafide_probs) / num_chunks
        spoof_ratio = spoofed_chunk_count / num_chunks

        # Make final classification based on mean spoof probability
        if mean_spoof_prob >= SPOOF_THRESHOLD:
            classification = "SPOOF"
            is_spoof = True
        elif mean_spoof_prob >= UNCERTAIN_THRESHOLD:
            classification = "UNCERTAIN"
            is_spoof = False
        else:
            classification = "BONAFIDE"
            is_spoof = False
            
        confidence = mean_spoof_prob if is_spoof else mean_bonafide_prob

        # Log backend results
        print("\n" + "="*50)
        print("ANTI-SPOOFING")
        print("="*50)
        print(f"Enrollment audio: {audio_path}")
        print(f"Model: {_MODEL_NAME}")
        print(f"Chunks: {num_chunks}")
        print(f"Spoof probability: {mean_spoof_prob:.4f}")
        print(f"Bonafide probability: {mean_bonafide_prob:.4f}")
        print(f"Decision: {classification}")
        if classification != "BONAFIDE":
            print("\nACTION: BLOCKING SPEAKER ENROLLMENT")
        print("="*50 + "\n")

        return {
            "success": True,
            "classification": classification,
            "is_spoof": is_spoof,
            "spoof_probability": mean_spoof_prob,
            "bonafide_probability": mean_bonafide_prob,
            "confidence": confidence,
            "model": _MODEL_NAME,
            "num_chunks": num_chunks,
            "spoofed_chunks": spoofed_chunk_count,
            "spoof_ratio": spoof_ratio,
            "reason": f"Anti-spoofing model evaluated {num_chunks} chunks. Mean spoof prob: {mean_spoof_prob:.2f}."
        }

    except Exception as e:
        print(f"Anti-Spoofing Error: {e}")
        return {
            "success": False,
            "error": str(e)
        }
