import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import butter, filtfilt
from pathlib import Path

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Ensure highcut is strictly less than nyquist frequency
    if high >= 1.0:
        high = 0.999
        
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def enhance_target_audio(input_path: str, output_path: str) -> bool:
    try:
        input_file = Path(input_path)
        output_file = Path(output_path)
        
        if not input_file.exists():
            print(f"Error: Audio file not found at {input_path}")
            return False
            
        fs, data = wavfile.read(str(input_file))
        
        if data.size == 0:
            print("Error: Audio file is empty.")
            return False
            
        # 1. Convert to mono if necessary
        if data.ndim > 1:
            data = np.mean(data, axis=1)
            
        # Convert to float32
        if np.issubdtype(data.dtype, np.integer):
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            else:
                max_val = float(np.iinfo(data.dtype).max)
                data = data.astype(np.float32) / max_val
        else:
            data = data.astype(np.float32)
            
        original_duration = len(data) / fs
        
        # 2. DC Offset Removal
        data -= np.mean(data)
        
        # 3. Conservative Bandpass Filter (80 Hz - 7500 Hz)
        # Avoid highcut being above nyquist
        highcut = min(7500.0, (fs / 2.0) - 100.0)
        if highcut > 80.0:
            data = apply_bandpass_filter(data, 80.0, highcut, fs, order=4)
        
        # 4. Noise Gating (Soft Knee)
        frame_len = int(fs * 0.02) # 20ms
        if frame_len > 0 and len(data) > frame_len:
            squared = data**2
            window = np.ones(frame_len) / frame_len
            envelope = np.sqrt(np.convolve(squared, window, mode='same'))
            
            threshold = np.max(envelope) * 0.05
            
            gain = np.ones_like(data)
            below_thresh = envelope < threshold
            # Avoid division by zero
            if threshold > 1e-6:
                gain[below_thresh] = (envelope[below_thresh] / threshold) ** 0.5
            data = data * gain
            
        # 5. Loudness Normalization
        current_rms = np.sqrt(np.mean(data**2))
        target_rms = 0.1 # ~ -20 dBFS
        if current_rms > 1e-6:
            data = data * (target_rms / current_rms)
            
        # 6. Peak Normalization / Hard Limiter
        peak = np.max(np.abs(data))
        max_amplitude = 0.95
        if peak > max_amplitude:
            data = data * (max_amplitude / peak)
            
        # Export as int16
        # Clip just in case before cast
        data = np.clip(data, -1.0, 1.0)
        data_int16 = (data * 32767.0).astype(np.int16)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(str(output_file), fs, data_int16)
        
        enhanced_duration = len(data_int16) / fs
        
        print("\n===== STAGE 2C: TARGET AUDIO ENHANCEMENT =====")
        print(f"Input: {input_file.name}")
        print(f"Output: {output_file.name}")
        print(f"Original Duration: {original_duration:.2f}s")
        print(f"Enhanced Duration: {enhanced_duration:.2f}s")
        print(f"Sample Rate: {fs} Hz")
        print(f"Peak Amplitude: {peak:.2f} -> {np.max(np.abs(data)):.2f}")
        print("Enhancement succeeded.")
        
        return True
        
    except Exception as e:
        print(f"\n===== STAGE 2C: TARGET AUDIO ENHANCEMENT =====")
        print(f"Audio enhancement failed: {str(e)}")
        print("Falling back to original isolated audio.")
        return False
