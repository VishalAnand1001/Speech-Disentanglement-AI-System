import os
import torch
import torchaudio

try:
    from speechbrain.inference.interfaces import foreign_class
    classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier", run_opts={"device":"cpu"})
    print("Speechbrain emotion model loaded successfully.")
except Exception as e:
    print(f"Error loading speechbrain emotion model: {e}")
