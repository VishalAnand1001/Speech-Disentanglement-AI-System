import torch
import torchaudio
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

if not hasattr(torch.amp, "custom_fwd"):
    torch.amp.custom_fwd = lambda fwd=None, **kwargs: torch.cuda.amp.custom_fwd(fwd=fwd)

if not hasattr(torch.amp, "custom_bwd"):
    torch.amp.custom_bwd = lambda bwd=None, **kwargs: torch.cuda.amp.custom_bwd(bwd=bwd)

ENROLLMENT_AUDIO = Path("data/03_cleaned_audio/input1_DeepFilterNet3_pf.wav")
MODEL_DIR = Path("models/ecapa_tdnn_weights")
VOICEPRINT_PATH = MODEL_DIR / "target_voiceprint.pt"


def extract_voiceprint():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not ENROLLMENT_AUDIO.exists():
        print("Enrollment audio not found:", ENROLLMENT_AUDIO)
        return

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(MODEL_DIR),
        local_strategy=LocalStrategy.COPY
    )

    signal, sample_rate = torchaudio.load(str(ENROLLMENT_AUDIO))

    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000
        )
        signal = resampler(signal)

    embedding = classifier.encode_batch(signal)

    torch.save(embedding, VOICEPRINT_PATH)

    print("Voiceprint created successfully.")
    print("Saved at:", VOICEPRINT_PATH)
    print("Embedding shape:", embedding.shape)


if __name__ == "__main__":
    extract_voiceprint()