import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.utils.fetching import LocalStrategy


# Torch/SpeechBrain AMP compatibility patch
if not hasattr(torch.amp, "custom_fwd"):
    def custom_fwd(fwd=None, **kwargs):
        return torch.cuda.amp.custom_fwd(fwd=fwd)

    torch.amp.custom_fwd = custom_fwd

if not hasattr(torch.amp, "custom_bwd"):
    def custom_bwd(bwd=None, **kwargs):
        return torch.cuda.amp.custom_bwd(bwd=bwd)

    torch.amp.custom_bwd = custom_bwd


MIXED_AUDIO = Path("data/03_cleaned_audio/mixed_audio2_DeepFilterNet3_pf.wav")
TARGET_VOICEPRINT = Path("models/ecapa_tdnn_weights/target_voiceprint.pt")

ECAPA_MODEL_DIR = Path("models/ecapa_tdnn_weights")
SEPFORMER_MODEL_DIR = Path("models/sepformer_weights")

OUTPUT_DIR = Path("data/04_isolated_audio")
FINAL_OUTPUT = OUTPUT_DIR / "target_speaker_isolated.wav"

TEMP_SOURCE_1 = OUTPUT_DIR / "separated_source_1.wav"
TEMP_SOURCE_2 = OUTPUT_DIR / "separated_source_2.wav"


def load_audio_for_ecapa(audio_path):
    signal, sample_rate = torchaudio.load(str(audio_path))

    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000
        )
        signal = resampler(signal)

    return signal


def get_embedding(classifier, audio_path):
    signal = load_audio_for_ecapa(audio_path)
    embedding = classifier.encode_batch(signal)
    return embedding.squeeze()


def load_audio_for_sepformer(audio_path):
    signal, sample_rate = torchaudio.load(str(audio_path))

    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    # SepFormer WHAMR model works at 8 kHz
    if sample_rate != 8000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=8000
        )
        signal = resampler(signal)
        sample_rate = 8000

    # Shape required by separate_batch: [batch, time]
    signal = signal.squeeze(0).unsqueeze(0)

    return signal, sample_rate


def isolate_target_speaker():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEPFORMER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not MIXED_AUDIO.exists():
        print("Mixed audio not found:", MIXED_AUDIO)
        return

    if not TARGET_VOICEPRINT.exists():
        print("Target voiceprint not found:", TARGET_VOICEPRINT)
        return

    print("Loading SepFormer separation model...")

    separator = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr",
        savedir=str(SEPFORMER_MODEL_DIR),
        local_strategy=LocalStrategy.COPY
    )

    print("Loading mixed audio with torchaudio...")

    mixed_signal, sepformer_sample_rate = load_audio_for_sepformer(MIXED_AUDIO)

    print("Separating mixed audio...")

    estimated_sources = separator.separate_batch(mixed_signal)

    # Expected shape: [batch, time, speakers]
    if estimated_sources.dim() != 3:
        print("Unexpected SepFormer output shape:", estimated_sources.shape)
        return

    num_speakers = estimated_sources.shape[2]

    if num_speakers < 2:
        print("SepFormer did not return 2 separated sources.")
        print("Output shape:", estimated_sources.shape)
        return

    source_1 = estimated_sources[0, :, 0].detach().cpu().unsqueeze(0)
    source_2 = estimated_sources[0, :, 1].detach().cpu().unsqueeze(0)

    torchaudio.save(str(TEMP_SOURCE_1), source_1, sepformer_sample_rate)
    torchaudio.save(str(TEMP_SOURCE_2), source_2, sepformer_sample_rate)

    print("Separated sources saved.")
    print("Source 1:", TEMP_SOURCE_1)
    print("Source 2:", TEMP_SOURCE_2)

    print("Loading ECAPA-TDNN speaker model...")

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ECAPA_MODEL_DIR),
        local_strategy=LocalStrategy.COPY
    )

    target_embedding = torch.load(TARGET_VOICEPRINT).squeeze()

    source_1_embedding = get_embedding(classifier, TEMP_SOURCE_1)
    source_2_embedding = get_embedding(classifier, TEMP_SOURCE_2)

    score_1 = F.cosine_similarity(
        target_embedding,
        source_1_embedding,
        dim=0
    ).item()

    score_2 = F.cosine_similarity(
        target_embedding,
        source_2_embedding,
        dim=0
    ).item()

    print("\nSimilarity Scores:")
    print("Source 1:", round(score_1, 4))
    print("Source 2:", round(score_2, 4))

    if score_1 > score_2:
        best_source = source_1
        best_score = score_1
        selected_source = "Source 1"
    else:
        best_source = source_2
        best_score = score_2
        selected_source = "Source 2"

    torchaudio.save(
        str(FINAL_OUTPUT),
        best_source,
        sepformer_sample_rate
    )

    print("\nSelected:", selected_source)
    print("Target speaker isolated successfully.")
    print("Saved at:", FINAL_OUTPUT)
    print("Best similarity score:", round(best_score, 4))


if __name__ == "__main__":
    isolate_target_speaker()