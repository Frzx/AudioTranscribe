import torch
import torchaudio
from transformers import WhisperProcessor,BatchFeature

def audio_processor(waveform,sampling_rate):

    data = {}

    # Load the Whisper processor
    processor = WhisperProcessor.from_pretrained('openai/whisper-large-v2')

    # Check if the sample rate matches the model's expected rate
    # Whisper expects a sample rate of 16000, so resample if necessary
    target_sample_rate = processor.feature_extractor.sampling_rate
    if sampling_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # The processor expects input in numpy format, so convert it
    waveform = waveform.numpy()

    # Whisper expects mono audio, so convert stereo to mono if necessary
    if waveform.shape[0] == 2:  # Check if stereo
        waveform = torch.mean(torch.tensor(waveform), dim=0, keepdim=True).numpy()

    # Process the audio with WhisperProcessor
    x = processor(audio=waveform, sampling_rate=target_sample_rate, return_tensors="pt")


    if "input_features" in x:
        data["audio_values"] = x.input_features
    else:
        data["audio_values"] = x.input_values

    return BatchFeature(data=data)