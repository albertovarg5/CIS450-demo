# W9A-demo.py
# Normalize, Trim Silence, 2x Speed Up
# LICENSE = MIT / BSD compatible

import numpy as np
from pedalboard.io import AudioFile

# Trim silence
def trim_silence(audio):
    chunk_size = 1024
    threshold = 0.1

    start = 0
    for i in range(0, len(audio), chunk_size):
        if np.max(np.abs(audio[i:i+chunk_size])) > threshold:
            start = i
            break

    end = len(audio)
    for i in range(len(audio)-chunk_size, 0, -chunk_size):
        if np.max(np.abs(audio[i:i+chunk_size])) > threshold:
            end = i + chunk_size
            break

    return audio[start:end]


# Normalize audio
def normalize_audio(audio):
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak
    return audio


# Speed change
def change_speed(audio, rate):
    indices = (np.arange(0, len(audio), rate)).astype(int)
    return audio[indices]


# Read file
with AudioFile("audio/output.wav") as f:
    samplerate = f.samplerate
    audio = f.read(f.frames)

if audio.ndim > 1:
    audio = audio[0]

# Normalize
normalized = normalize_audio(audio)

# Trim silence
trimmed = trim_silence(normalized)

# Speed up
sped_up = change_speed(trimmed, 2.0)

# Save file
with AudioFile("audio/processed.wav", "w", samplerate, 1) as out:
    out.write(sped_up)

print("Saved processed.wav")