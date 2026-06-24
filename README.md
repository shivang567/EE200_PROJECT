# Sonic Signatures — Audio Fingerprinting System

This project implements an audio fingerprinting and identification system,
developed as part of the EE200 — Signals and Systems coursework. The
system is inspired by the core algorithmic principles behind commercial
audio recognition services such as Shazam, and demonstrates how
fundamental signal processing concepts can be applied to solve a
real-world identification problem.

## Project Description

The system identifies a short audio clip by matching it against a
pre-built fingerprint database of songs, without ever comparing raw
waveforms directly. Instead, it relies on a constellation-based hashing
approach:

1. Each song is converted into a spectrogram using the Short-Time
   Fourier Transform, preserving both time and frequency information.
2. The strongest, most stable time–frequency peaks are extracted to
   form a sparse "constellation" map.
3. Nearby peaks are paired together and encoded into compact hash keys
   based on their frequencies and time gap.
4. A query clip is identified by checking which song's hashes align
   with the query's hashes at a single, consistent time offset.

This approach is significantly more robust and scalable than direct
waveform comparison, and the project includes a detailed analysis of
its performance under various real-world distortions.

## Key Components

- Core fingerprinting engine for spectrogram generation, peak
  detection, and hash-based matching
- A fingerprint database built from a 50-song library
- An interactive Streamlit web application supporting both
  single-clip and batch identification modes
- A full technical report covering methodology, design choices, and
  experimental results

## Live App

🔗 [Deployed Streamlit app](<<INSERT DEPLOYED STREAMLIT APP URL HERE>>)

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Evaluation

The system was tested for robustness against additive noise, pitch
shifting, and time stretching, with results and discussion documented
in the accompanying report.
