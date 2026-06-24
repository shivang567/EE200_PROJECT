# EE200_PROJECT
# Sonic Signatures — Audio Fingerprinting (EE200 Q3)

A small Shazam-style audio identification system. It fingerprints songs
by converting each track into a spectrogram, keeping only the strongest
time–frequency peaks ("constellation"), and hashing pairs of nearby
peaks into compact `(f1, f2, Δt)` keys. A query clip is identified by
finding the song whose hashes line up at a single, consistent time
offset against the database — the same core idea behind Shazam.

## Contents

- **`fingerprint_full.py`** — core fingerprinting engine: spectrogram
  generation, peak picking, paired hashing, and offset-voting matcher.
  Used to generate all the analysis in the report (DFT vs. spectrogram,
  single-peak vs. paired hashes, noise/pitch/tempo robustness).
- **`app.py`** — Streamlit web app wrapping the same engine, with three
  tabs:
  - **Library** — browse the indexed songs
  - **Identify** — upload one query clip, see its spectrogram,
    constellation map, offset histogram, match result, and a top-5
    candidate ranking
  - **Batch** — upload multiple clips at once, get a `results.csv`
    (`filename, prediction`)
- **`songs/`** — the 50-song library (Beatles/Queen) used to build the
  fingerprint database.
- **`fingerprint_cache.pkl`** — pre-built fingerprint database, cached
  so the app doesn't re-index on every launch.
- **`report.pdf` / `report.tex`** — full write-up: why a plain DFT
  fails, the spectrogram time–frequency tradeoff, why paired hashes beat
  single peaks, and robustness tests against noise, pitch shift, and
  time stretch.

## Live App

🔗 [Deployed Streamlit app](<<INSERT DEPLOYED STREAMLIT APP URL HERE>>)

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Key Findings

- A whole-track DFT has no time axis — the spectrogram is required.
- Pairing peaks into hashes is far more discriminative than single
  peaks (match score 419 vs. 71 on the same query).
- The system is robust to additive noise and to tempo changes, but
  fails almost immediately under pitch shifts as small as 0.25
  semitones, due to the exact-match frequency requirement in the hash
  key.

See `report.pdf` for the full analysis.
