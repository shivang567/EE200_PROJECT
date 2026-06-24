import os
import io
import time
import pickle
import tempfile
import librosa
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from collections import defaultdict, Counter

# =====================================================
# PARAMETERS
# =====================================================

SONG_FOLDER = "songs"
SAMPLE_FOLDER = "samples"
CACHE_FILE = "fingerprint_cache.pkl"

SR = 22050
NPERSEG = 2048
NOVERLAP = 1024

PEAK_PERCENTILE = 99
NEIGHBORHOOD_SIZE = 20
FAN_VALUE = 10
MIN_SCORE = 5  # minimum aligned-hash cluster size required to call it a match

# Theme palette (kept in sync with .streamlit/config.toml)
BG = "#F5F6FA"
PANEL_BG = "#FFFFFF"
BORDER = "#E2E5EC"
TEXT = "#111827"
TEXT_DIM = "#6B7280"
AXIS_TEXT = "#374151"
CYAN = "#0891B2"
PURPLE = "#7C3AED"

CHART_CONFIG = {"displayModeBar": False}

# =====================================================
# AUDIO FUNCTIONS
# =====================================================

def load_audio(path):
    audio, sr = librosa.load(path, sr=SR, mono=True)
    return audio, sr

def compute_spectrogram(audio, sr):

    f, t, Sxx = spectrogram(
        audio,
        fs=sr,
        nperseg=NPERSEG,
        noverlap=NOVERLAP
    )

    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    return f, t, Sxx_db

# =====================================================
# PEAKS
# =====================================================

def find_peaks(Sxx):

    local_max = (
        Sxx ==
        maximum_filter(
            Sxx,
            size=NEIGHBORHOOD_SIZE
        )
    )

    threshold = np.percentile(
        Sxx,
        PEAK_PERCENTILE
    )

    detected = (
        local_max &
        (Sxx > threshold)
    )

    peaks = np.argwhere(detected)

    return peaks

# =====================================================
# HASHES
# =====================================================

def create_hashes(peaks, freqs, times):
    """
    `peaks` is expected to already be sorted by time (column index 1).
    Sorting is done once by the caller so this function can be reused
    for both database-building and query-matching without repeating an
    O(n log n) sort on every call.
    """

    hashes = []

    for i in range(len(peaks)):

        for j in range(
            1,
            FAN_VALUE + 1
        ):

            if i + j >= len(peaks):
                break

            p1 = peaks[i]
            p2 = peaks[i+j]

            f1 = int(freqs[p1[0]])
            f2 = int(freqs[p2[0]])

            t1 = times[p1[1]]
            t2 = times[p2[1]]

            dt = round(
                t2 - t1,
                1
            )

            if dt > 0:

                hashes.append(
                    (
                        (f1, f2, dt),
                        t1
                    )
                )

    return hashes

# =====================================================
# PERSISTENT FINGERPRINT CACHE
#
# Each song's hashes are cached to disk (fingerprint_cache.pkl) keyed
# by filename + last-modified time, so a restart only reprocesses
# songs that are new or changed.
# =====================================================

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

# =====================================================
# DATABASE
# =====================================================

@st.cache_resource
def build_database():

    os.makedirs(SONG_FOLDER, exist_ok=True)
    os.makedirs(SAMPLE_FOLDER, exist_ok=True)

    cache = load_cache()

    files_on_disk = [
        f for f in os.listdir(SONG_FOLDER)
        if f.endswith((".mp3", ".wav"))
    ]

    removed = [f for f in cache if f not in files_on_disk]
    for f in removed:
        del cache[f]

    to_process = []
    for fname in files_on_disk:
        path = os.path.join(SONG_FOLDER, fname)
        mtime = os.path.getmtime(path)
        if fname not in cache or cache[fname]["mtime"] != mtime:
            to_process.append((fname, mtime))

    if to_process:

        progress = st.progress(
            0,
            text=f"Fingerprinting {len(to_process)} new song(s)..."
        )

        for i, (fname, mtime) in enumerate(to_process):

            path = os.path.join(SONG_FOLDER, fname)

            audio, sr = load_audio(path)
            f_axis, t_axis, Sxx = compute_spectrogram(audio, sr)
            peaks = find_peaks(Sxx)
            peaks_sorted = peaks[np.argsort(peaks[:, 1])]
            hashes = create_hashes(peaks_sorted, f_axis, t_axis)

            cache[fname] = {
                "mtime": mtime,
                "hashes": hashes
            }

            progress.progress(
                (i + 1) / len(to_process),
                text=f"Fingerprinted {fname} ({i + 1}/{len(to_process)})"
            )

        progress.empty()

    if to_process or removed:
        save_cache(cache)

    database = defaultdict(list)

    for fname, data in cache.items():
        for h, offset in data["hashes"]:
            database[h].append((fname, offset))

    return database

# =====================================================
# IDENTIFICATION (instrumented with real per-stage timing
# and full candidate ranking, not just the single best match)
# =====================================================

def identify_song_detailed(path, database):

    timings = {}

    t0 = time.perf_counter()
    audio, sr = load_audio(path)
    t1 = time.perf_counter()
    timings["Audio Load"] = (t1 - t0) * 1000

    f_axis, t_axis, Sxx = compute_spectrogram(audio, sr)
    t2 = time.perf_counter()
    timings["Spectrogram"] = (t2 - t1) * 1000

    peaks = find_peaks(Sxx)
    t3 = time.perf_counter()
    timings["Constellation"] = (t3 - t2) * 1000

    hashes = create_hashes(peaks[np.argsort(peaks[:, 1])], f_axis, t_axis)
    t4 = time.perf_counter()
    timings["Hashing"] = (t4 - t3) * 1000

    votes = defaultdict(list)
    for h, q_time in hashes:
        if h not in database:
            continue
        for song, db_time in database[h]:
            offset = round(db_time - q_time, 1)
            votes[song].append(offset)
    t5 = time.perf_counter()
    timings["Lookup"] = (t5 - t4) * 1000

    candidates = []
    for song, offsets in votes.items():
        score = max(Counter(offsets).values())
        candidates.append((song, score, offsets))
    candidates.sort(key=lambda c: c[1], reverse=True)
    t6 = time.perf_counter()
    timings["Scoring"] = (t6 - t5) * 1000

    # A handful of hashes aligning at one offset can happen by pure
    # chance, especially on short or noisy clips. Require a minimum
    # cluster size before declaring a match, instead of always trusting
    # whichever candidate happens to score highest.
    if candidates and candidates[0][1] >= MIN_SCORE:
        best_song, best_score, best_offsets = candidates[0]
    else:
        best_song, best_score, best_offsets = None, 0, []

    duration = librosa.get_duration(y=audio, sr=sr)

    return {
        "song": best_song,
        "score": best_score,
        "offsets": best_offsets,
        "total_hashes": len(hashes),
        "candidates": [(c[0], c[1]) for c in candidates[:5]],
        "timings": timings,
        "f_axis": f_axis,
        "t_axis": t_axis,
        "Sxx": Sxx,
        "peaks": peaks,
        "hashes": hashes,
        "duration": duration,
    }

# =====================================================
# PLOTLY CHART HELPERS
# =====================================================

def plotly_spectrogram(t_axis, f_axis, Sxx, height=300):

    # Spectrogram dB values often have a handful of extreme outlier
    # peaks. Letting Plotly auto-scale to raw min/max crushes the rest
    # of the data into one saturated color (this was rendering as a
    # wash of red). Clipping to a percentile range restores contrast.
    zmin = float(np.percentile(Sxx, 5))
    zmax = float(np.percentile(Sxx, 99.5))

    fig = go.Figure(data=go.Heatmap(
        z=Sxx, x=t_axis, y=f_axis,
        colorscale="Viridis", zmin=zmin, zmax=zmax, showscale=False
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=45, r=10, t=10, b=40),
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=AXIS_TEXT, size=12),
        xaxis=dict(title="Time (s)", gridcolor=BORDER, zeroline=False, color=AXIS_TEXT),
        yaxis=dict(title="Frequency (Hz)", gridcolor=BORDER, zeroline=False, color=AXIS_TEXT),
    )

    return fig

def plotly_constellation(t_axis, f_axis, peaks, height=300, color=CYAN):

    fig = go.Figure(data=go.Scattergl(
        x=t_axis[peaks[:, 1]], y=f_axis[peaks[:, 0]],
        mode="markers", marker=dict(size=4, color=color, opacity=0.85)
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=45, r=10, t=10, b=40),
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=AXIS_TEXT, size=12),
        xaxis=dict(title="Time (s)", gridcolor=BORDER, zeroline=False, color=AXIS_TEXT),
        yaxis=dict(title="Frequency (Hz)", gridcolor=BORDER, zeroline=False, color=AXIS_TEXT),
    )

    return fig

def plotly_hash_pairs(peaks, f_axis, t_axis, sample_anchors=6, height=340):

    sorted_peaks = sorted(peaks.tolist(), key=lambda p: p[1])

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=t_axis[peaks[:, 1]], y=f_axis[peaks[:, 0]],
        mode="markers", marker=dict(size=4, color=TEXT_DIM, opacity=0.45),
        showlegend=False, hoverinfo="skip"
    ))

    if sorted_peaks:

        # Precompute id(peak) -> index once instead of calling
        # sorted_peaks.index(p1) (an O(n) linear scan) inside the loop.
        index_map = {id(p): idx for idx, p in enumerate(sorted_peaks)}

        step = max(1, len(sorted_peaks) // (sample_anchors * 3))
        anchors = sorted_peaks[::step][:sample_anchors]

        for p1 in anchors:
            i = index_map[id(p1)]
            for j in range(1, min(FAN_VALUE, len(sorted_peaks) - i - 1) + 1):
                p2 = sorted_peaks[i + j]
                fig.add_trace(go.Scatter(
                    x=[t_axis[p1[1]], t_axis[p2[1]]],
                    y=[f_axis[p1[0]], f_axis[p2[0]]],
                    mode="lines", line=dict(color=PURPLE, width=1.5),
                    opacity=0.6, showlegend=False, hoverinfo="skip"
                ))

        anchor_x = [t_axis[p[1]] for p in anchors]
        anchor_y = [f_axis[p[0]] for p in anchors]

        fig.add_trace(go.Scatter(
            x=anchor_x, y=anchor_y, mode="markers",
            marker=dict(size=11, color=CYAN, line=dict(width=1.5, color="white")),
            showlegend=False, hoverinfo="skip"
        ))

    fig.update_layout(
        height=height,
        margin=dict(l=55, r=20, t=15, b=45),
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=AXIS_TEXT, size=13),
        xaxis=dict(title="Time (s)", gridcolor=BORDER, zeroline=False, color=AXIS_TEXT, tickfont=dict(size=12)),
        yaxis=dict(title="Frequency (Hz)", gridcolor=BORDER, zeroline=False, color=AXIS_TEXT, tickfont=dict(size=12)),
    )

    return fig

def plotly_histogram(offsets, height=340):

    fig = go.Figure()

    if offsets:
        counts = Counter(offsets)
        xs = sorted(counts.keys())
        ys = [counts[x] for x in xs]
        best_idx = max(range(len(ys)), key=lambda i: ys[i])
        colors = [PURPLE] * len(xs)
        colors[best_idx] = CYAN

        fig.add_bar(x=xs, y=ys, marker_color=colors)
        fig.add_annotation(
            x=xs[best_idx], y=ys[best_idx],
            text=f"Peak alignment: {xs[best_idx]}s",
            showarrow=True, arrowcolor=CYAN, arrowwidth=2,
            font=dict(color=TEXT, size=13),
            bgcolor="#FFFFFF", bordercolor=CYAN, borderwidth=1,
            ay=-35
        )

    fig.update_layout(
        height=height,
        margin=dict(l=55, r=20, t=40, b=45),
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=AXIS_TEXT, size=13),
        xaxis=dict(title="Time Offset (s)", gridcolor=BORDER, zeroline=False, color=AXIS_TEXT, tickfont=dict(size=12)),
        yaxis=dict(title="Votes", gridcolor=BORDER, zeroline=False, color=AXIS_TEXT, tickfont=dict(size=12)),
    )

    return fig

def plotly_candidates(candidates, height=260):

    fig = go.Figure()

    if candidates:
        ordered = list(reversed(candidates))
        names = [c[0] for c in ordered]
        scores = [c[1] for c in ordered]
        colors = [PURPLE] * (len(ordered) - 1) + [CYAN]

        fig.add_bar(
            x=scores, y=names, orientation="h",
            marker_color=colors, text=scores, textposition="outside",
            textfont=dict(color=TEXT)
        )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=40, t=10, b=40),
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT, size=12),
        xaxis=dict(title="Cluster Score", gridcolor=BORDER, zeroline=False, color=AXIS_TEXT),
        yaxis=dict(gridcolor=BORDER, color=AXIS_TEXT),
    )

    return fig

# =====================================================
# SMALL HTML HELPERS
# =====================================================

def metric_card_html(label, value):
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>'
    )

# =====================================================
# STREAMLIT APP
# =====================================================

st.set_page_config(
    page_title="EE200 Audio Fingerprinting",
    page_icon=":material/graphic_eq:",
    layout="wide"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, .stApp {
        color-scheme: light only;
        background: #F5F6FA;
    }

    .stApp, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] li, label {
        color: #111827;
    }

    h1, h2, h3, h4, h5 { font-family: 'Space Grotesk', sans-serif !important; color: #111827 !important; }

    [data-testid="stCaptionContainer"], .stCaption, small {
        color: #6B7280 !important;
    }

    .stButton button {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #E2E5EC !important;
    }
    .stButton button:hover {
        border-color: #0891B2 !important;
        color: #0891B2 !important;
    }
    .stButton button p { color: inherit !important; }

    .hero {
        padding: 2rem 2.5rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(8,145,178,0.08), rgba(124,58,237,0.08));
        border: 1px solid #E2E5EC;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #0891B2, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        color: #6B7280;
        margin-top: 4px;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
    }

    .metric-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 16px 14px;
        border: 1px solid #E2E5EC;
        text-align: center;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);
    }
    .metric-label {
        color: #6B7280;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        color: #111827;
        font-size: 1.35rem;
        font-weight: 600;
        margin-top: 4px;
    }

    .match-card {
        background: linear-gradient(135deg, #0891B2, #7C3AED);
        border-radius: 24px;
        padding: 28px 32px;
        margin: 0.25rem 0 1.5rem 0;
    }
    .match-card.no-match { background: #F1F2F6; border: 1px solid #E2E5EC; }
    .match-label { color: rgba(255,255,255,0.85); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em; }
    .match-card.no-match .match-label { color: #6B7280; }
    .match-song {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.9rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;
    }
    .match-card.no-match .match-song { color: #111827; }
    .match-score { font-family: 'JetBrains Mono', monospace; color: rgba(255,255,255,0.9); font-size: 0.95rem; }
    .match-card.no-match .match-score { color: #6B7280; }

    [data-testid="stTabs"] button { border-radius: 10px 10px 0 0; font-family: 'Space Grotesk', sans-serif; color: #111827; }
    div[data-testid="stDataFrame"] { border: 1px solid #E2E5EC; border-radius: 12px; }
    .stButton button { border-radius: 10px; }
    [data-testid="stFileUploaderDropzone"] { background: #FFFFFF; border: 1px dashed #CBD2DE; border-radius: 14px; }

    /* ---- extra hardening: native widgets that tend to keep
       Streamlit's default dark styling even after base="light" ---- */

    [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    .main, .block-container {
        background-color: #F5F6FA !important;
        color: #111827 !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid #E2E5EC !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #6B7280 !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: #0891B2 !important;
        background-color: #FFFFFF !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: #0891B2 !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #374151 !important;
        fill: #374151 !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #E2E5EC !important;
    }
    [data-testid="stFileUploaderDropzone"] small {
        color: #6B7280 !important;
    }

    /* File chips that appear after upload (the dark/black squares
       in the screenshot are the file-type icon swatches and the
       remove "x" icon — force them onto the light palette too) */
    [data-testid="stFileUploaderFile"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #E2E5EC !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploaderFile"] * { color: #111827 !important; }
    [data-testid="stFileUploaderFile"] svg,
    [data-testid="stFileUploaderFileIcon"] svg,
    [data-testid="stFileUploaderDropzone"] svg {
        fill: #0891B2 !important;
        color: #0891B2 !important;
        stroke: #0891B2 !important;
    }
    [data-testid="stFileUploaderFileIcon"] {
        background-color: #E0F2FE !important;
        border-radius: 6px !important;
    }
    [data-testid="stFileUploaderFile"] button,
    [data-testid="stFileUploaderFile"] [data-testid="stBaseButton-icon"],
    [data-testid="stFileUploaderFile"] [role="button"] {
        background-color: transparent !important;
    }
    [data-testid="stFileUploaderFile"] button svg,
    [data-testid="stFileUploaderFile"] [data-testid="stBaseButton-icon"] svg {
        fill: #6B7280 !important;
        stroke: #6B7280 !important;
    }
    [data-testid="stFileUploaderFile"] button:hover svg,
    [data-testid="stFileUploaderFile"] [data-testid="stBaseButton-icon"]:hover svg {
        fill: #EF4444 !important;
        stroke: #EF4444 !important;
    }

    [data-testid="stRadio"] label, [data-testid="stCheckbox"] label,
    [data-testid="stSelectbox"] label, [data-testid="stMultiSelect"] label {
        color: #111827 !important;
    }
    [data-baseweb="select"] > div, [data-baseweb="popover"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border-color: #E2E5EC !important;
    }
    [data-baseweb="menu"], [role="listbox"] {
        background-color: #FFFFFF !important;
    }
    [role="option"] { color: #111827 !important; }

    [data-testid="stProgress"] > div > div {
        background-color: #E2E5EC !important;
    }
    [data-testid="stProgress"] > div > div > div {
        background-color: #0891B2 !important;
    }

    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E5EC !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary { color: #111827 !important; }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-color: #E2E5EC !important;
    }

    div[data-testid="stDataFrame"] * {
        color: #111827 !important;
    }
    div[data-testid="stDataFrame"] [role="columnheader"] {
        background-color: #F5F6FA !important;
    }

    [data-testid="stAlert"] { color: #111827 !important; }
    [data-testid="stAlert"] p { color: #111827 !important; }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
    }

    code { color: #0891B2 !important; background-color: #F1F2F6 !important; }
    a { color: #0891B2 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">EE200 Audio Fingerprinting</div>'
    '<div class="hero-sub">Signals, Systems &amp; Networks — Acoustic Fingerprint Identification</div>'
    '</div>',
    unsafe_allow_html=True
)

database = build_database()

left_col, right_col = st.columns([1, 4])

with left_col:

    _metrics_cache = load_cache()
    _indexed_tracks = len(_metrics_cache)
    _total_hash_records = sum(len(d["hashes"]) for d in _metrics_cache.values())

    st.markdown("##### Database Metrics")

    st.markdown(metric_card_html("Indexed Tracks", f"{_indexed_tracks}"), unsafe_allow_html=True)
    st.markdown(metric_card_html("Total Hash Records", f"{_total_hash_records:,}"), unsafe_allow_html=True)

    st.divider()

    st.caption(
        f"Window {NPERSEG} \u00b7 Overlap {NOVERLAP} \u00b7 "
        f"Fan-out {FAN_VALUE} \u00b7 Peak percentile {PEAK_PERCENTILE}"
    )

with right_col:

    tab_library, tab_identify, tab_batch = st.tabs(
        ["Library", "Identify", "Batch"]
    )

    # =====================================================
    # LIBRARY TAB
    # =====================================================

    with tab_library:

        st.markdown("##### Add to Library")

        new_uploads = st.file_uploader(
            "Upload songs to add permanently to the library",
            type=["mp3", "wav"],
            accept_multiple_files=True,
            key="library_uploader"
        )

        if new_uploads:

            already_added = st.session_state.setdefault("library_added_filenames", set())
            to_add = [f for f in new_uploads if f.name not in already_added]

            if to_add:

                os.makedirs(SONG_FOLDER, exist_ok=True)
                cache = load_cache()

                progress = st.progress(0, text=f"Adding {len(to_add)} song(s) to the library...")
                added_names = []

                for i, file in enumerate(to_add):

                    try:
                        dest_path = os.path.join(SONG_FOLDER, file.name)
                        with open(dest_path, "wb") as fh:
                            fh.write(file.getbuffer())

                        mtime = os.path.getmtime(dest_path)

                        audio, sr = load_audio(dest_path)
                        f_axis, t_axis, Sxx = compute_spectrogram(audio, sr)
                        peaks = find_peaks(Sxx)
                        peaks_sorted = peaks[np.argsort(peaks[:, 1])]
                        hashes = create_hashes(peaks_sorted, f_axis, t_axis)

                        cache[file.name] = {"mtime": mtime, "hashes": hashes}

                        # Update the already-built in-memory database too, so the
                        # new song is matchable immediately without restarting the app.
                        for h, offset in hashes:
                            database[h].append((file.name, offset))

                        added_names.append(file.name)

                    except Exception as e:
                        st.warning(f"Could not process {file.name}: {e}")

                    progress.progress((i + 1) / len(to_add), text=f"Added {file.name} ({i + 1}/{len(to_add)})")

                save_cache(cache)
                progress.empty()

                already_added.update(f.name for f in to_add)

                if added_names:
                    st.success(f"Added {len(added_names)} song(s) to the library: {', '.join(added_names)}")

        st.divider()

        cache = load_cache()

        if not cache:
            st.info(f"No songs indexed yet. Upload some above, or add audio files to the **{SONG_FOLDER}/** folder and restart the app.")
        else:
            lc1, lc2 = st.columns([3, 1])
            with lc1:
                total_hashes_lib = sum(len(d["hashes"]) for d in cache.values())
                st.caption(f"{len(cache)} song(s) · {total_hashes_lib:,} hashes total")
            with lc2:
                library_hash_rows = [
                    [fname, f1, f2, dt, t1]
                    for fname, d in cache.items()
                    for (f1, f2, dt), t1 in d["hashes"]
                ]
                library_hashes_df = pd.DataFrame(
                    library_hash_rows,
                    columns=["filename", "freq1_hz", "freq2_hz", "delta_t_s", "anchor_time_s"]
                )
                st.download_button(
                    "Download library hashes (CSV)",
                    library_hashes_df.to_csv(index=False),
                    file_name="library_hashes.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            names = list(cache.keys())
            cols = st.columns(3)

            for i, fname in enumerate(names):

                data = cache[fname]

                anchors = [(t1, f1) for (f1, f2, dt), t1 in data["hashes"]]
                if len(anchors) > 400:
                    step = max(1, len(anchors) // 400)
                    anchors = anchors[::step]

                xs = [a[0] for a in anchors]
                ys = [a[1] for a in anchors]

                with cols[i % 3]:
                    with st.container(border=True):

                        if len(xs) < 5:
                            st.markdown(
                                '<div style="height:130px; display:flex; align-items:center; '
                                'justify-content:center; color:#6B7280; font-size:0.8rem; text-align:center;">'
                                'Too few peaks<br>to visualize'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            fig, ax = plt.subplots(figsize=(3, 1.3))
                            fig.patch.set_facecolor("#FFFFFF")
                            ax.set_facecolor("#FFFFFF")
                            ax.scatter(xs, ys, s=2, color=CYAN, alpha=0.7)
                            ax.axis("off")
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
                            plt.close(fig)
                            buf.seek(0)
                            st.image(buf, use_container_width=True)

                        st.markdown(f"**{fname}**")
                        st.caption(f"{len(data['hashes']):,} hashes")

                        if st.button("View Profile Mapping", key=f"profile_{fname}", use_container_width=True):
                            st.session_state["library_profile_view"] = fname

            # ---- full profile view for whichever song was last clicked ----
            profile_name = st.session_state.get("library_profile_view")

            if profile_name and profile_name in cache:

                profile_data = cache[profile_name]
                profile_anchors = [(t1, f1) for (f1, f2, dt), t1 in profile_data["hashes"]]

                st.divider()

                ph1, ph2 = st.columns([5, 1])
                with ph1:
                    st.markdown(f"##### Profile Mapping — {profile_name}")
                with ph2:
                    if st.button("Close", key="close_profile_view", use_container_width=True):
                        del st.session_state["library_profile_view"]
                        st.rerun()

                pm1, pm2 = st.columns(2)
                with pm1:
                    st.markdown(metric_card_html("Total Hashes", f"{len(profile_data['hashes']):,}"), unsafe_allow_html=True)
                with pm2:
                    st.markdown(metric_card_html("Unique Anchor Points", f"{len(set(profile_anchors)):,}"), unsafe_allow_html=True)

                if profile_anchors:
                    px = [a[0] for a in profile_anchors]
                    py = [a[1] for a in profile_anchors]

                    profile_fig = go.Figure(go.Scattergl(
                        x=px, y=py, mode="markers",
                        marker=dict(size=3, color=CYAN, opacity=0.6)
                    ))
                    profile_fig.update_layout(
                        height=380,
                        margin=dict(l=45, r=10, t=10, b=40),
                        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
                        font=dict(color=TEXT_DIM, size=11),
                        xaxis=dict(title="Time (s)", gridcolor=BORDER, zeroline=False),
                        yaxis=dict(title="Frequency (Hz)", gridcolor=BORDER, zeroline=False),
                    )
                    st.plotly_chart(
                        profile_fig, use_container_width=True,
                        config=CHART_CONFIG, key=f"profile_chart_{profile_name}"
                    )
                else:
                    st.info("No hash anchors available to plot for this song.")

    # =====================================================
    # IDENTIFY TAB
    # =====================================================

    with tab_identify:

        sample_files = []
        if os.path.isdir(SAMPLE_FOLDER):
            sample_files = [
                f for f in os.listdir(SAMPLE_FOLDER)
                if f.endswith((".mp3", ".wav"))
            ]

        if sample_files:
            st.markdown("##### Quick Test")
            for fname in sample_files:
                sc1, sc2, sc3 = st.columns([3, 3, 1])
                with sc1:
                    st.markdown(f"`{fname}`")
                with sc2:
                    st.audio(os.path.join(SAMPLE_FOLDER, fname))
                with sc3:
                    if st.button("Try", key=f"try_{fname}", use_container_width=True):
                        full_path = os.path.join(SAMPLE_FOLDER, fname)
                        with open(full_path, "rb") as fh:
                            st.session_state["identify_query_bytes"] = fh.read()
                        st.session_state["identify_query"] = full_path
                        st.session_state["identify_query_name"] = fname
            st.divider()

        uploaded = st.file_uploader(
            "Upload Query Audio",
            type=["mp3", "wav"],
            key="identify_uploader"
        )

        if uploaded:
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                query_path = tmp.name
            st.session_state["identify_query"] = query_path
            st.session_state["identify_query_name"] = uploaded.name
            st.session_state["identify_query_bytes"] = uploaded.getvalue()

        if "identify_query" in st.session_state:

            q_path = st.session_state["identify_query"]
            q_name = st.session_state["identify_query_name"]

            cache_key = f"result_{q_name}"
            if cache_key not in st.session_state:
                st.session_state[cache_key] = identify_song_detailed(q_path, database)
            result = st.session_state[cache_key]
            confidence = (result["score"] / result["total_hashes"] * 100) if result["total_hashes"] else 0

            st.divider()
            st.markdown(f"**{q_name}**")
            st.audio(st.session_state["identify_query_bytes"])

            # ---- processing pipeline timing ----
            st.markdown("##### Processing Pipeline")
            timing_cols = st.columns(len(result["timings"]))
            for col, (stage, ms) in zip(timing_cols, result["timings"].items()):
                with col:
                    st.markdown(metric_card_html(stage, f"{ms:.0f} ms"), unsafe_allow_html=True)

            # ---- match result hero card ----
            st.markdown("##### Match Result")
            if result["song"]:
                st.markdown(
                    f'<div class="match-card">'
                    f'<div class="match-label">MATCH FOUND</div>'
                    f'<div class="match-song">{result["song"]}</div>'
                    f'<div class="match-score">Cluster score {result["score"]} · {confidence:.0f}% confidence</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="match-card no-match">'
                    '<div class="match-label">NO MATCH</div>'
                    '<div class="match-song">Not found in library</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

            # ---- candidate ranking ----
            if result["candidates"]:
                st.markdown("##### Candidate Ranking")
                st.plotly_chart(
                    plotly_candidates(result["candidates"]),
                    use_container_width=True, config=CHART_CONFIG, key="candidate_ranking"
                )

            # ---- explainability ----
            st.markdown("##### How It Works")

            e1, e2 = st.columns(2)
            with e1:
                st.caption("Step 1 · Spectrogram")
                st.plotly_chart(
                    plotly_spectrogram(result["t_axis"], result["f_axis"], result["Sxx"]),
                    use_container_width=True, config=CHART_CONFIG, key="exp_spectrogram"
                )
            with e2:
                st.caption("Step 1 · Constellation Map")
                st.plotly_chart(
                    plotly_constellation(result["t_axis"], result["f_axis"], result["peaks"]),
                    use_container_width=True, config=CHART_CONFIG, key="exp_constellation"
                )

            st.caption("Step 2 · Constellation → Hash Pairs (sample anchors fanning out to nearby peaks)")
            st.plotly_chart(
                plotly_hash_pairs(result["peaks"], result["f_axis"], result["t_axis"]),
                use_container_width=True, config=CHART_CONFIG, key="exp_hashpairs"
            )

            st.caption("Step 3 · Alignment Histogram (matching hashes vote on a time offset; the tallest bar wins)")
            st.plotly_chart(
                plotly_histogram(result["offsets"]),
                use_container_width=True, config=CHART_CONFIG, key="exp_histogram"
            )

    # =====================================================
    # BATCH TAB
    # =====================================================

    with tab_batch:

        files = st.file_uploader(
            "Upload Multiple Audio Files",
            type=["mp3", "wav"],
            accept_multiple_files=True,
            key="batch_uploader"
        )

        if files:

            rows = []
            hash_rows = []

            progress = st.progress(0, text="Matching uploaded tracks...")

            for i, file in enumerate(files):

                suffix = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.getbuffer())
                    path = tmp.name

                result = identify_song_detailed(path, database)
                confidence = (result["score"] / result["total_hashes"] * 100) if result["total_hashes"] else 0
                duration = result.get("duration", 0)

                rows.append([
                    file.name,
                    os.path.splitext(result["song"])[0] if result["song"] else "No match",
                ])
                df = pd.DataFrame(rows, columns=["filename", "prediction"])

                for (f1, f2, dt), t1 in result["hashes"]:
                    hash_rows.append([file.name, f1, f2, dt, t1])

                progress.progress((i + 1) / len(files), text=f"Matched {file.name}")

            progress.empty()

            df = pd.DataFrame(rows, columns=["filename", "prediction"])
            st.dataframe(df, use_container_width=True, hide_index=True)

            hashes_df = pd.DataFrame(
                hash_rows,
                columns=["filename", "freq1_hz", "freq2_hz", "delta_t_s", "anchor_time_s"]
            )

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download predictions (CSV)",
                    df.to_csv(index=False),
                    file_name="batch_results.csv",
                    mime="text/csv"
                )
            with d2:
                st.download_button(
                    "Download hashes (CSV)",
                    hashes_df.to_csv(index=False),
                    file_name="batch_hashes.csv",
                    mime="text/csv"
                )