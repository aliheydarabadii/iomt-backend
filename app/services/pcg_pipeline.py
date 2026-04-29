from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pywt
from scipy.io import wavfile
from scipy.io.wavfile import write as wav_write
from scipy.ndimage import uniform_filter1d
from scipy.signal import (
    butter,
    filtfilt,
    find_peaks,
    hilbert,
    iirnotch,
    savgol_filter,
)
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    import librosa
except Exception:  # pragma: no cover - fallback is tested through runtime behavior
    librosa = None


@dataclass
class PCGConfig:
    lowcut: float = 25.0
    highcut: float = 200.0
    notch_freqs: tuple[float, ...] = (50.0, 100.0, 150.0)
    notch_q: float = 35.0
    filter_order: int = 4
    wavelet: str = "db6"
    wavelet_level: int = 4
    envelope_cutoff: float = 8.0
    min_peak_dist: float = 0.25
    bpm_min: int = 40
    bpm_max: int = 200
    normal_ranges: dict[str, tuple[float, float]] | None = None
    murmur_grade_thresholds: list[float] | None = None

    def __post_init__(self) -> None:
        if self.normal_ranges is None:
            self.normal_ranges = {
                "s1_duration_ms": (50, 200),
                "s2_duration_ms": (40, 150),
                "systolic_ms": (150, 450),
                "diastolic_ms": (150, 1200),
                "s1_s2_amp_ratio": (0.5, 3.0),
                "heart_rate_bpm": (45, 180),
            }
        if self.murmur_grade_thresholds is None:
            self.murmur_grade_thresholds = [0.15, 0.30, 0.50, 0.70, 0.90]


def _apply_zero_phase_filter(data: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    if len(data) < 8:
        return data.astype(float)
    padlen = 3 * max(len(a), len(b))
    if len(data) <= padlen:
        return data.astype(float)
    return filtfilt(b, a, data)


def multi_notch_filter(
    data: np.ndarray,
    fs: float,
    freqs: tuple[float, ...] = (50.0, 100.0, 150.0),
    q: float = 35.0,
) -> np.ndarray:
    filtered = data.copy().astype(float)
    nyq = 0.5 * fs
    for freq in freqs:
        if freq < nyq - 1.0:
            b, a = iirnotch(freq / nyq, q)
            filtered = _apply_zero_phase_filter(filtered, b, a)
    return filtered


def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    nyq = 0.5 * fs
    if nyq <= 1.0:
        return data.astype(float)
    safe_lowcut = max(0.1, min(lowcut, nyq - 2.0))
    safe_highcut = min(highcut, nyq - 1.0)
    if safe_highcut <= safe_lowcut:
        return data.astype(float)
    b, a = butter(order, [safe_lowcut / nyq, safe_highcut / nyq], btype="band")
    return _apply_zero_phase_filter(data, b, a)


def lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    safe_cutoff = min(cutoff, nyq - 1.0)
    if safe_cutoff <= 0:
        return data.astype(float)
    b, a = butter(order, safe_cutoff / nyq, btype="low")
    return _apply_zero_phase_filter(data, b, a)


def wavelet_denoise(
    data: np.ndarray,
    wavelet: str = "db6",
    level: int = 4,
    thresh_scale: float = 0.6,
) -> np.ndarray:
    if len(data) < 8:
        return data.astype(float)

    wavelet_def = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(data), wavelet_def.dec_len)
    if max_level < 1:
        return data.astype(float)

    level = min(level, max_level)
    coeffs = pywt.wavedec(data, wavelet, level=level)
    if not coeffs or len(coeffs[-1]) == 0:
        return data.astype(float)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
    uthresh = thresh_scale * sigma * np.sqrt(2 * np.log(len(data)))
    denoised = [coeffs[0]] + [
        pywt.threshold(component, value=uthresh, mode="soft")
        for component in coeffs[1:]
    ]
    return pywt.waverec(denoised, wavelet)[: len(data)]


def shannon_envelope(signal: np.ndarray, sr: int, cutoff: float = 8.0) -> np.ndarray:
    norm = signal / (np.max(np.abs(signal)) + 1e-10)
    eps = 1e-10
    shannon_energy = -norm**2 * np.log(norm**2 + eps)
    envelope = lowpass_filter(shannon_energy, cutoff, sr)
    return np.maximum(envelope, 0)


def estimate_sound_width(
    envelope: np.ndarray,
    peak_idx: int,
    sr: int,
    max_width_s: float = 0.15,
) -> tuple[int, int]:
    half_height = envelope[peak_idx] * 0.4
    max_width = int(sr * max_width_s)

    left = peak_idx
    for index in range(peak_idx, max(0, peak_idx - max_width), -1):
        if envelope[index] < half_height:
            left = index
            break

    right = peak_idx
    for index in range(peak_idx, min(len(envelope), peak_idx + max_width)):
        if envelope[index] < half_height:
            right = index
            break

    return left, right


def segment_heart_sounds(
    filtered: np.ndarray,
    envelope: np.ndarray,
    s1_peaks: np.ndarray,
    s2_peaks: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    n_samples = len(filtered)
    states = np.full(n_samples, 3, dtype=int)

    s1_bounds = []
    for peak in s1_peaks:
        left, right = estimate_sound_width(envelope, int(peak), sr, 0.08)
        states[left:right] = 0
        s1_bounds.append((left, right))

    s2_bounds = []
    for peak in s2_peaks:
        left, right = estimate_sound_width(envelope, int(peak), sr, 0.07)
        states[left:right] = 2
        s2_bounds.append((left, right))

    for _, s1_end in s1_bounds:
        next_s2 = [s2_left for s2_left, _ in s2_bounds if s2_left > s1_end]
        if next_s2 and (next_s2[0] - s1_end) / sr < 0.40:
            states[s1_end:next_s2[0]] = 1

    for _, s2_end in s2_bounds:
        next_s1 = [s1_left for s1_left, _ in s1_bounds if s1_left > s2_end]
        if next_s1:
            states[s2_end:next_s1[0]] = 3

    state_names = ["S1", "Systole", "S2", "Diastole"]
    segments: list[tuple[str, int, int]] = []
    index = 0
    while index < n_samples:
        end = index
        while end < n_samples and states[end] == states[index]:
            end += 1
        segments.append((state_names[int(states[index])], index, end))
        index = end

    return states, segments


def spectral_centroid(values: np.ndarray, sr: int) -> float:
    if len(values) < 8:
        return 0.0
    magnitude = np.abs(np.fft.rfft(values))
    freqs = np.fft.rfftfreq(len(values), 1.0 / sr)
    return float(np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10))


def zero_crossing_rate(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.sum(np.diff(np.sign(values)) != 0) / len(values))


def excess_kurtosis(values: np.ndarray) -> float:
    if len(values) < 4:
        return 0.0
    mean_value = np.mean(values)
    std_value = np.std(values)
    if std_value < 1e-10:
        return 0.0
    return float(np.mean(((values - mean_value) / std_value) ** 4) - 3)


def extract_cycle_features(
    filtered: np.ndarray,
    segments: list[tuple[str, int, int]],
    sr: int,
) -> list[dict[str, float]]:
    cycles = []
    index = 0
    while index + 3 < len(segments):
        if (
            segments[index][0] == "S1"
            and segments[index + 1][0] == "Systole"
            and segments[index + 2][0] == "S2"
            and segments[index + 3][0] == "Diastole"
        ):
            s1_start, s1_end = segments[index][1], segments[index][2]
            sys_start, sys_end = segments[index + 1][1], segments[index + 1][2]
            s2_start, s2_end = segments[index + 2][1], segments[index + 2][2]
            dia_start, dia_end = segments[index + 3][1], segments[index + 3][2]

            s1_signal = filtered[s1_start:s1_end]
            sys_signal = filtered[sys_start:sys_end]
            s2_signal = filtered[s2_start:s2_end]
            dia_signal = filtered[dia_start:dia_end]
            full_cycle = filtered[s1_start:dia_end]

            if len(s1_signal) < 3 or len(s2_signal) < 3 or len(full_cycle) < 10:
                index += 1
                continue

            s1_rms = np.sqrt(np.mean(s1_signal**2))
            s2_rms = np.sqrt(np.mean(s2_signal**2))
            sys_rms = np.sqrt(np.mean(sys_signal**2)) if len(sys_signal) > 0 else 0.0
            dia_rms = np.sqrt(np.mean(dia_signal**2)) if len(dia_signal) > 0 else 0.0

            s1_duration = len(s1_signal) / sr * 1000
            s2_duration = len(s2_signal) / sr * 1000
            systolic = (s2_start - s1_start) / sr * 1000
            diastolic = (dia_end - s2_start) / sr * 1000
            cycle_duration = (dia_end - s1_start) / sr * 1000
            heart_rate = 60000.0 / cycle_duration if cycle_duration > 0 else 0.0

            if librosa is not None and len(full_cycle) >= 16:
                nfft = min(64, len(full_cycle))
                try:
                    mfccs = librosa.feature.mfcc(y=full_cycle, sr=sr, n_mfcc=8, n_fft=nfft)
                    mfcc_means = np.mean(mfccs, axis=1)
                except Exception:
                    mfcc_means = np.zeros(8)
            else:
                mfcc_means = np.zeros(8)

            cycle = {
                "s1_duration_ms": s1_duration,
                "s2_duration_ms": s2_duration,
                "systolic_ms": systolic,
                "diastolic_ms": diastolic,
                "cycle_duration_ms": cycle_duration,
                "heart_rate_bpm": heart_rate,
                "sd_ratio": systolic / (diastolic + 1e-10),
                "s1_rms": s1_rms,
                "s2_rms": s2_rms,
                "s1_s2_amp_ratio": s1_rms / (s2_rms + 1e-10),
                "energy_concentration": (
                    (np.sum(s1_signal**2) + np.sum(s2_signal**2))
                    / (np.sum(full_cycle**2) + 1e-10)
                ),
                "s1_zcr": zero_crossing_rate(s1_signal),
                "s2_zcr": zero_crossing_rate(s2_signal),
                "s1_kurtosis": excess_kurtosis(s1_signal),
                "s2_kurtosis": excess_kurtosis(s2_signal),
                "s1_centroid": spectral_centroid(s1_signal, sr),
                "s2_centroid": spectral_centroid(s2_signal, sr),
                "sys_noise_ratio": sys_rms / (s1_rms + 1e-10),
                "dia_noise_ratio": dia_rms / (s1_rms + 1e-10),
                **{f"mfcc_{item_index}": value for item_index, value in enumerate(mfcc_means)},
                "_s1_start": s1_start,
                "_s1_end": s1_end,
                "_sys_start": sys_start,
                "_sys_end": sys_end,
                "_s2_start": s2_start,
                "_s2_end": s2_end,
                "_dia_start": dia_start,
                "_dia_end": dia_end,
            }
            cycles.append(cycle)
            index += 4
        else:
            index += 1
    return cycles


def murmur_grade(ratio: float, thresholds: list[float]) -> int:
    for index, threshold in enumerate(thresholds):
        if ratio < threshold:
            return index
    return 6


def detect_murmur(
    filtered: np.ndarray,
    cycle: dict[str, float],
    sr: int,
    thresholds: list[float],
) -> dict[str, Any]:
    systolic_signal = filtered[int(cycle["_sys_start"]) : int(cycle["_sys_end"])]
    diastolic_signal = filtered[int(cycle["_dia_start"]) : int(cycle["_dia_end"])]
    s1_rms = cycle["s1_rms"]

    result: dict[str, Any] = {
        "systolic_murmur": False,
        "diastolic_murmur": False,
        "systolic_grade": 0,
        "diastolic_grade": 0,
        "systolic_ratio": 0.0,
        "diastolic_ratio": 0.0,
    }

    if len(systolic_signal) > 4:
        systolic_ratio = np.sqrt(np.mean(systolic_signal**2)) / (s1_rms + 1e-10)
        result["systolic_ratio"] = systolic_ratio
        result["systolic_grade"] = murmur_grade(systolic_ratio, thresholds)
        result["systolic_murmur"] = systolic_ratio > thresholds[0]
        if len(systolic_signal) > 15:
            envelope = uniform_filter1d(
                np.abs(hilbert(systolic_signal)),
                max(3, len(systolic_signal) // 5),
            )
            peak_position = np.argmax(envelope) / len(envelope)
            result["sys_diamond"] = 0.25 < peak_position < 0.75

    if len(diastolic_signal) > 4:
        diastolic_ratio = np.sqrt(np.mean(diastolic_signal**2)) / (s1_rms + 1e-10)
        result["diastolic_ratio"] = diastolic_ratio
        result["diastolic_grade"] = murmur_grade(diastolic_ratio, thresholds)
        result["diastolic_murmur"] = diastolic_ratio > thresholds[0]
        if len(diastolic_signal) > 15:
            envelope = uniform_filter1d(
                np.abs(hilbert(diastolic_signal)),
                max(3, len(diastolic_signal) // 5),
            )
            q1 = np.mean(envelope[: len(envelope) // 4])
            q4 = np.mean(envelope[3 * len(envelope) // 4 :])
            result["dia_decrescendo"] = q1 > 1.5 * q4 if q4 > 0 else False

    return result


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def run_pcg_pipeline(
    filename: str,
    config: PCGConfig | None = None,
    save_filtered_wav: bool = False,
    output_filename: str | None = None,
    include_signals: bool = False,
) -> dict[str, Any]:
    cfg = config or PCGConfig()

    sample_rate, raw_data = wavfile.read(filename)
    if raw_data.ndim > 1:
        raw_data = raw_data[:, 0]

    data = raw_data.astype(np.float64)
    data = data - np.mean(data)
    n_samples = len(data)
    duration = n_samples / sample_rate if sample_rate > 0 else 0.0
    nyquist = sample_rate / 2.0
    time_axis = np.linspace(0, duration, n_samples, endpoint=False)

    notched = multi_notch_filter(data, sample_rate, cfg.notch_freqs, cfg.notch_q)
    bandpassed = bandpass_filter(notched, cfg.lowcut, cfg.highcut, sample_rate, cfg.filter_order)
    denoised = wavelet_denoise(
        bandpassed,
        cfg.wavelet,
        cfg.wavelet_level,
        thresh_scale=0.6,
    )[:n_samples]

    if len(denoised) >= 11:
        window_length = min(11, len(denoised) if len(denoised) % 2 == 1 else len(denoised) - 1)
        filtered = savgol_filter(denoised, window_length=window_length, polyorder=3)
    else:
        filtered = denoised

    envelope = shannon_envelope(filtered, sample_rate, cfg.envelope_cutoff)

    threshold = np.percentile(envelope, 75) if len(envelope) else 0.0
    peaks, _ = find_peaks(
        envelope,
        height=threshold,
        distance=max(1, int(sample_rate * cfg.min_peak_dist)),
        prominence=threshold * 0.3 if threshold > 0 else None,
    )

    s1_peaks: list[int] = []
    s2_peaks: list[int] = []
    if len(peaks) >= 3:
        peak_times = peaks / sample_rate
        for index in range(len(peaks) - 2):
            interval_1 = peak_times[index + 1] - peak_times[index]
            interval_2 = peak_times[index + 2] - peak_times[index + 1]
            if interval_1 < interval_2:
                if int(peaks[index]) not in s1_peaks:
                    s1_peaks.append(int(peaks[index]))
                if int(peaks[index + 1]) not in s2_peaks:
                    s2_peaks.append(int(peaks[index + 1]))
            else:
                if int(peaks[index]) not in s2_peaks:
                    s2_peaks.append(int(peaks[index]))
                if int(peaks[index + 1]) not in s1_peaks:
                    s1_peaks.append(int(peaks[index + 1]))

    s1_peaks_np = np.array(sorted(set(s1_peaks)))
    s2_peaks_np = np.array(sorted(set(s2_peaks)))

    states, segments = segment_heart_sounds(filtered, envelope, s1_peaks_np, s2_peaks_np, sample_rate)

    state_names = ["S1", "Systole", "S2", "Diastole"]
    segmentation_stats: dict[str, dict[str, float]] = {}
    for state_name in state_names:
        durations = [
            (end - start) / sample_rate * 1000
            for name, start, end in segments
            if name == state_name
        ]
        if durations:
            segmentation_stats[state_name] = {
                "count": len(durations),
                "mean_ms": float(np.mean(durations)),
                "std_ms": float(np.std(durations)),
                "min_ms": float(np.min(durations)),
                "max_ms": float(np.max(durations)),
            }

    cycles = extract_cycle_features(filtered, segments, sample_rate)

    flagged_cycles: list[dict[str, Any]] = []
    for cycle_index, cycle in enumerate(cycles):
        violations: list[str] = []
        for key, (lower, upper) in cfg.normal_ranges.items():
            if key in cycle:
                value = cycle[key]
                if value < lower:
                    violations.append(f"{key}={value:.1f} < {lower}")
                elif value > upper:
                    violations.append(f"{key}={value:.1f} > {upper}")
        if violations:
            flagged_cycles.append({"cycle_index": cycle_index, "violations": violations})

    feature_keys: list[str] = []
    labels: np.ndarray = np.array([])
    scores: np.ndarray = np.array([])
    anomaly_summary = {
        "normal_cycles": 0,
        "anomaly_cycles": 0,
        "score_min": None,
        "score_max": None,
    }

    if len(cycles) >= 2:
        feature_keys = [key for key in cycles[0] if not key.startswith("_")]
        feature_matrix = np.array([[cycle[key] for key in feature_keys] for cycle in cycles])
        feature_matrix = np.nan_to_num(feature_matrix)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature_matrix)

        isolation_forest = IsolationForest(
            contamination=0.15,
            random_state=42,
            n_estimators=200,
        )
        labels = isolation_forest.fit_predict(scaled)
        scores = isolation_forest.decision_function(scaled)

        anomaly_summary = {
            "normal_cycles": int(np.sum(labels == 1)),
            "anomaly_cycles": int(np.sum(labels == -1)),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
        }
    elif len(cycles) == 1:
        feature_keys = [key for key in cycles[0] if not key.startswith("_")]
        labels = np.array([1])
        scores = np.array([0.0])
        anomaly_summary = {
            "normal_cycles": 1,
            "anomaly_cycles": 0,
            "score_min": 0.0,
            "score_max": 0.0,
        }

    per_cycle_stats: dict[str, dict[str, float]] = {}
    stat_keys = [
        "heart_rate_bpm",
        "s1_duration_ms",
        "s2_duration_ms",
        "systolic_ms",
        "diastolic_ms",
        "s1_s2_amp_ratio",
        "sd_ratio",
    ]
    if cycles:
        for key in stat_keys:
            values = np.array([cycle[key] for cycle in cycles])
            per_cycle_stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    hrv_metrics: dict[str, float] | None = None
    if cycles:
        rr_intervals = np.array([cycle["cycle_duration_ms"] for cycle in cycles])
        valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 1500)]
        if len(valid_rr) > 2:
            sdnn = np.std(valid_rr)
            rmssd = np.sqrt(np.mean(np.diff(valid_rr) ** 2))
            nn50 = np.sum(np.abs(np.diff(valid_rr)) > 50)
            pnn50 = 100 * nn50 / len(valid_rr)
            heart_rates = 60000 / valid_rr
            hrv_metrics = {
                "heart_rate_mean_bpm": float(np.mean(heart_rates)),
                "heart_rate_std_bpm": float(np.std(heart_rates)),
                "sdnn_ms": float(sdnn),
                "rmssd_ms": float(rmssd),
                "pnn50_pct": float(pnn50),
            }

    murmur_results = [
        detect_murmur(filtered, cycle, sample_rate, cfg.murmur_grade_thresholds)
        for cycle in cycles
    ]
    systolic_count = sum(1 for item in murmur_results if item["systolic_murmur"])
    diastolic_count = sum(1 for item in murmur_results if item["diastolic_murmur"])
    total_cycles = len(murmur_results)

    systolic_pct = (100 * systolic_count / total_cycles) if total_cycles > 0 else 0.0
    diastolic_pct = (100 * diastolic_count / total_cycles) if total_cycles > 0 else 0.0

    if total_cycles == 0:
        murmur_assessment = "No complete cycles available for murmur assessment"
    elif systolic_pct > 50 or diastolic_pct > 50:
        murmur_assessment = (
            f"Significant murmur activity (Sys: {systolic_pct:.0f}%, Dia: {diastolic_pct:.0f}%)"
        )
    elif systolic_pct > 20 or diastolic_pct > 20:
        murmur_assessment = "Some murmur-like activity detected (may be recording noise)"
    else:
        murmur_assessment = "No significant murmur activity"

    saved_filtered_wav: str | None = None
    filtered_normalized = filtered / (np.max(np.abs(filtered)) + 1e-10)
    if save_filtered_wav:
        if output_filename is None:
            output_filename = str(Path(filename).with_name(Path(filename).stem + "_filtered.wav"))
        wav_write(output_filename, sample_rate, (filtered_normalized * 32767).astype(np.int16))
        saved_filtered_wav = output_filename

    result: dict[str, Any] = {
        "file_info": {
            "filename": filename,
            "sample_rate_hz": sample_rate,
            "nyquist_hz": nyquist,
            "duration_s": duration,
            "samples": n_samples,
            "highcut_clamped": bool(cfg.highcut >= nyquist),
            "effective_highcut_hz": min(cfg.highcut, nyquist - 1.0),
        },
        "config": asdict(cfg),
        "peaks": {
            "total_peaks": int(len(peaks)),
            "s1_count": int(len(s1_peaks_np)),
            "s2_count": int(len(s2_peaks_np)),
            "peak_indices": peaks,
            "s1_indices": s1_peaks_np,
            "s2_indices": s2_peaks_np,
            "peak_times_s": peaks / sample_rate,
            "s1_times_s": s1_peaks_np / sample_rate,
            "s2_times_s": s2_peaks_np / sample_rate,
            "threshold": float(threshold),
        },
        "segmentation": {
            "state_names": state_names,
            "states": states,
            "segments": [
                {
                    "state": state_name,
                    "start_index": start,
                    "end_index": end,
                    "start_s": start / sample_rate,
                    "end_s": end / sample_rate,
                }
                for state_name, start, end in segments
            ],
            "stats": segmentation_stats,
        },
        "classification": {
            "cycles": cycles,
            "rule_based": {
                "normal_cycles": len(cycles) - len(flagged_cycles),
                "flagged_cycles": len(flagged_cycles),
                "flagged_details": flagged_cycles,
            },
            "isolation_forest": {
                "feature_keys": feature_keys,
                "labels": labels,
                "scores": scores,
                **anomaly_summary,
            },
            "per_cycle_stats": per_cycle_stats,
            "hrv_metrics": hrv_metrics,
        },
        "murmur": {
            "analysis_range_hz": [0.0, nyquist],
            "systolic_murmur_cycles": systolic_count,
            "diastolic_murmur_cycles": diastolic_count,
            "total_cycles": total_cycles,
            "systolic_pct": systolic_pct,
            "diastolic_pct": diastolic_pct,
            "assessment": murmur_assessment,
            "cycle_results": murmur_results,
        },
        "exports": {
            "saved_filtered_wav": saved_filtered_wav,
        },
    }

    if include_signals:
        result["signals"] = {
            "time_axis_s": time_axis,
            "raw": raw_data,
            "filtered": filtered,
            "filtered_normalized": filtered_normalized,
            "envelope": envelope,
        }

    return _to_builtin(result)


def run_pipeline(filename: str, **kwargs: Any) -> dict[str, Any]:
    return run_pcg_pipeline(filename=filename, **kwargs)
