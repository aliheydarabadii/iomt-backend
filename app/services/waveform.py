import math
from statistics import fmean


def clamp_waveform(samples: list[float]) -> list[float]:
    return [round(min(1.0, max(0.0, sample)), 4) for sample in samples]


def normalize_waveform(samples: list[float]) -> list[float]:
    if not samples:
        return []
    minimum = min(samples)
    maximum = max(samples)
    if math.isclose(minimum, maximum):
        return [0.5 for _ in samples]
    scale = maximum - minimum
    normalized = [(sample - minimum) / scale for sample in samples]
    return clamp_waveform(normalized)


def generate_idle_waveform(size: int, seed: int = 0) -> list[float]:
    samples = []
    for index in range(size):
        theta = (index + seed) / 9
        value = 0.5 + (0.02 * math.sin(theta)) + (0.01 * math.cos(theta / 3))
        samples.append(value)
    return clamp_waveform(samples)


def generate_live_waveform(size: int, runtime_ms: int, area_id: str, seed: int) -> list[float]:
    area_factor = {
        "aortic": 1.15,
        "pulmonic": 1.08,
        "erbs-point": 1.0,
        "tricuspid": 0.94,
        "mitral": 0.9,
    }.get(area_id, 1.0)
    offset = runtime_ms / 1000
    raw_samples: list[float] = []
    for index in range(size):
        t = offset + (index / 48)
        fundamental = math.sin((t * 6.2 * area_factor) + (seed % 17))
        harmonic = 0.55 * math.sin((t * 12.4) + (seed % 11))
        envelope = 1.0 + (0.35 * math.sin((t * 1.7) + area_factor))
        shimmer = 0.12 * math.sin((t * 22.0) + (seed % 5))
        raw_samples.append((fundamental + harmonic) * envelope + shimmer)
    normalized = normalize_waveform(raw_samples)
    return [round((sample * 0.88) + 0.06, 4) for sample in normalized]


def summarize_waveform(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        return {"sampleCount": 0, "min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "sampleCount": len(samples),
        "min": round(min(samples), 4),
        "max": round(max(samples), 4),
        "avg": round(fmean(samples), 4),
    }
