import wave
from pathlib import Path


class AudioStorage:
    def __init__(self, base_url: str, storage_dir: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def build_audio_url(self, recording_id: str) -> str:
        return f"{self.base_url}/{recording_id}.wav"

    def save_wav(
        self,
        *,
        recording_id: str,
        raw_samples: list[int] | None,
        waveform: list[float] | None,
        source_sample_rate: int,
        audio_sample_rate: int,
        gain: float,
    ) -> Path:
        target_path = self.storage_dir / f"{recording_id}.wav"
        pcm_samples = self._build_pcm_samples(
            raw_samples=raw_samples or [],
            waveform=waveform or [],
            source_sample_rate=source_sample_rate,
            audio_sample_rate=audio_sample_rate,
            gain=gain,
        )
        with wave.open(str(target_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(audio_sample_rate)
            wav_file.writeframes(pcm_samples)
        return target_path

    def _build_pcm_samples(
        self,
        *,
        raw_samples: list[int],
        waveform: list[float],
        source_sample_rate: int,
        audio_sample_rate: int,
        gain: float,
    ) -> bytes:
        if raw_samples:
            normalized = self._normalize_raw_samples(raw_samples)
        else:
            normalized = self._normalize_waveform(waveform)
        upsampled = self._upsample(normalized, source_sample_rate, audio_sample_rate)
        pcm = bytearray()
        for sample in upsampled:
            value = int(max(-1.0, min(1.0, sample * gain)) * 32767)
            pcm.extend(value.to_bytes(2, byteorder="little", signed=True))
        return bytes(pcm)

    @staticmethod
    def _normalize_raw_samples(samples: list[int]) -> list[float]:
        if not samples:
            return [0.0]
        mean_value = sum(samples) / len(samples)
        centered = [sample - mean_value for sample in samples]
        peak = max((abs(sample) for sample in centered), default=0.0)
        if peak == 0:
            return [0.0 for _ in samples]
        return [sample / peak for sample in centered]

    @staticmethod
    def _normalize_waveform(waveform: list[float]) -> list[float]:
        if not waveform:
            return [0.0]
        return [max(-1.0, min(1.0, (sample - 0.5) * 2.0)) for sample in waveform]

    @staticmethod
    def _upsample(samples: list[float], source_rate: int, target_rate: int) -> list[float]:
        if not samples:
            return [0.0]
        if source_rate <= 0 or target_rate <= 0 or source_rate == target_rate:
            return samples
        if target_rate % source_rate == 0:
            factor = target_rate // source_rate
            return [sample for sample in samples for _ in range(factor)]

        output_length = max(1, round(len(samples) * target_rate / source_rate))
        result: list[float] = []
        for index in range(output_length):
            position = index * (len(samples) - 1) / max(output_length - 1, 1)
            left = int(position)
            right = min(left + 1, len(samples) - 1)
            blend = position - left
            result.append((samples[left] * (1 - blend)) + (samples[right] * blend))
        return result
