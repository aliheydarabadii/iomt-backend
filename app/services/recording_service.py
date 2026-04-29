from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi.responses import FileResponse, RedirectResponse, Response
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.time import utcnow
from app.core.errors import AppError, NotFoundError
from app.models import HeartRecording
from app.schemas.recordings import RecordingAnalysisResponse, RecordingPlotResponse


class RecordingService:
    def __init__(self, db: Session, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()

    def build_audio_url(self, recording_id: str) -> str:
        return f"{self.settings.api_prefix}/heart-recordings/{recording_id}/audio"

    def build_analysis_url(self, recording_id: str) -> str:
        return f"{self.settings.api_prefix}/heart-recordings/{recording_id}/analysis"

    def get_audio_response(self, recording_id: str) -> Response:
        recording = self._get_recording_or_404(recording_id)

        local_file = self._get_local_audio_path(recording_id)
        if local_file.exists():
            return FileResponse(
                path=local_file,
                media_type="audio/wav",
                filename=f"{recording_id}.wav",
            )

        parsed = urlparse(recording.audio_url)
        if parsed.scheme in {"http", "https"}:
            return RedirectResponse(recording.audio_url)

        raise NotFoundError(
            "recording_audio_not_found",
            "Recorded audio file was not found",
            {"recordingId": recording_id},
        )

    def get_analysis(
        self,
        *,
        recording_id: str,
        include_signals: bool,
        save_filtered_wav: bool,
        max_points: int,
    ) -> RecordingAnalysisResponse:
        recording = self._get_recording_or_404(recording_id)
        local_file = self._get_local_audio_path(recording_id)
        if not local_file.exists():
            raise NotFoundError(
                "recording_analysis_audio_not_found",
                "Local recorded audio file was not found for analysis",
                {"recordingId": recording_id},
            )

        try:
            from app.services.pcg_pipeline import run_pcg_pipeline
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages
            raise AppError(
                status_code=500,
                code="analysis_dependency_missing",
                message="PCG analysis dependencies are not installed",
                details={"recordingId": recording_id},
            ) from exc

        output_filename = None
        if save_filtered_wav:
            output_filename = str(local_file.with_name(f"{recording_id}_filtered.wav"))

        analysis = run_pcg_pipeline(
            filename=str(local_file),
            save_filtered_wav=save_filtered_wav,
            output_filename=output_filename,
            include_signals=include_signals,
        )

        plot = None
        if include_signals and "signals" in analysis:
            plot = self._build_plot_payload(analysis, max_points=max_points)
            analysis["signals"] = self._downsample_analysis_signals(
                analysis["signals"],
                max_points=max_points,
            )

        return RecordingAnalysisResponse(
            recordingId=recording.id,
            audioUrl=self.build_audio_url(recording.id),
            generatedAt=utcnow(),
            plot=plot,
            analysis=analysis,
        )

    def _get_recording_or_404(self, recording_id: str) -> HeartRecording:
        recording = self.db.get(HeartRecording, recording_id)
        if not recording:
            raise NotFoundError("recording_not_found", "Recording was not found")
        return recording

    def _get_local_audio_path(self, recording_id: str) -> Path:
        return Path(self.settings.audio_storage_dir) / f"{recording_id}.wav"

    def _build_plot_payload(
        self,
        analysis: dict[str, Any],
        *,
        max_points: int,
    ) -> RecordingPlotResponse:
        signals = analysis.get("signals", {})
        file_info = analysis.get("file_info", {})
        peaks = analysis.get("peaks", {})

        time_axis = [float(item) for item in signals.get("time_axis_s", [])]
        amplitude = [float(item) for item in signals.get("filtered_normalized", [])]
        envelope = [float(item) for item in signals.get("envelope", [])]

        time_axis, amplitude, envelope, downsampled = self._downsample_triplet(
            time_axis,
            amplitude,
            envelope,
            max_points=max_points,
        )

        return RecordingPlotResponse(
            sampleRateHz=int(file_info.get("sample_rate_hz", 0)),
            durationS=float(file_info.get("duration_s", 0.0)),
            pointCount=len(time_axis),
            maxPoints=max_points,
            downsampled=downsampled,
            timeAxisS=time_axis,
            amplitude=amplitude,
            envelope=envelope,
            peakTimesS=[float(item) for item in peaks.get("peak_times_s", [])],
            s1TimesS=[float(item) for item in peaks.get("s1_times_s", [])],
            s2TimesS=[float(item) for item in peaks.get("s2_times_s", [])],
        )

    def _downsample_analysis_signals(
        self,
        signals: dict[str, Any],
        *,
        max_points: int,
    ) -> dict[str, Any]:
        time_axis = [float(item) for item in signals.get("time_axis_s", [])]
        filtered = [float(item) for item in signals.get("filtered", [])]
        filtered_normalized = [float(item) for item in signals.get("filtered_normalized", [])]
        envelope = [float(item) for item in signals.get("envelope", [])]
        raw = [int(item) for item in signals.get("raw", [])]

        if time_axis:
            indices = self._select_indices(len(time_axis), max_points=max_points)
            return {
                "time_axis_s": [time_axis[index] for index in indices],
                "raw": [raw[index] for index in indices] if len(raw) == len(time_axis) else raw,
                "filtered": [filtered[index] for index in indices]
                if len(filtered) == len(time_axis)
                else filtered,
                "filtered_normalized": [
                    filtered_normalized[index] for index in indices
                ]
                if len(filtered_normalized) == len(time_axis)
                else filtered_normalized,
                "envelope": [envelope[index] for index in indices]
                if len(envelope) == len(time_axis)
                else envelope,
            }
        return signals

    def _downsample_triplet(
        self,
        first: list[float],
        second: list[float],
        third: list[float],
        *,
        max_points: int,
    ) -> tuple[list[float], list[float], list[float], bool]:
        if not first:
            return first, second, third, False
        indices = self._select_indices(len(first), max_points=max_points)
        if len(indices) == len(first):
            return first, second, third, False
        return (
            [first[index] for index in indices],
            [second[index] for index in indices] if len(second) == len(first) else second,
            [third[index] for index in indices] if len(third) == len(first) else third,
            True,
        )

    @staticmethod
    def _select_indices(length: int, *, max_points: int) -> list[int]:
        if length <= max_points:
            return list(range(length))
        if max_points <= 1:
            return [0]
        step = (length - 1) / (max_points - 1)
        return [min(length - 1, round(index * step)) for index in range(max_points)]
