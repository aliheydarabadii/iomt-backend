import numpy as np
from pathlib import Path
from scipy.io import wavfile

from app.models import HeartRecording
from app.services.recording_service import RecordingService


def test_filtered_wav_is_resampled_for_browser_playback(tmp_path) -> None:
    source_rate = 500
    duration_seconds = 2
    time_axis = np.arange(source_rate * duration_seconds) / source_rate
    samples = (np.sin(2 * np.pi * 80 * time_axis) * 12000).astype(np.int16)
    filtered_file = tmp_path / "filtered.wav"
    wavfile.write(filtered_file, source_rate, samples)

    playback_rate = RecordingService._make_filtered_wav_browser_playable(
        filtered_file
    )
    saved_rate, saved_samples = wavfile.read(filtered_file)

    assert playback_rate == 8000
    assert saved_rate == 8000
    assert len(saved_samples) / saved_rate == duration_seconds


def test_original_playback_copy_preserves_analysis_source(tmp_path) -> None:
    source_rate = 500
    duration_seconds = 3
    time_axis = np.arange(source_rate * duration_seconds) / source_rate
    samples = (np.sin(2 * np.pi * 75 * time_axis) * 10000).astype(np.int16)
    source_file = tmp_path / "original.wav"
    playback_file = tmp_path / "original_playback.wav"
    wavfile.write(source_file, source_rate, samples)

    served_file, playback_rate = RecordingService._ensure_browser_playback_wav(
        source_file,
        playback_file,
    )
    source_rate_after, source_samples_after = wavfile.read(source_file)
    saved_rate, saved_samples = wavfile.read(playback_file)

    assert served_file == playback_file
    assert playback_rate == 8000
    assert source_rate_after == source_rate
    assert np.array_equal(source_samples_after, samples)
    assert saved_rate == 8000
    assert len(saved_samples) / saved_rate == duration_seconds


def test_delete_recording_removes_database_row_and_all_wav_files(
    db_session,
    test_settings,
) -> None:
    recording_id = "rec_001"
    storage_dir = Path(test_settings.audio_storage_dir)
    paths = [
        storage_dir / f"{recording_id}.wav",
        storage_dir / f"{recording_id}_filtered.wav",
        storage_dir / f"{recording_id}_playback.wav",
    ]
    storage_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        path.write_bytes(b"RIFF-test")

    RecordingService(
        db_session,
        settings=test_settings,
    ).delete_recording(recording_id)

    assert db_session.get(HeartRecording, recording_id) is None
    assert all(not path.exists() for path in paths)
