import wave

from app.services.audio_storage import AudioStorage


def test_save_wav_preserves_duration_for_fractional_source_rate(tmp_path) -> None:
    storage = AudioStorage("/media/audio", str(tmp_path))
    raw_samples = [2048 + (index % 128) for index in range(3750)]

    target = storage.save_wav(
        recording_id="rec_fractional_rate",
        raw_samples=raw_samples,
        waveform=None,
        source_sample_rate=62.5,
        audio_sample_rate=500,
        gain=1.0,
    )

    with wave.open(str(target), "rb") as wav_file:
        assert wav_file.getframerate() == 500
        assert wav_file.getnframes() == 30_000
        assert wav_file.getnframes() / wav_file.getframerate() == 60.0
