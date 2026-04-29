from app.services.waveform import generate_idle_waveform, generate_live_waveform, normalize_waveform


def test_normalize_waveform_maps_values_into_zero_to_one_range() -> None:
    assert normalize_waveform([-2.0, 0.0, 2.0]) == [0.0, 0.5, 1.0]


def test_idle_waveform_stays_low_amplitude() -> None:
    waveform = generate_idle_waveform(180, seed=7)
    assert len(waveform) == 180
    assert all(0.0 <= sample <= 1.0 for sample in waveform)
    assert max(waveform) - min(waveform) < 0.08


def test_live_waveform_returns_normalized_window() -> None:
    waveform = generate_live_waveform(180, runtime_ms=18_640, area_id="aortic", seed=12345)
    assert len(waveform) == 180
    assert all(0.0 <= sample <= 1.0 for sample in waveform)
    assert min(waveform) < 0.2
    assert max(waveform) > 0.8
