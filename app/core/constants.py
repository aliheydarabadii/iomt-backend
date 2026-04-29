AUSCULTATION_AREAS: dict[str, dict[str, str]] = {
    "aortic": {
        "label": "Aortic",
        "short": "2nd ICS · right sternal border",
    },
    "pulmonic": {
        "label": "Pulmonic",
        "short": "2nd ICS · left sternal border",
    },
    "erbs-point": {
        "label": "Erb's Point",
        "short": "3rd ICS · left sternal border",
    },
    "tricuspid": {
        "label": "Tricuspid",
        "short": "4th ICS · lower left sternal border",
    },
    "mitral": {
        "label": "Mitral / Apex",
        "short": "5th ICS · apex / left midclavicular line",
    },
}

DEFAULT_RECORDING_STATUS = "Stored on server"
LIVE_STREAM_STATUS = "Receiving waveform"
IDLE_STREAM_STATUS = "Idle"
DEFAULT_SIGNAL_QUALITY = "Good"
IDLE_SIGNAL_QUALITY = "No live signal"
