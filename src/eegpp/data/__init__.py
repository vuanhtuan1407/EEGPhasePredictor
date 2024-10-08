import os
from pathlib import Path

from src.eegpp import EEGPP_DIR

DATA_DIR = os.path.join(EEGPP_DIR, 'data')
DUMP_DATA_DIR = os.path.join(DATA_DIR, 'dump')

SEQ_FILES = [
    str(Path(DATA_DIR, "raw_K3_EEG3_11h.txt")),
    str(Path(DATA_DIR, "raw_RS2_EEG1_23 hr.txt")),
    str(Path(DATA_DIR, "raw_S1_EEG1_23 hr.txt")),
]

LABEL_FILES = [
    str(Path(DATA_DIR, "K3_EEG3_11h.txt")),
    str(Path(DATA_DIR, "RS2_EEG1_23 hr.txt")),
    str(Path(DATA_DIR, "S1_EEG1_23 hr.txt")),
]

DUMP_DATA_FILES = {
    "train": [
        str(Path(DUMP_DATA_DIR, "dump_eeg_1.pkl")),
        str(Path(DUMP_DATA_DIR, "dump_eeg_2.pkl")),
        str(Path(DUMP_DATA_DIR, "dump_eeg_3.pkl")),

    ],
    "infer": [
        str(Path(DUMP_DATA_DIR, "dump_eeg_1_infer.pkl")),
        str(Path(DUMP_DATA_DIR, "dump_eeg_2_infer.pkl")),
        str(Path(DUMP_DATA_DIR, "dump_eeg_3_infer.pkl")),
    ]
}
