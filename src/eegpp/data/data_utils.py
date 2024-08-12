# import joblib
import os.path

import joblib
from tqdm import tqdm

from src.eegpp import utils as ut
from src.eegpp.data import SEQ_FILES, LABEL_FILES, DUMP_DATA_FILES, DUMP_DATA_DIR

LABEL_DICT = {0: "W", 1: "W*", 2: "NR", 3: "NR*", 4: "R", 5: "R*", -1: "others"}

PATH_SLASH = ut.get_path_slash()


def get_lb_idx(lb_text):
    lb_idx = -1
    for k, v in LABEL_DICT.items():
        if v == lb_text:
            lb_idx = k
            break
    return lb_idx


def dump_seq_with_labels(seq_files=SEQ_FILES, lb_files=LABEL_FILES, step_ms=4000):
    try:
        all_start_ms1, all_eeg, all_emg, all_mot = load_seq(seq_files, step_ms)
        all_start_ms2, all_lbs = load_lbs(lb_files)

        # =========================================
        # Handle if all_start_ms1 != all_start_ms2
        all_start_ms = []
        for start_ms1, start_ms2 in zip(all_start_ms1, all_start_ms2):
            all_start_ms.append(start_ms1 if len(start_ms1) < len(start_ms2) else start_ms2)

        # =========================================

        # Assume that all_start_ms1 == all_start_ms2
        print('Dump data files...')
        for i, start_ms in enumerate(all_start_ms):
            eeg, emg, mot, lbs = all_eeg[i], all_emg[i], all_mot[i], all_lbs[i]
            phases = []
            for tmp_start_ms, tmp_eeg, tmp_emg, tmp_mot, tmp_lb in zip(start_ms, eeg, emg, mot, lbs):
                tmp_start_datetime = ut.convert_ms2datetime(tmp_start_ms)
                phase = {
                    "start_ms": tmp_start_ms,
                    "start_datetime": tmp_start_datetime,
                    "eeg": tmp_eeg,
                    "emg": tmp_emg,
                    "mot": tmp_mot,
                }
                phases.append(phase)
            joblib.dump(phases, DUMP_DATA_FILES['train'][i])
            print(f'Dump data in file {DUMP_DATA_FILES["train"][i]}')
    except Exception as e:
        raise e


def dump_seq_with_no_labels(seq_files=SEQ_FILES, step_ms=4000):
    try:
        all_start_ms, all_eeg, all_emg, all_mot = load_seq(seq_files, step_ms)
        print('Dump data files...')
        for i, start_ms, eeg, emg, mot in enumerate(zip(all_start_ms, all_eeg, all_emg, all_mot)):
            phases = []
            for tmp_start_ms, tmp_eeg, tmp_emg, tmp_mot in zip(start_ms, eeg, emg, mot):
                tmp_start_datetime = ut.convert_ms2datetime(tmp_start_ms)
                phase = {
                    "start_ms": tmp_start_ms,
                    "start_datetime": tmp_start_datetime,
                    "eeg": tmp_eeg,
                    "emg": tmp_emg,
                    "mot": tmp_mot,
                }
                phases.append(phase)
            joblib.dump(phases, DUMP_DATA_FILES['infer'][i])
            print(f'Dump data in file {DUMP_DATA_FILES["infer"][i]}')
    except Exception as e:
        raise e


def load_seq(data_files=SEQ_FILES, step_ms=4000):
    print('Processing sequences...')
    all_start_ms, all_eeg, all_emg, all_mot = [[], [], [], []]
    for data_file in data_files:
        start_ms, eeg, emg, mot = [[], [], [], []]
        with open(data_file, 'r', encoding='utf-8', errors='replace') as f:
            data = f.readlines()
            start_line = 0
            while not data[start_line].__contains__('Time') and start_line < len(data):
                start_line = start_line + 1
            tmp_ms = 0
            tmp_eeg, tmp_emg, tmp_mot = [[], [], []]
            for line in tqdm(data[start_line + 1:], total=len(data[start_line + 1:]),
                             desc=data_file.split(PATH_SLASH)[-1]):
                split_data = line.split('\t')
                if len(split_data) == 2:  # Final line in raw_S1_EEG1_23 hr.txt
                    dt, eeg6 = split_data
                    emg6, mot6 = 0, 0  # Fill missing value
                elif len(split_data) == 4:  # Final line in raw_RS2_EEG1_23 hr.txt
                    dt, eeg6, emg6, mot6 = split_data
                else:
                    dt, eeg6, emg6, mot6, _ = split_data

                ms = ut.convert_datetime2ms(dt)
                if tmp_ms == 0:
                    tmp_ms = ms
                if ms - tmp_ms >= step_ms:
                    start_ms.append(tmp_ms)
                    eeg.append(tmp_eeg)
                    emg.append(tmp_emg)
                    mot.append(tmp_mot)
                    tmp_ms = ms
                    tmp_eeg, tmp_emg, tmp_mot = [[], [], []]

                tmp_eeg.append(eeg6)
                tmp_emg.append(emg6)
                tmp_mot.append(mot6)

            start_ms.append(tmp_ms)
            eeg.append(tmp_eeg)
            emg.append(tmp_emg)
            mot.append(tmp_mot)

        all_start_ms.append(start_ms)
        all_eeg.append(eeg)
        all_emg.append(emg)
        all_mot.append(mot)

    return all_start_ms, all_eeg, all_emg, all_mot


def load_lbs(data_files=LABEL_FILES):
    print("Processing labels...")
    all_start_ms, all_lbs = [[], []]
    for data_file in data_files:
        with open(data_file, 'r', errors='replace', encoding='utf-8') as f:
            data = f.readlines()
            start_line = 0
            while not data[start_line].__contains__('Time') and start_line < len(data):
                start_line = start_line + 1
            tmp_ms, tmp_lbs = [[], []]
            for line in tqdm(data[start_line + 1:-1], total=len(data[start_line + 1:-1]),
                             desc=data_file.split(PATH_SLASH)[-1]):
                _, lb_text, dt = line.split('\t')[:3]
                ms = ut.convert_datetime2ms(dt)
                lb = get_lb_idx(lb_text)
                tmp_ms.append(ms)
                tmp_lbs.append(lb)
        all_start_ms.append(tmp_ms)
        all_lbs.append(tmp_lbs)

    return all_start_ms, all_lbs


def split_dataset_into_dirs():
    pass


if __name__ == '__main__':
    os.makedirs(DUMP_DATA_DIR, exist_ok=True)
    dump_seq_with_labels()
