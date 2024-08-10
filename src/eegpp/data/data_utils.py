# import joblib
import re

import numpy as np
from tqdm import tqdm

from src.eegpp import utils as ut
from src.eegpp.data import DATA_FILES, LABEL_FILES


def load_data_with_labels(step_ms=4000):
    all_start_ms, all_eeg, all_emg, all_mot, all_lbs = [[], [], [], [], []]
    for DATA_FILE, LABEL_FILE in zip(DATA_FILES, LABEL_FILES):
        start_ms, eeg, emg, mot, lbs = [[], [], [], [], []]
        with open(DATA_FILE, 'r') as f:
            data = f.readlines()
            start_line = 0
            while start_line < len(data):
                # print(start_line)
                if data[start_line].__contains__('Time'):
                    start_line = start_line + 1
                    break
                else:
                    start_line = start_line + 1

            tmp_ms, tmp_eeg, tmp_emg, tmp_mot = [[], [], [], []]
            for line in tqdm(data[start_line:], total=len(data[start_line:]), desc=DATA_FILE.split('\\')[-1]):
                ms, eeg6, emg6, mot6, _ = line.split('\t')
                ms = ut.convert_time2ms(ms)
                if len(tmp_ms) == 0:
                    start_ms.append(ms)
                    tmp_ms.append(ms)
                if ms - tmp_ms[0] < step_ms:
                    tmp_eeg.append(eeg6)
                    tmp_emg.append(emg6)
                    tmp_mot.append(mot6)
                else:
                    eeg.append(tmp_eeg)
                    emg.append(tmp_emg)
                    mot.append(tmp_mot)
                    tmp_ms, tmp_eeg, tmp_emg, tmp_mot = [[], [], [], []]

        with open(LABEL_FILE, 'r') as f:
            data = f.readlines()
            reg_pattern = re.compile(r'\d+\t(?:W|R|NR)\*?\t\d{4}\.\d{2}\.\d{2}\.\s\d{2}:\d{2}:\d{2}')
            # tmp_ms, tmp_lbs = [[], []]
            tmp_idx = 0
            for line in tqdm(data, total=len(data)):
                match = re.search(reg_pattern, line)
                if match is not None:
                    _, lb, sec = match.group().split('\t')
                    # sec = ut.convert_time2ms(sec)
                    lbs.append(lb)
                    # if sec == start_ms[tmp_idx]:
                    #     tmp_idx += 1
                    #     lbs.append(lb)
                    # else:
                    #     raise ValueError("Data and Label mismatch")

        print(np.array(start_ms).shape, np.array(eeg).shape, np.array(mot).shape, np.array(lbs).shape)

        all_start_ms.append(start_ms)
        all_eeg.append(eeg)
        all_emg.append(emg)
        all_mot.append(mot)
        all_lbs.append(lbs)


def load_data_with_no_labels(step_ms=4000):
    all_start_ms, all_eeg, all_emg, all_mot = [[], [], [], []]
    for DATA_FILE in DATA_FILES:
        start_ms, eeg, emg, mot = [[], [], [], []]
        with open(DATA_FILE, 'r') as f:
            data = f.readlines()
            start_line = 0
            while start_line < len(data):
                # print(start_line)
                if data[start_line].__contains__('Time'):
                    start_line = start_line + 1
                    break
                else:
                    start_line = start_line + 1

            tmp_ms, tmp_eeg, tmp_emg, tmp_mot = [[], [], [], []]
            for line in tqdm(data[start_line:], total=len(data[start_line:]), desc=DATA_FILE.split('\\')[-1]):
                ms, eeg6, emg6, mot6, _ = line.split('\t')
                ms = ut.convert_time2ms(ms)
                if len(tmp_ms) == 0:
                    start_ms.append(ms)
                    tmp_ms.append(ms)
                if ms - tmp_ms[0] < step_ms:
                    tmp_eeg.append(eeg6)
                    tmp_emg.append(emg6)
                    tmp_mot.append(mot6)
                else:
                    eeg.append(tmp_eeg)
                    emg.append(tmp_emg)
                    mot.append(tmp_mot)
                    tmp_ms, tmp_eeg, tmp_emg, tmp_mot = [[], [], [], []]

        print(start_ms[0], eeg[0], emg[0], mot[0])
        all_start_ms.append(start_ms)
        all_eeg.append(eeg)
        all_emg.append(emg)
        all_mot.append(mot)


def split_dataset_into_dirs():
    pass


if __name__ == '__main__':
    load_data_with_labels()
