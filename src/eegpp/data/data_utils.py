# import joblib

import joblib
from tqdm import tqdm

from src.eegpp import utils as ut
from src.eegpp.data import SEQ_FILES, LABEL_FILES, DUMP_DATA_FILES

LABEL_DICT = {0: "W", 1: "W*", 2: "NR", 3: "NR*", 4: "R", 5: "R*", -1: "others"}

PATH_SLASH = ut.get_path_slash()

INF_V = -1e6


def get_lb_idx(lb_text):
    lb_idx = -1
    for k, v in LABEL_DICT.items():
        if v == lb_text:
            lb_idx = k
            break
    return lb_idx


# def dump_seq_with_labels(seq_files=SEQ_FILES, lb_files=LABEL_FILES, step_ms=4000):
#     try:
#         all_start_ms1, all_eeg, all_emg, all_mot = load_seq(seq_files, step_ms)
#         all_start_ms2, all_lbs = load_lbs(lb_files)
#
#         # =========================================
#         # Handle if all_start_ms1 != all_start_ms2
#         all_start_ms = []
#         for start_ms1, start_ms2 in zip(all_start_ms1, all_start_ms2):
#             all_start_ms.append(start_ms1 if len(start_ms1) < len(start_ms2) else start_ms2)
#
#         # =========================================
#
#         # Assume that all_start_ms1 == all_start_ms2
#         print('Dump data files...')
#         for i, start_ms in enumerate(all_start_ms):
#             eeg, emg, mot, lbs = all_eeg[i], all_emg[i], all_mot[i], all_lbs[i]
#             start_datetime = [ut.convert_ms2datetime(ms) for ms in start_ms]
#             joblib.dump((start_datetime, eeg, emg, mot, lbs), DUMP_DATA_FILES['train'][i])
#
#             # phases = []
#             # for tmp_start_ms, tmp_eeg, tmp_emg, tmp_mot, tmp_lb in zip(start_ms, eeg, emg, mot, lbs):
#             #     tmp_start_datetime = ut.convert_ms2datetime(tmp_start_ms)
#             #     phase = {
#             #         "start_ms": tmp_start_ms,
#             #         "start_datetime": tmp_start_datetime,
#             #         "eeg": tmp_eeg,
#             #         "emg": tmp_emg,
#             #         "mot": tmp_mot,
#             #     }
#             #     phases.append(phase)
#             # joblib.dump(phases, DUMP_DATA_FILES['train'][i])
#             print(f'Dump data in file {DUMP_DATA_FILES["train"][i]}')
#     except Exception as e:
#         raise e


def dump_seq_with_labels(seq_files=SEQ_FILES, lb_files=LABEL_FILES):
    try:
        all_start_ms, all_eeg, all_emg, all_mot, all_lbs, all_mxs = load_seq_with_labels(seq_files, lb_files)
        print('Dump data files...')
        for i, start_ms, eeg, emg, mot, lbs in enumerate(zip(all_start_ms, all_eeg, all_emg, all_mot, all_lbs)):
            eeg, emg, mot, lbs = all_eeg[i], all_emg[i], all_mot[i], all_lbs[i]
            start_datetime = [ut.convert_ms2datetime(ms) for ms in start_ms]
            joblib.dump((start_datetime, eeg, emg, mot, lbs), DUMP_DATA_FILES['train'][i])
            print(f'Dump data in file {DUMP_DATA_FILES["train"][i]}')
    except Exception as e:
        raise e


def dump_seq_with_no_labels(seq_files=SEQ_FILES, step_ms=4000):
    try:
        all_start_ms, all_eeg, all_emg, all_mot, all_mxs = load_seq_only(seq_files, step_ms)
        print('Dump data files...')
        for i, start_ms, eeg, emg, mot in enumerate(zip(all_start_ms, all_eeg, all_emg, all_mot)):
            eeg, emg, mot = all_eeg[i], all_emg[i], all_mot[i]
            start_datetime = [ut.convert_ms2datetime(ms) for ms in start_ms]
            joblib.dump((start_datetime, eeg, emg, mot), DUMP_DATA_FILES['infer'][i])

            # phases = []
            # for tmp_start_ms, tmp_eeg, tmp_emg, tmp_mot in zip(start_ms, eeg, emg, mot):
            #     tmp_start_datetime = ut.convert_ms2datetime(tmp_start_ms)
            #     phase = {
            #         "start_ms": tmp_start_ms,
            #         "start_datetime": tmp_start_datetime,
            #         "eeg": tmp_eeg,
            #         "emg": tmp_emg,
            #         "mot": tmp_mot,
            #     }
            #     phases.append(phase)
            # joblib.dump(phases, DUMP_DATA_FILES['infer'][i])
            print(f'Dump data in file {DUMP_DATA_FILES["infer"][i]}')
    except Exception as e:
        raise e


def load_seq_with_labels(seq_files=SEQ_FILES, lb_files=LABEL_FILES):
    print('Processing labels...')
    all_start_ms, all_lbs = load_lbs(lb_files)
    print('Processing sequences...')
    all_eeg, all_emg, all_mot, all_mxs = [[], [], [], []]
    for i, seq_file in enumerate(seq_files):
        start_ms = all_start_ms[i]
        eeg, emg, mot = [[], [], []]
        mxs = [-INF_V, -INF_V, -INF_V]
        tmp_idx = 0
        with open(seq_file, 'r', encoding='utf-8', errors='replace') as f:
            data = f.readlines()
            start_line = 0
            while not data[start_line].__contains__('Time') and start_line < len(data):
                start_line = start_line + 1
            tmp_eeg, tmp_emg, tmp_mot = [[], [], []]
            for line in tqdm(data[start_line + 1:], total=len(data[start_line + 1:]),
                             desc=seq_file.split(PATH_SLASH)[-1]):
                info = line.split('\t')
                if len(info) == 2:  # Final line in raw_S1_EEG1_23 hr.txt
                    dt, values = info[0], (info[1], 0.0, 0.0)  # Fill missing value
                else:
                    dt, values = info[0], (info[1], info[2], info[3])

                for j, value in enumerate(values):
                    if abs(float(value)) > mxs[j]:
                        mxs[j] = abs(float(value))

                ms = ut.convert_datetime2ms(dt)
                if 0 < tmp_idx < len(all_start_ms) - 1 and ms == start_ms[tmp_idx + 1]:
                    eeg.append(tmp_eeg)
                    emg.append(tmp_emg)
                    mot.append(tmp_mot)
                    tmp_idx += 1
                    tmp_eeg, tmp_emg, tmp_mot = [[], [], []]

                tmp_eeg.append(values[0])
                tmp_emg.append(values[1])
                tmp_mot.append(values[2])

            eeg.append(tmp_eeg)
            emg.append(tmp_emg)
            mot.append(tmp_mot)

        all_start_ms.append(start_ms[: tmp_idx + 1])
        all_eeg.append(eeg)
        all_emg.append(emg)
        all_mot.append(mot)
        all_mxs.append(mxs)

    return all_start_ms, all_eeg, all_emg, all_mot, all_lbs, all_mxs


def load_seq_only(data_files=SEQ_FILES, step_ms=None):
    if step_ms is None:
        step_ms = 4000
    print('Processing sequences...')
    all_start_ms, all_eeg, all_emg, all_mot, all_mxs = [[], [], [], [], []]
    for data_file in data_files:
        start_ms, eeg, emg, mot = [[], [], [], []]
        mxs = [-INF_V, -INF_V, -INF_V]
        with open(data_file, 'r', encoding='utf-8', errors='replace') as f:
            data = f.readlines()
            start_line = 0
            while not data[start_line].__contains__('Time') and start_line < len(data):
                start_line = start_line + 1
            tmp_ms = 0
            tmp_eeg, tmp_emg, tmp_mot = [[], [], []]
            for line in tqdm(data[start_line + 1:], total=len(data[start_line + 1:]),
                             desc=data_file.split(PATH_SLASH)[-1]):
                info = line.split('\t')
                if len(info) == 2:  # Final line in raw_S1_EEG1_23 hr.txt
                    dt, values = info[0], (info[1], 0.0, 0.0)  # Fill missing value
                else:
                    dt, values = info[0], (info[1], info[2], info[3])

                for j, value in enumerate(values):
                    if abs(float(value)) > mxs[j]:
                        mxs[j] = abs(float(value))

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

                tmp_eeg.append(values[0])
                tmp_emg.append(values[1])
                tmp_mot.append(values[2])

            start_ms.append(tmp_ms)
            eeg.append(tmp_eeg)
            emg.append(tmp_emg)
            mot.append(tmp_mot)

        all_start_ms.append(start_ms)
        all_eeg.append(eeg)
        all_emg.append(emg)
        all_mot.append(mot)
        all_mxs.append(mxs)

    return all_start_ms, all_eeg, all_emg, all_mot, all_mxs


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
    # os.makedirs(DUMP_DATA_DIR, exist_ok=True)
    # dump_seq_with_labels()
    # load_seq_only(step_ms=4000)
    load_seq_with_labels()
