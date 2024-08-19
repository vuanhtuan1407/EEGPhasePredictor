import numpy as np
import torch

# from src.eegpp.train import train

if __name__ == '__main__':
    # from src.eegpp.data.data_utils import LABEL_DICT
    # print(len(LABEL_DICT))
    # train()
    # import yaml
    # config = yaml.safe_load(open('./src/eegpp/configs/cnn1d_config.yml'))
    # print(config['conv_layers'])
    # for layer in config['conv_layers']:
    #     print(layer['out_channels'])
    import joblib
    value_seqs, label_seqs, mx, misc = joblib.load('./src/eegpp/data/dump_eeg_1.pkl')
    print(len(value_seqs[1]), len(label_seqs), len(mx))
    start_dt, eeg, emg, mot, lbs, mxs = joblib.load('./src/eegpp/data/dump/dump_eeg_1.pkl')
    print(len(start_dt), len(eeg), len(emg), len(mot), len(lbs), len(mxs))
    # print(np.array(start_datetime).shape, np.array(eeg).shape, np.array(emg).shape, np.array(mot).shape, np.array(lbs).shape, np.array(mxs).shape)
    # print(phases[0])
    # t = []
    # for i in range(3):
    #     a = torch.randint(1, 10, (3, 4))
    #     b = torch.randint(1, 10, (3, 4))
    #     c = torch.randint(1, 10, (3, 4))
    #     # print(a, b, c)
    #     s = torch.concat([a, b, c], dim=-1)
    #     t.append(s)
    #     print(s.shape)
    # t = torch.concat(t)
    # # print(t, t.shape)
    # # t = torch.tile(t, (3, 1))
    # # print(t, t.shape)
    # t = t.unsqueeze(1)
    # # print(t, t.shape)
    # t = t[:, :, 2, :]
    # # print(t, t.shape)
    # t = torch.squeeze(t, 2)
    # # print(t, t.shape)
    # # print(t2)
    # t = torch.tensor(t, dtype=torch.float32)
    # # t2 = torch.rand((3, 2, 1), dtype=torch.float32)
    # conv = torch.nn.Conv1d(in_channels=1, out_channels=2, kernel_size=1)
    # print(conv(t).shape)
