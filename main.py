import math

import torch

from src.eegpp.train import train

if __name__ == '__main__':
    train()
    # s = torch.tensor([
    #     [[0, 0, 1, 2, 1, 3, 4, 5]],
    #     [[1, 2, 3, 4, 5, 6, 7, 7]],
    #     [[0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4]]
    # ], dtype=torch.float32)
    # # s = torch.transpose(s, 1, 2)
    # s = s.reshape(s.shape[0], -1)
    # print(s.shape)
    # c = torch.fft.fft(s)
    # print(c, c.shape)
    # print(c.real)
    # print(c.imag)
    # print(torch.sqrt(c.real**2 + c.imag**2))
    # c2 = torch.fft.fft(torch.tensor([0, 1, 2, 3]))
    # print(c2, c2.shape, c2.real, c2.imag)
    # from src.eegpp.data.data_utils import LABEL_DICT
    # print(len(LABEL_DICT))
    # train()
    # import yaml
    # config = yaml.safe_load(open('./src/eegpp/configs/cnn1d_config.yml'))
    # print(config['conv_layers'])
    # for layer in config['conv_layers']:
    #     print(layer['out_channels'])
    # import joblib
    # value_seqs, label_seqs, mx, misc = joblib.load('./src/eegpp/data/dump_eeg_1.pkl')
    # print(len(value_seqs[1]), len(label_seqs), len(mx))
    # start_dt, eeg, emg, mot, lbs, mxs = joblib.load('./src/eegpp/data/dump/dump_eeg_1.pkl')
    # print(len(start_dt), len(eeg), len(emg), len(mot), len(lbs), len(mxs))
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
