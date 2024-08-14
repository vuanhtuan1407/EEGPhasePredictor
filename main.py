import torch

# from src.eegpp.train import train

if __name__ == '__main__':
    # train()
    # import yaml
    # config = yaml.safe_load(open('./src/eegpp/configs/cnn1d_3c_config.yml'))
    # print(config['conv_layers'])
    # import joblib
    # phases = joblib.load('./src/eegpp/data/dump/dump_eeg_1.pkl')
    # print(phases[0])
    t = []
    for i in range(3):
        a = torch.randint(1, 10, (1, 3, 4))
        b = torch.randint(1, 10, (1, 3, 4))
        c = torch.randint(1, 10, (1, 3, 4))
        # print(a, b, c)
        t.append(torch.concat([a, b, c], dim=-1))
    t = torch.concat(t)
    # print(t, t.shape)
    # t = torch.tile(t, (3, 1))
    # print(t, t.shape)
    t = t.unsqueeze(1)
    # print(t, t.shape)
    t = t[:, :, 2, :]
    # print(t, t.shape)
    t = torch.squeeze(t, 2)
    # print(t, t.shape)
    # print(t2)
    t = torch.tensor(t, dtype=torch.float32)
    # t2 = torch.rand((3, 2, 1), dtype=torch.float32)
    conv = torch.nn.Conv1d(in_channels=1, out_channels=2, kernel_size=1)
    print(conv(t).shape)
