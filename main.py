import os

# from src.eegpp.train import train

if __name__ == '__main__':
    # train()
    import yaml
    config = yaml.safe_load(open('./src/eegpp/configs/cnn1d_3c_config.yml'))
    print(config['conv_layers'])
