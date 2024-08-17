import platform
from datetime import datetime
from pathlib import Path

import yaml

from src.eegpp.configs import CONFIG_DIR

FORMAT1 = '%Y.%m.%d.  %H:%M:%S.%f'  # use in original seq files
FORMAT2 = '%Y.%m.%d.  %H:%M:%S'  # use in original label files and use as default format


def convert_datetime2ms(datetime_str: str, offset=946659600000):
    if datetime_str.__contains__("/"):
        format_seq = '%m/%d/%Y  %H:%M:%S.%f'
        format_lb = '%m/%d/%Y  %H:%M:%S'
    else:
        format_seq = FORMAT1
        format_lb = FORMAT2

    try:
        if datetime_str[-5:].__contains__("."):
            dt_obj = datetime.strptime(datetime_str, format_seq)
        else:
            dt_obj = datetime.strptime(datetime_str, format_lb)

        ms = int(dt_obj.timestamp() * 1000) - offset
    except:
        ms = -1
    return ms


def convert_ms2datetime(ms, offset=946659600000):
    dt = datetime.fromtimestamp((ms + offset) / 1000)
    return str(datetime.strftime(dt, FORMAT2))


def load_config_yaml(yaml_file):
    yml_config_path = str(Path(CONFIG_DIR, yaml_file))
    with open(yml_config_path, 'r') as f:
        return yaml.safe_load(f)


def get_os():
    return platform.system()


def get_path_slash():
    os_system = platform.system()
    if os_system == 'Windows':
        return '\\'
    else:
        return '/'


if __name__ == '__main__':
    print(convert_ms2datetime(686505744000))
    print(convert_datetime2ms("2021.10.02. 21:59:52"))
    print((convert_datetime2ms("2021.10.02. 21:59:52") - convert_datetime2ms("2021.10.02. 10:06:12")) / 4000 + 1)
