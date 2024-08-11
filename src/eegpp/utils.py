from datetime import datetime


def convert_datetime2ms(datetime_str: str, offset=946659600000):
    FORMAT_IN_DATA_FILE = '%Y.%m.%d.  %H:%M:%S.%f'
    FORMAT1_IN_LABEL_FILE = '%Y.%m.%d.  %H:%M:%S'
    if datetime_str.__contains__("/"):
        FORMAT_IN_DATA_FILE = '%m/%d/%Y  %H:%M:%S.%f'
        FORMAT1_IN_LABEL_FILE = '%m/%d/%Y  %H:%M:%S'

    try:
        if datetime_str[-5:].__contains__("."):
            dt_obj = datetime.strptime(datetime_str, FORMAT_IN_DATA_FILE)
        else:
            dt_obj = datetime.strptime(datetime_str, FORMAT1_IN_LABEL_FILE)

        ms = int(dt_obj.timestamp() * 1000) - offset
    except:
        ms = -1
    return ms


def convert_ms2datetime(ms, offset=946659600000):
    return str(datetime.fromtimestamp((ms + offset) / 1000))


if __name__ == '__main__':
    print(convert_ms2datetime(686505744000))
    print(convert_datetime2ms("2021.10.02. 21:59:52"))
    print((convert_datetime2ms("2021.10.02. 21:59:52") - convert_datetime2ms("2021.10.02. 10:06:12")) / 4000 + 1)
