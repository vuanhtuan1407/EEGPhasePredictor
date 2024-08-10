from datetime import datetime


def convert_time2ms(time_str: str, offset=946659600000):
    FORMAT_IN_DATA_FILE = '%Y.%m.%d.  %H:%M:%S.%f'
    FORMAT1_IN_LABEL_FILE = '%Y.%m.%d.  %H:%M:%S'
    if time_str.__contains__("/"):
        FORMAT_IN_DATA_FILE = '%m/%d/%Y  %H:%M:%S.%f'
        FORMAT1_IN_LABEL_FILE = '%m/%d/%Y  %H:%M:%S'

    try:
        if time_str[-5:].__contains__("."):
            dt_obj = datetime.strptime(time_str, FORMAT_IN_DATA_FILE)
        else:
            dt_obj = datetime.strptime(time_str, FORMAT1_IN_LABEL_FILE)

        ms = int(dt_obj.timestamp() * 1000) - offset
    except:
        ms = -1
    return ms
