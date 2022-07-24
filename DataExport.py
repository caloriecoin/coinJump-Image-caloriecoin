import os
import tarfile
import urllib
import pandas as pd

DOWNLOAD_ROOT = ""
PATH = os.path.join("", "")
URL = DOWNLOAD_ROOT + ""

def fetch_data(data_url=URL, data_path=PATH):
    os.makedirs(PATH, exist=True)
    tgz_path = os.path.join(data_path, "")
    urllib.request.urlretrieve(data_url, tgz_path)
    data_tgz = tarfile.open(tgz_path)
    data_tgz.extractall(path=data_path)
    data_tgz.close()


def load_data(data_path=PATH):
    csv_path = os.path.join(PATH, "")
    return pd.read_csv(csv_path)
