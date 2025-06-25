import numpy as np
import requests
import os


def download(data_set: str, save_dir: str):

    os.makedirs(save_dir, exist_ok=True)

    if data_set is '21cm':
        files = ['Par_test_21cmGEM.txt', 'Par_train_21cmGEM.txt', 
                'T21_test_21cmGEM.txt', 'T21_train_21cmGEM.txt']
        saves = ['test_data.txt', 'train_data.txt', 
                'test_labels.txt', 'train_labels.txt']

        for i in range(len(files)):
            url = 'https://zenodo.org/record/4541500/files/' + files[i]
            with open(save_dir + saves[i], 'wb') as f:
                f.write(requests.get(url).content)