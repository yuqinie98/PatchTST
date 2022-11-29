from numpy.lib.npyio import save
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import trange
import zipfile


def load_data(filedir):
    data_frame = pd.read_csv(filedir, header=0, parse_dates=True)  #names=['app_name', 'zone', 'time', 'value']
    data_frame = data_frame.drop(data_frame.columns[0], axis=1)
    grouped_data = list(data_frame.groupby(["app_name", "zone"]))
    # covariates = gen_covariates(data_frame.index, 3)
    all_data = []
    for i in range(len(grouped_data)):
        single_df = grouped_data[i][1].drop(labels=['app_name', 'zone'], axis=1).sort_values(by="time", ascending=True)
        times = pd.to_datetime(single_df.time)
        single_df['weekday'] = times.dt.dayofweek / 6
        single_df['hour'] = times.dt.hour / 23
        single_df['month'] = times.dt.month / 12
        temp_data = single_df.values[:, 1:]
        if (temp_data[:, 0] == 0).sum() / len(temp_data) > 0.2:
            continue

        all_data.append(temp_data)

    return all_data


def visualize(data, index, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(index):
        x = np.arange(len(data[i]))
        f = plt.figure()
        plt.plot(x, data[i][:, 0])
        f.savefig(os.path.join(save_dir, "visual_{}.png".format(i)))
        plt.close()


def split_seq(sequences, seq_length, slide_step, predict_length, save_dir):
    """Divide the training sequence into windows"""
    train_data = []
    test_data = []
    for seq_id in trange(len(sequences)):
        split_start = 0
        single_seq = sequences[seq_id][:, 0]
        single_covariate = sequences[seq_id][:, 1:]
        windows = (len(single_seq)-seq_length+slide_step) // slide_step
        count = 0
        train_count = int(0.97 * windows)
        while len(single_seq[split_start:]) > (seq_length + predict_length):
            seq_data = single_seq[split_start:(split_start+seq_length+predict_length-1)]
            single_data = np.zeros((seq_length+predict_length-1, 5))
            single_data[:, 0] = seq_data.copy()
            single_data[:, 1:4] = single_covariate[split_start:(split_start+seq_length+predict_length-1)]
            single_data[:, -1] = seq_id

            count += 1
            if count < train_count:
                train_data.append(single_data)
            else:
                test_data.append(single_data)
            split_start += slide_step

    os.makedirs(save_dir, exist_ok=True)

    train_data = np.array(train_data, dtype=np.float32)
    train_data, v = normalize(train_data, seq_length)
    save(train_data, v, save_dir + 'train')
    test_data = np.array(test_data, dtype=np.float32)
    test_data, v = normalize(test_data, seq_length)
    save(test_data, v, save_dir + 'test')


def normalize(inputs, seq_length):
    base_seq = inputs[:, :(seq_length-1), 0]
    nonzeros = (base_seq > 0).sum(1)
    v = base_seq.sum(1) / nonzeros
    v[v == 0] = 1
    inputs[:, :, 0] = inputs[:, :, 0] / v[:, None]

    return inputs, v


def save(data, v, save_dir):
    np.save(save_dir+'_data_flow.npy', data)
    np.save(save_dir+'_v_flow.npy', v)


def dezip(filedir):
    zip_file = zipfile.ZipFile(filedir)
    zip_list = zip_file.namelist()

    parent_dir = filedir.split('/')[0]
    for f in zip_list:
        zip_file.extract(f, parent_dir)

    zip_file.close()


if __name__ == '__main__':
    zip_dir = 'data/app_zone_rpc_hour_encrypted.zip'
    dezip(zip_dir)
    data_dir = 'data/app_zone_rpc_hour_encrypted.csv'
    data = load_data(data_dir)
    split_seq(data, 192, 24, 24, 'data/flow/')
