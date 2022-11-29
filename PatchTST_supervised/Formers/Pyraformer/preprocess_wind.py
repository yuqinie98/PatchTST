import numpy as np
from numpy.core.defchararray import split
import pandas as pd
from datetime import datetime
from scipy import stats
import os


def load_data(datadir):
    df = pd.read_csv(datadir)
    data = (df.values).transpose(1, 0)

    return data


def get_covariates(data_len, start_day):
    """Get covariates"""
    start_timestamp = datetime.timestamp(datetime.strptime(start_day, '%Y-%m-%d %H:%M:%S'))
    timestamps = np.arange(data_len) * 3600 + start_timestamp
    timestamps = [datetime.fromtimestamp(i) for i in timestamps]

    weekdays = stats.zscore(np.array([i.weekday() for i in timestamps]))
    hours = stats.zscore(np.array([i.hour for i in timestamps]))
    months = stats.zscore(np.array([i.month for i in timestamps]))

    covariates = np.stack([weekdays, hours, months], axis=1)

    return covariates


def split_seq(sequences, covariates, seq_length, slide_step, predict_length, save_dir):
    """Divide the training sequence into windows"""
    data_length = len(sequences[0])
    windows = (data_length-seq_length+slide_step) // slide_step
    train_windows = int(0.97 * windows)
    test_windows = windows - train_windows
    train_data = np.zeros((train_windows*len(sequences), seq_length+predict_length-1, 5), dtype=np.float32)
    test_data = np.zeros((test_windows*len(sequences), seq_length+predict_length-1, 5), dtype=np.float32)

    count = 0
    split_start = 0
    seq_ids = np.arange(len(sequences))[:, None]
    end = split_start + seq_length + predict_length - 1
    while end <= data_length:
        if count < train_windows:
            train_data[count*len(sequences):(count+1)*len(sequences), :, 0] = sequences[:, split_start:end]
            train_data[count*len(sequences):(count+1)*len(sequences), :, 1:4] = covariates[split_start:end, :]
            train_data[count*len(sequences):(count+1)*len(sequences), :, -1] = seq_ids
        else:
            test_data[(count-train_windows)*len(sequences):(count-train_windows+1)*len(sequences), :, 0] = sequences[:, split_start:end]
            test_data[(count-train_windows)*len(sequences):(count-train_windows+1)*len(sequences), :, 1:4] = covariates[split_start:end, :]
            test_data[(count-train_windows)*len(sequences):(count-train_windows+1)*len(sequences), :, -1] = seq_ids

        count += 1
        split_start += slide_step
        end = split_start + seq_length + predict_length - 1

    os.makedirs(save_dir, exist_ok=True)

    train_data, v = normalize(train_data, seq_length)
    save(train_data, v, save_dir + 'train')
    test_data, v = normalize(test_data, seq_length)
    save(test_data, v, save_dir + 'test')


def normalize(inputs, seq_length):
    base_seq = inputs[:, :seq_length, 0]
    nonzeros = (base_seq > 0).sum(1)
    inputs = inputs[nonzeros > 0]

    base_seq = inputs[:, :seq_length, 0]
    nonzeros = nonzeros[nonzeros > 0]
    v = base_seq.sum(1) / nonzeros
    v[v == 0] = 1
    inputs[:, :, 0] = inputs[:, :, 0] / v[:, None]

    return inputs, v


def save(data, v, save_dir):
    np.save(save_dir+'_data_wind.npy', data)
    np.save(save_dir+'_v_wind.npy', v)


if __name__ == '__main__':
    datadir = 'data/EMHIRESPV_TSh_CF_Country_19862015.csv'
    all_data = load_data(datadir)
    covariates = get_covariates(len(all_data[0]), '1986-01-01 00:00:00')
    split_seq(all_data, covariates, 192, 24, 24, 'data/wind/')
