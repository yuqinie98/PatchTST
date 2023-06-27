import os
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler
from utils.timefeatures import time_features
import numpy as np
import torch

import warnings
warnings.filterwarnings('ignore')


"""Long range dataloader"""
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', dataset='ETTh1', inverse=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=1, freq='h')

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, self.scaler.mean, self.scaler.std
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data, seq_y, mean, std):
        return self.scaler.inverse_transform(data), seq_y


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTm1.csv', dataset='ETTm1', inverse=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=1, freq='h')
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, self.scaler.mean, self.scaler.std
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data, seq_y, mean, std):
        return self.scaler.inverse_transform(data), seq_y



class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, dataset='',timeenc=0, freq='h',inverse=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark,self.scaler.mean, self.scaler.std

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    


# """Long range dataloader for dataset elect and app flow"""
class Dataset_Custom2(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', dataset='elect', 
                inverse=False):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test']
        self.flag = flag

        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        preprocess_path = os.path.join(self.root_path, self.data_path)
        self.all_data, self.covariates, self.train_end = eval('preprocess_flow')(preprocess_path)
        self.all_data = torch.from_numpy(self.all_data).transpose(0, 1)
        self.covariates = torch.from_numpy(self.covariates)
        self.test_start = self.train_end - self.seq_len + 1
        self.window_stride = 24
        self.seq_num = self.all_data.size(0)

    def fit(self, data):
        mean = data.mean()
        std = data.std()
        return mean, std

    def inverse_transform(self, output, seq_y, mean, std):
        output = output *  (mean.unsqueeze(1).unsqueeze(1) + 1)
        seq_y = seq_y * (mean.unsqueeze(1).unsqueeze(1) + 1)
        return output, seq_y

    def __len__(self):
        if self.flag == 'train':
            self.window_per_seq = (self.train_end - self.seq_len - self.pred_len) // self.window_stride
            return self.window_per_seq * self.seq_num
        else:
            self.window_per_seq = (self.all_data.size(1) - self.test_start - self.seq_len - self.pred_len) // self.window_stride
            return self.window_per_seq * self.seq_num

    def __getitem__(self, index):
        seq_idx = index // self.window_per_seq
        window_idx = index % self.window_per_seq

        if self.flag == 'train':
            s_begin = window_idx * self.window_stride
        else:
            s_begin = self.test_start + window_idx * self.window_stride

        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.all_data[seq_idx, s_begin:s_end].clone()
        seq_y = self.all_data[seq_idx, r_begin:r_end].clone()
        mean, std = self.fit(seq_x)
        if mean > 0:
            seq_x = seq_x / (mean + 1)
            seq_y = seq_y / (mean + 1)

        if len(self.covariates.size()) == 2:
            seq_x_mark = self.covariates[s_begin:s_end]
            seq_x_mark[:, -1] = int(seq_idx)
            seq_y_mark = self.covariates[r_begin:r_end]
            seq_y_mark[:, -1] = int(seq_idx)
        else:
            seq_x_mark = self.covariates[s_begin:s_end, seq_idx]
            seq_x_mark[:, -1] = int(seq_idx)
            seq_y_mark = self.covariates[r_begin:r_end, seq_idx]
            seq_y_mark[:, -1] = int(seq_idx)

        return seq_x.unsqueeze(1), seq_y.unsqueeze(1), seq_x_mark, seq_y_mark, mean, std


"""Long range dataloader for synthetic dataset"""
class Dataset_Synthetic(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='synthetic.npy', dataset='synthetic', inverse=False):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test']
        self.flag = flag
        self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path
        preprocess_path = os.path.join(self.root_path, self.data_path)
        self.all_data =np.load(preprocess_path)
        self.all_data = torch.from_numpy(self.all_data)
        self.all_data, self.covariates = self.all_data[:, :, 0], self.all_data[:, :, 1:]
        self.seq_num = self.all_data.size(0)

        self.window_stride = 24
        window_per_seq = (self.all_data.shape[1] - self.seq_len - self.pred_len) / self.window_stride
        self.train_end = self.seq_len + self.pred_len + int(0.9 * window_per_seq) * self.window_stride
        self.test_start = self.train_end - self.seq_len + 1

    def fit(self, data):
        mean = data.mean()
        std = data.std()
        return mean, std

    def inverse_transform(self, output, seq_y, mean, std):
        output = output *  (mean.unsqueeze(1).unsqueeze(1) + 1)
        seq_y = seq_y * (mean.unsqueeze(1).unsqueeze(1) + 1)
        return output, seq_y

    def __len__(self):
        if self.flag == 'train':
            self.window_per_seq = (self.train_end - self.seq_len - self.pred_len) // self.window_stride
            return self.window_per_seq * self.seq_num
        else:
            self.window_per_seq = (self.all_data.size(1) - self.test_start - self.seq_len - self.pred_len) // self.window_stride
            return self.window_per_seq * self.seq_num

    def __getitem__(self, index):
        seq_idx = index // self.window_per_seq
        window_idx = index % self.window_per_seq

        if self.flag == 'train':
            s_begin = window_idx * self.window_stride
        else:
            s_begin = self.test_start + window_idx * self.window_stride

        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.all_data[seq_idx, s_begin:s_end].clone()
        seq_y = self.all_data[seq_idx, r_begin:r_end].clone()

        mean, std = self.fit(seq_x)
        if mean > 0:
            seq_x = seq_x / (mean + 1)
            seq_y = seq_y / (mean + 1)

        seq_x_mark = self.covariates[seq_idx, s_begin:s_end]
        seq_y_mark = self.covariates[seq_idx, r_begin:r_end]

        return seq_x.unsqueeze(1), seq_y.unsqueeze(1), seq_x_mark, seq_y_mark, mean, std


def get_all_v(train_data, train_end, seq_len, pred_len, window_stride, type):
    """Get the normalization parameters of each sequence"""
    seq_num = train_data.size(0)
    window_per_seq = (train_end - seq_len - pred_len) // window_stride
    window_number = seq_num * window_per_seq

    v = torch.zeros(window_number, dtype=torch.float64)
    for index in range(window_number):
        seq_idx = index // window_per_seq
        window_idx = index % window_per_seq

        s_begin = window_idx * window_stride
        s_end = s_begin + seq_len

        seq_x = train_data[seq_idx, s_begin:s_end].clone()
        if type == 'mean':
            mean = seq_x.mean()
            v[index] = mean + 1
        else:
            std = seq_x.std()
            v[index] = std

    return v


def gen_covariates(times, num_covariates):
    """Get covariates"""
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 0] = input_time.weekday() / 7
        covariates[i, 1] = input_time.hour / 24
        covariates[i, 2] = input_time.month / 12

    return covariates


def preprocess_elect(csv_path):
    """preprocess the elect dataset for long range forecasting"""
    num_covariates = 4
    train_start = '2011-01-01 00:00:00'
    train_end = '2014-04-01 23:00:00'
    test_start = '2014-04-01 00:00:00'
    test_end = '2014-09-07 23:00:00'

    data_frame = pd.read_csv(csv_path, sep=";", index_col=0, parse_dates=True, decimal=',')
    data_frame = data_frame.resample('1H',label = 'left',closed = 'right').sum()[train_start:test_end]
    data_frame.fillna(0, inplace=True)

    covariates = gen_covariates(data_frame[train_start:test_end].index, num_covariates)
    all_data = data_frame[train_start:test_end].values
    data_start = (all_data!=0).argmax(axis=0) #find first nonzero value in each time series
    train_end = len(data_frame[train_start:train_end].values)

    all_data = all_data[:, data_start < 10000]
    data_start = data_start[data_start < 10000]
    split_start = data_start.max()
    all_data = all_data[split_start:]
    covariates = covariates[split_start:]
    train_end = train_end - split_start

    return all_data.astype(np.float32), covariates.astype(np.float32), train_end


def preprocess_flow(csv_path):
    """preprocess the app flow dataset for long range forecasting"""
    data_frame = pd.read_csv(csv_path, names=['app_name', 'zone', 'time', 'value'], parse_dates=True)
    grouped_data = list(data_frame.groupby(["app_name", "zone"]))
    # covariates = gen_covariates(data_frame.index, 3)
    all_data = []
    min_length = 10000
    for i in range(len(grouped_data)):
        single_df = grouped_data[i][1].drop(labels=['app_name', 'zone'], axis=1).sort_values(by="time", ascending=True)
        times = pd.to_datetime(single_df.time)
        single_df['weekday'] = times.dt.dayofweek / 7
        single_df['hour'] = times.dt.hour / 24
        single_df['month'] = times.dt.month / 12
        temp_data = single_df.values[:, 1:]
        if (temp_data[:, 0] == 0).sum() / len(temp_data) > 0.2 or len(temp_data) < 3000:
            continue

        if len(temp_data) < min_length:
            min_length = len(temp_data)

        all_data.append(temp_data)

    all_data = np.array([data[len(data)-min_length:, :] for data in all_data]).transpose(1, 0, 2).astype(np.float32)
    train_end = min(int(0.8 * min_length), min_length - 1000)
    covariates = all_data.copy()
    covariates[:, :, :-1] = covariates[:, :, 1:]

    return all_data[:, :, 0], covariates, train_end


"""Single step dataloader"""
def split(split_start, label, cov, pred_length):
    all_data = []
    for batch_idx in range(len(label)):
        batch_label = label[batch_idx]
        for i in range(pred_length):
            single_data = batch_label[i:(split_start+i)].clone().unsqueeze(1)
            single_data[-1] = -1
            single_cov = cov[batch_idx, i:(split_start+i), :].clone()
            temp_data = [single_data, single_cov]
            single_data = torch.cat(temp_data, dim=1)
            all_data.append(single_data)
    data = torch.stack(all_data, dim=0)
    label = label[:, -pred_length:].reshape(pred_length*len(label))

    return data, label


"""Single step training dataloader for the electricity dataset"""
class electTrainDataset(Dataset):
    def __init__(self, data_path, data_name, predict_length, batch_size):
        self.data = torch.from_numpy(np.load(os.path.join(data_path, f'train_data_{data_name}.npy')))

        # Resample windows according to the average amplitude
        v = np.load(os.path.join(data_path, f'train_v_{data_name}.npy'))
        weights = torch.as_tensor(np.abs(v[:,0])/np.sum(np.abs(v[:,0])), dtype=torch.double)
        num_samples = weights.size(0)
        sample_index = torch.multinomial(weights, num_samples, True)
        self.data = self.data[sample_index]

        self.label = torch.from_numpy(np.load(os.path.join(data_path, f'train_label_{data_name}.npy')))
        self.label = self.label[sample_index]

        self.train_len = len(self.data) // batch_size
        self.pred_length = predict_length
        self.batch_size = batch_size

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        if (index+1) <= self.train_len:
            all_data = self.data[index*self.batch_size:(index+1)*self.batch_size].clone()
            label = self.label[index*self.batch_size:(index+1)*self.batch_size].clone()
        else:
            all_data = self.data[index*self.batch_size:].clone()
            label = self.label[index*self.batch_size:].clone()

        cov = all_data[:, :, 2:]

        split_start = len(label[0]) - self.pred_length + 1
        data, label = split(split_start, label, cov, self.pred_length)

        return data, label


"""Single step testing dataloader for the electricity dataset"""
class electTestDataset(Dataset):
    def __init__(self, data_path, data_name, predict_length):
        self.data = np.load(os.path.join(data_path, f'test_data_{data_name}.npy'))
        self.v = np.load(os.path.join(data_path, f'test_v_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, f'test_label_{data_name}.npy'))
        self.test_len = self.data.shape[0]
        self.pred_length = predict_length

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        all_data = torch.from_numpy(self.data[index].copy())
        cov = all_data[:, 2:]
        label = torch.from_numpy(self.label[index].copy())
        v = float(self.v[index][0])
        if v > 0:
            data = label / v
        else:
            data = label

        split_start = len(label) - self.pred_length + 1
        all_data = []
        for i in range(self.pred_length):
            single_data = data[i:(split_start+i)].clone().unsqueeze(1)
            single_data[-1] = -1
            single_cov = cov[i:(split_start+i), :].clone()
            single_data = torch.cat([single_data, single_cov], dim=1)
            all_data.append(single_data)
        all_data = torch.stack(all_data, dim=0)
        label = label[-self.pred_length:]

        return all_data, label, v


"""Single step training dataloader for the app flow dataset"""
class flowTrainDataset(Dataset):
    def __init__(self, data_path, data_name, predict_length, batch_size):
        self.data = torch.from_numpy(np.load(os.path.join(data_path, f'train_data_{data_name}.npy')))

        # Resample windows according to the average amplitude
        v = np.load(os.path.join(data_path, f'train_v_{data_name}.npy'))
        weights = torch.as_tensor(np.abs(v)/np.sum(np.abs(v)), dtype=torch.double)
        num_samples = weights.size(0)
        sample_index = torch.multinomial(weights, num_samples, True)
        self.data = self.data[sample_index]

        self.label = self.data[:, :, 0]

        self.train_len = len(self.data) // batch_size
        self.pred_length = predict_length
        self.batch_size = batch_size

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        if (index+1) <= self.train_len:
            all_data = self.data[index*self.batch_size:(index+1)*self.batch_size].clone()
            label = self.label[index*self.batch_size:(index+1)*self.batch_size].clone()
        else:
            all_data = self.data[index*self.batch_size:].clone()
            label = self.label[index*self.batch_size:].clone()

        cov = all_data[:, :, 1:]

        split_start = len(label[0]) - self.pred_length + 1
        data, label = split(split_start, label, cov, self.pred_length)

        return data, label


"""Single step testing dataloader for the all flow dataset"""
class flowTestDataset(Dataset):
    def __init__(self, data_path, data_name, predict_length):
        self.data = np.load(os.path.join(data_path, f'test_data_{data_name}.npy'))
        self.v = np.load(os.path.join(data_path, f'test_v_{data_name}.npy'))
        self.label = self.data
        self.test_len = self.data.shape[0]
        self.pred_length = predict_length

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        all_data = torch.from_numpy(self.data[index].copy())
        cov = all_data[:, 1:]
        data = all_data[:, 0]
        label = torch.from_numpy(self.label[index, :, 0].copy())
        v = float(self.v[index])

        split_start = len(label) - self.pred_length + 1
        all_data = []
        for i in range(self.pred_length):
            single_data = data[i:(split_start+i)].clone().unsqueeze(1)
            single_data[-1] = -1
            single_cov = cov[i:(split_start+i), :].clone()
            single_data = torch.cat([single_data, single_cov], dim=1)
            all_data.append(single_data)
        all_data = torch.stack(all_data, dim=0)
        label = label[-self.pred_length:] * v

        return all_data, label, v


"""Single step training dataloader for the wind dataset"""
class windTrainDataset(Dataset):
    def __init__(self, data_path, data_name, predict_length, batch_size):
        self.data = torch.from_numpy(np.load(os.path.join(data_path, f'train_data_{data_name}.npy')))

        # Resample windows according to the average amplitude
        v = np.load(os.path.join(data_path, f'train_v_{data_name}.npy'))
        weights = torch.as_tensor(np.abs(v)/np.sum(np.abs(v)), dtype=torch.double)
        num_samples = weights.size(0)
        sample_index = torch.multinomial(weights, num_samples, True)
        self.data = self.data[sample_index]

        self.train_len = len(self.data) // batch_size
        self.pred_length = predict_length
        self.batch_size = batch_size

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        if (index+1) <= self.train_len:
            all_data = self.data[index*self.batch_size:(index+1)*self.batch_size].clone()
        else:
            all_data = self.data[index*self.batch_size:].clone()

        cov = all_data[:, :, 1:]
        label = all_data[:, :, 0]

        split_start = len(label[0]) - self.pred_length + 1
        data, label = split(split_start, label, cov, self.pred_length)

        return data, label


"""Single step testing dataloader for the wind dataset"""
class windTestDataset(Dataset):
    def __init__(self, data_path, data_name, predict_length):
        self.data = np.load(os.path.join(data_path, f'test_data_{data_name}.npy'))
        self.v = np.load(os.path.join(data_path, f'test_v_{data_name}.npy'))
        self.test_len = self.data.shape[0]
        self.pred_length = predict_length

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        all_data = torch.from_numpy(self.data[index].copy())
        cov = all_data[:, 1:]
        data = all_data[:, 0]
        v = float(self.v[index])
        label = data * v

        split_start = len(label) - self.pred_length + 1
        all_data = []
        for i in range(self.pred_length):
            single_data = data[i:(split_start+i)].clone().unsqueeze(1)
            single_data[-1] = -1
            single_cov = cov[i:(split_start+i), :].clone()
            single_data = torch.cat([single_data, single_cov], dim=1)
            all_data.append(single_data)
        all_data = torch.stack(all_data, dim=0)
        label = label[-self.pred_length:]

        return all_data, label, v

