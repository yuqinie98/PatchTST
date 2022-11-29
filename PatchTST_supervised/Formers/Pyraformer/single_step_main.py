import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
import os

import pyraformer.Pyraformer_SS as Pyraformer
from data_loader import *
import os
from utils.tools import SingleStepLoss as LossFactory
from utils.tools import AE_loss


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    data_dir = opt.data_path
    dataset = opt.dataset
    train_set = eval(dataset+'TrainDataset')(data_dir, dataset, opt.predict_step, opt.inner_batch)
    test_set = eval(dataset+'TestDataset')(data_dir, dataset, opt.predict_step)
    train_sampler = RandomSampler(train_set)
    test_sampler = RandomSampler(test_set)

    trainloader = DataLoader(train_set, batch_size=1, sampler=train_sampler, num_workers=0)
    testloader = DataLoader(test_set, batch_size=1, sampler=test_sampler, num_workers=0)

    return trainloader, testloader


def get_dataset_parameters(opt):
    """Prepare specific parameters for different datasets"""
    dataset2num = {
        'elect': 370,
        'flow': 1083,
        'wind': 29
    }
    dataset2covariate = {
        'elect':3,
        'flow': 3,
        'wind': 3
    }
    dataset2input_len = {
        'elect':169,
        'flow': 192,
        'wind': 192
    }
    dataset2ignore_zero = {
        'elect': True,
        'flow': True,
        'wind': False
    }

    opt.num_seq = dataset2num[opt.dataset]
    opt.covariate_size = dataset2covariate[opt.dataset]
    opt.input_size = dataset2input_len[opt.dataset]
    opt.ignore_zero = dataset2ignore_zero[opt.dataset]
    return opt


def get_topk(epoch, batch_size):
    if epoch <= 1:
        topk = 0
    elif 1 < epoch < 4:
        topk = int(batch_size * (5 - epoch) / (6 - epoch))
    else:
        topk = int(batch_size * 0.5)

    return topk


def train_epoch(model, training_data, optimizer, opt, epoch):
    """ Epoch operation in training phase. """
    model.train()

    total_likelihood = 0
    total_mse = 0
    total_pred_number = 0
    index = 0

    criterion = LossFactory(opt.ignore_zero)
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        sequence, label = map(lambda x: x.to(opt.device).squeeze(0), batch)

        optimizer.zero_grad()

        mean_pre, sigma_pre = model(sequence)

        if epoch == 0 and opt.pretrain:
            full_label = sequence[:, :, 0].clone()
            full_label[:, -1] = label
            likelihood_losses, mse_losses = criterion(mean_pre, sigma_pre, full_label, 0)
            mean_pre = mean_pre[:, -1]
            sigma_pre = sigma_pre[:, -1]
        else:
            if opt.hard_sample_mining:
                topk = get_topk(epoch, len(sequence))
            else:
                topk = 0
            mean_pre = mean_pre[:, -1]
            sigma_pre = sigma_pre[:, -1]
            likelihood_losses, mse_losses = criterion(mean_pre, sigma_pre, label, topk)

        likelihood_loss = likelihood_losses.mean()
        mse_loss = mse_losses.mean()

        if index % opt.visualize_fre == 0:
            print('Likelihood loss:{}, MSE loss:{}'.format(likelihood_loss, mse_loss))

        loss = likelihood_loss + 100 * mse_loss
        loss.backward()
        index += 1
        total_likelihood += likelihood_losses.sum().item()
        total_mse += mse_losses.sum().item()
        total_pred_number += likelihood_losses.numel()

        optimizer.step()

    return total_likelihood / total_pred_number, total_mse / total_pred_number


def eval_epoch(model, validation_data, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()
    total_likelihood = 0
    total_se = 0
    total_ae = 0
    total_label = 0
    total_pred_num = 0
    index = 0
    criterion = LossFactory(opt.ignore_zero)
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            sequence, label, v = map(lambda x: x.to(opt.device).squeeze(0), batch)

            """ forward """
            mu_pre, sigma_pre = model.test(sequence, v)

            likelihood_losses, mse_losses = criterion(mu_pre, sigma_pre, label)
            ae_losses = AE_loss(mu_pre, label, opt.ignore_zero)

            index += 1

            total_likelihood += torch.sum(likelihood_losses).detach().double()
            total_se += torch.sum(mse_losses).detach().double()
            total_ae += torch.sum(ae_losses).detach().double()
            total_label += torch.sum(label).detach().item()
            total_pred_num += len(likelihood_losses)

    se = torch.sqrt(total_se / total_pred_num) / (total_label / total_pred_num)
    ae = total_ae / total_label

    return total_likelihood / total_pred_num, se, ae


def train(model, optimizer, scheduler, opt, model_save_dir):
    """ Start training. """
    best_metrics = []
    best_nrmse = 10000

    index_names = ['Best Epoch', 'Log-Likelihood', 'NMSE', 'NMAE']

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        """ prepare dataloader """
        training_data, validation_data = prepare_dataloader(opt)

        start = time.time()
        train_likelihood, train_mse = train_epoch(model, training_data, optimizer, opt, epoch_i)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'MSE: {mse: 8.5f}'
              'elapse: {elapse:3.3f} min'
              .format(ll=train_likelihood, mse=train_mse, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_likelihood, valid_mse, valid_mae = eval_epoch(model, validation_data, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'RMSE: {RMSE: 8.5f}, '
              'NMAE: {accuracy: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_likelihood, RMSE=valid_mse, accuracy=valid_mae, elapse=(time.time() - start) / 60))

        scheduler.step()

        # Choose NRMSE as the metric to select the best model.
        if best_nrmse > valid_mse:
            best_nrmse = valid_mse
            best_metrics = [epoch, valid_likelihood, valid_mse, valid_mae]
            torch.save(
                {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_metrics': best_metrics
                },
                model_save_dir
            )

        print(index_names)
        print(best_metrics)

    return index_names, best_metrics


def evaluate(model, opt, model_save_dir):
    """Evaluate preptrained models"""
    index_names = ['Log-Likelihood', 'NMSE', 'NMAE']

    """ prepare dataloader """
    _, validation_data = prepare_dataloader(opt)

    """ load pretrained model """
    checkpoint = torch.load(model_save_dir)["model"]
    model.load_state_dict(checkpoint)

    start = time.time()
    valid_likelihood, valid_mse, valid_mae = eval_epoch(model, validation_data, opt)
    print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
            'RMSE: {RMSE: 8.5f}, '
            'NMAE: {accuracy: 8.5f}, '
            'elapse: {elapse:3.3f} min'
            .format(ll=valid_likelihood, RMSE=valid_mse, accuracy=valid_mae, elapse=(time.time() - start) / 60))

    best_metrics = [valid_likelihood, valid_mse, valid_mae]

    print(index_names)
    print(best_metrics)

    return index_names, best_metrics


def arg_parser():
    parser = argparse.ArgumentParser()

    # running mode
    parser.add_argument('-eval', action='store_true', default=False)

    # Path parameters
    parser.add_argument('-data_path', type=str, default='data/elect/')
    parser.add_argument('-dataset', type=str, default='elect')

    # Train parameters
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-inner_batch', type=int, default=8) # Equivalent batch size
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('-visualize_fre', type=int, default=2000)
    parser.add_argument('-pretrain', action='store_false', default=True)
    parser.add_argument('-hard_sample_mining', action='store_false', default=True)

    # Model parameters
    parser.add_argument('-model', type=str, default='Pyraformer')
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=128)
    parser.add_argument('-d_v', type=int, default=128)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layer', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)
    # Pyraformer parameters
    parser.add_argument('-window_size', type=str, default='[4, 4, 4]') # # The number of children of a parent node.
    parser.add_argument('-inner_size', type=int, default=3) # The number of ajacent nodes.
    parser.add_argument('-use_tvm', action='store_true', default=False) # Whether to use TVM.

    # Test parameter
    parser.add_argument('-predict_step', type=int, default=24)

    opt = parser.parse_args()
    return opt


def main():
    """ Main function. """
    opt = arg_parser()
    opt = get_dataset_parameters(opt)
    opt.window_size = eval(opt.window_size)
    print('[Info] parameters: {}'.format(opt))

    # default device is CUDA
    if torch.cuda.is_available():
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')

    """ prepare model """
    model = eval(opt.model).Model(opt)
    model.to(opt.device)

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    model_save_dir = 'models/SingleStep/{}/'.format(opt.dataset)
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_dir += 'best_model.pth'
    if opt.eval:
        index_name, best_metrics = evaluate(model, opt, model_save_dir)
    else:
        """ optimizer and scheduler """
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), opt.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
        index_name, best_metrics = train(model, optimizer, scheduler, opt, model_save_dir)

    print(index_name)
    print(best_metrics)


if __name__ == '__main__':
    main()

