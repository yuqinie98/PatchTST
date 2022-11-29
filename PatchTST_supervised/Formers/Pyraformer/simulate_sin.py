import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM


def generate_sin(x, T, A):
    """Generate a mixed sinusoidal sequence"""
    y = np.zeros(len(x))
    for i in range(len(T)):
        y += A[i] * np.sin(2 * np.pi / T[i] * x)

    return y


def gen_covariates(x, index):
    """Generate covariates"""
    covariates = np.zeros((x.shape[0], 4))
    covariates[:, 0] = (x // 24) % 7
    covariates[:, 1] = x % 24
    covariates[:, 2] = (x // (24 * 30)) % 12
    covariates[:, 0] = covariates[:, 0] / 6
    covariates[:, 1] = covariates[:, 1] / 23
    covariates[:, 2] = covariates[:, 2] / 11

    covariates[:, -1] = np.zeros(x.shape[0]) + index
    return covariates


def fractional_brownian_noise(length, hurst, step):
    """Genereate fractional brownian noise"""
    f = FBM(length, hurst, step)
    noise = f.fbm()
    return noise


def synthesis_data():
    """synthesis a mixed sinusoidal dataset"""
    T = [24, 168, 720]
    seq_num = 60
    seq_len = T[-1] * 20
    data = []
    covariates = []
    for i in range(seq_num):
        start = int(np.random.uniform(0, T[-1]))
        x = start + np.arange(seq_len)
        A = np.random.uniform(5, 10, 3)
        y = generate_sin(x, T, A)
        data.append(y)
        covariates.append(gen_covariates(x, i))
        # plt.plot(x[:T[-1]], y[:T[-1]])
        # plt.show()

    data = np.array(data)
    mean, cov = polynomial_decay_cov(seq_len)
    noise = multivariate_normal(mean, cov, seq_num)
    data = data + noise
    covariates = np.array(covariates)
    data = np.concatenate([data[:, :, None], covariates], axis=2)
    np.save('data/synthetic.npy', data)


def covariance(data):
    """compute the covariance of the data"""
    data_mean = data.mean(0)
    data = data - data_mean
    length = data.shape[1]
    data_covariance = np.zeros((length, length))

    for i in range(length):
        for j in range(length):
            data_covariance[i, j] = (data[:, i] * data[:, j]).mean()

    return data_covariance


def test_fbm():
    """Plot the covariance of the generated fractional brownian noise"""
    f = FBM(300, 0.3, 1)
    fbm_data = []
    for i in range(100):
        sample = f.fbm()
        fbm_data.append(sample[1:])
    fbm_data = np.array(fbm_data)
    cov = covariance(fbm_data)
    plt.imshow(cov)
    plt.savefig('fbm_cov.jpg')


def polynomial_decay_cov(length):
    """Define the function of covariance decay with distance"""
    mean = np.zeros(length)

    x_axis = np.arange(length)
    distance = x_axis[:, None] - x_axis[None, :]
    distance = np.abs(distance)
    cov = 1 / (distance + 1)
    return mean, cov


def multivariate_normal(mean, cov, seq_num):
    """Generate multivariate normal distribution"""
    noise = np.random.multivariate_normal(mean, cov, (seq_num,), 'raise')
    return noise


if __name__ == '__main__':
    synthesis_data()

