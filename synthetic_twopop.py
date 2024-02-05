import numpy as np
from scipy.stats import expon, norm
import os
from matplotlib import pyplot as plt
from cmdstanpy import CmdStanModel, CmdStanMCMC, from_csv
from stan_plot import plot_pair, plot_ppc
cwd = '/Users/julian/Documents/postdoc/braking_pop'
figdir = os.path.join(cwd, 'figs')

def generate_truth_dic(n_mu1, n_sig1, n_mu2, n_sig2, frac):
    truth_dic = {
    "n_mu1": n_mu1,
    "n_sig1": n_sig1,
    "n_mu2": n_mu2,
    "n_sig2": n_sig2,
    "frac": frac,
    }
    return truth_dic

def generate_data_from_truth(truth_dic, num_n=100, rng=None):
    if rng==None:
        rng = np.random.default_rng()

    num1 = int(truth_dic['frac'] * num_n)
    num2 = num_n - num1

    n1s = rng.normal(truth_dic['n_mu1'], truth_dic['n_sig1'], size=num1)
    n2s = rng.normal(truth_dic['n_mu2'], truth_dic['n_sig2'], size=num2)
    ninh = np.concatenate([n1s, n2s]).ravel()
    rng.shuffle(ninh)

    # "pulsar"-specific truths, magic numbers from atnf_fdots() and atnf_zeta(), i.e. broadly matches ATNF catalog
    zeta = rng.lognormal(7.3, 6.7, size=num_n)
    # zeta = rng.lognormal(1, 1, size=num_n)

    n_meas = rng.normal(ninh, zeta, size=num_n)

    zeta_unc = 0.1 * zeta # assume a 10% uncertainty on each "measurement" of zetea

    data_dic = {"N": num_n,
            "n_meas": n_meas,
            "zeta_meas": zeta,
            "zeta_unc": zeta_unc,
            }
    return data_dic

def get_samples_assuming_one_pop(data_dic, seed=None, new_samples=True, save=False):
    if seed==None:
        seed = np.random.randint(low=0, high=int(1e9))

    if new_samples:
        stanfile = os.path.join(cwd, 'zeta_unc.stan')
        model = CmdStanModel(stan_file=stanfile) # this compiles the stan model if it has changed
        # sampling!
        fit = model.sample(data=data_dic, iter_sampling=1000, seed=seed, show_console=False) 

        # can save samples
        if save:
            fit.save_csvfiles(os.path.join(cwd, 'twopop_wrong_model_draws'))

    # or load samples
    else:
        fit = from_csv(os.path.join(cwd, 'twopop_wrong_model_draws'))

    return fit

def get_samples_assuming_two_pop(data_dic, seed=None, new_samples=True, save=False):
    if seed==None:
        seed = np.random.randint(low=0, high=int(1e9))

    if new_samples:
        stanfile = os.path.join(cwd, 'twopop.stan')
        model = CmdStanModel(stan_file=stanfile) # this compiles the stan model if it has changed
        # sampling!
        fit = model.sample(data=data_dic, iter_sampling=1000, seed=seed, show_console=False) 

        # can save samples
        if save:
            fit.save_csvfiles(os.path.join(cwd, 'twopop_draws'))

    # or load samples
    else:
        fit = from_csv(os.path.join(cwd, 'twopop_draws'))

    return fit



def plot_ninh_assuming_one_pop(fit, truth_dic, num_n):
    ninh_post = fit.n_mu
    n_sig_post = fit.n_sig

    # this generates a posterior predictive check, i.e. folding together posterior on n_sig and n_mu 
    # i.e., after seeing all the data and fitting the model, these are our predictions for future samples of the inherent braking index
    n_per_posterior = 100
    post_samples = np.ravel([np.random.normal(p_mean, p_sig, size=n_per_posterior) for (p_mean, p_sig) in zip(ninh_post, n_sig_post)])

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.hist(post_samples, bins=50, histtype='step', density=True, label='Posterior')

    # plot injected distribution of inherent braking indices
    n_plot = np.linspace(2.5, 5.5, num=1000)
    gauss = lambda x, mu, sig: 1 / np.sqrt(2 * np.pi * sig**2) * np.exp(-(x - mu)**2 / sig**2)
    ninh1 =  gauss(n_plot, truth_dic['n_mu1'], truth_dic['n_sig1'])
    ninh2 =  gauss(n_plot, truth_dic['n_mu2'], truth_dic['n_sig2'])
    norm_truth = truth_dic['frac'] * ninh1 + (1 - truth_dic['frac']) * ninh2
    ax.plot(n_plot, norm_truth, label='Injected truth')
    ax.set_xlabel('Inherent braking index')
    ax.set_ylabel('Probability density')
    ax.legend(loc='upper right')
    ax.set_title(f'Run with {num_n:d} fake pulsars')
    ax.set_xlim(1, 8)

    # plt.savefig(os.path.join(figdir, f'ninh_ppc_{num_n:d}pulsars.png'), dpi=450)
    plt.show()

def plot_ninh_assuming_two_pop(fit, truth_dic, num_n):
    # this generates a posterior predictive check, i.e. folding together posterior on n_sig and n_mu 
    # i.e., after seeing all the data and fitting the model, these are our predictions for future samples of the inherent braking index
    n_per_posterior = 100
    post_samples = []
    for theta, mu1, mu2, sig1, sig2 in zip(fit.theta, fit.n_mu1, fit.n_mu2, fit.n_sig[:,0], fit.n_sig[:,1]):
        # print(theta)
        if theta > np.random.uniform(0, 1):
            samples = np.random.normal(mu1, sig1, size=n_per_posterior)
        else:
            samples = np.random.normal(mu2, sig2, size=n_per_posterior)
        post_samples.append(samples)
    post_samples = np.ravel(post_samples)

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.hist(post_samples, bins=50, histtype='step', density=True, label='Posterior')

    # plot injected distribution of inherent braking indices
    n_plot = np.linspace(2.5, 5.5, num=1000)
    gauss = lambda x, mu, sig: 1 / np.sqrt(2 * np.pi * sig**2) * np.exp(-(x - mu)**2 / sig**2)
    ninh1 =  gauss(n_plot, truth_dic['n_mu1'], truth_dic['n_sig1'])
    ninh2 =  gauss(n_plot, truth_dic['n_mu2'], truth_dic['n_sig2'])
    norm_truth = truth_dic['frac'] * ninh1 + (1 - truth_dic['frac']) * ninh2
    ax.plot(n_plot, norm_truth, label='Injected truth')
    ax.set_xlabel('Inherent braking index')
    ax.set_ylabel('Probability density')
    ax.legend(loc='upper right')
    ax.set_title(f'Run with {num_n:d} fake pulsars')
    ax.set_xlim(1, 8)

    plt.savefig(os.path.join(figdir, f'twopop_ninh_ppc_{num_n:d}pulsars.png'), dpi=450)
    plt.show()



if __name__=='__main__':
    truth_dic = generate_truth_dic(3, 0.2, 5, 0.2, 0.7)
    data_dic = generate_data_from_truth(truth_dic, num_n=200)

    # fit = get_samples_assuming_one_pop(data_dic)
    # plot_ninh_assuming_one_pop(fit, truth_dic, num_n=200)
    
    fit = get_samples_assuming_two_pop(data_dic, save=True, new_samples=True)
    # print(fit.summary())
    plot_ninh_assuming_two_pop(fit, truth_dic, num_n=200)