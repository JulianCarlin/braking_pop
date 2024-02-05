import numpy as np
from scipy.stats import expon, norm
import os
from matplotlib import pyplot as plt
from cmdstanpy import CmdStanModel, CmdStanMCMC, from_csv
from stan_plot import plot_pair, plot_ppc
cwd = '/Users/julian/Documents/postdoc/braking_pop'
figdir = os.path.join(cwd, 'figs')

atnf_file = os.path.join(cwd, 'all_atnf.txt')

# read in atnf pulsars, to get idea of spread of nu, nudot, for synthetic data below
def atnf_fdots(plot=False):
    with open(atnf_file, 'r') as open_file:
        names = ['id', 'PSRJ', 'ra', 'dec', 'p0', 'p1', 'binary_model', 'type', 'ngltch']
        dtypes = ['i4', 'U12', 'U10', 'U11', 'f8', 'f8', 'U10', 'U20', 'i4']
        data = np.genfromtxt(open_file, delimiter=';', skip_header=2, missing_values='*', names=names, dtype=dtypes)

    p1 = data['p1']
    p0 = data['p0']

    # don't include pulsars spinning up or without measured spin-down
    p1_mask = np.logical_or(np.isnan(p1), p1 <= 0)
    # with too-small spin-down (not rotationally-powered)
    p1_mask = np.logical_or(p1_mask, p1 < 1e-18)
    # or without measured spin
    mask = np.logical_or(p1_mask, np.isnan(p0))

    p1 = p1[~mask]
    p0 = p0[~mask]
    
    f = 1 / p0
    fdot = p1 / p0**2

    f_mean = np.mean(np.log(f))
    f_sig = np.std(np.log(f))
    fdot_mean = np.mean(np.log(fdot))
    fdot_sig = np.std(np.log(fdot))
    print(f'2sigma range for f: ({np.exp(f_mean - 2 * f_sig):0.3e}, {np.exp(f_mean + 2 * f_sig):0.3e})')
    print(f'2sigma range for fdot: ({np.exp(fdot_mean - 2 * fdot_sig):0.3e}, {np.exp(fdot_mean + 2 * fdot_sig):0.3e})')

    print(f'LN mu and sig for freq: {f_mean:0.3e}, {f_sig:0.3e}')
    print(f'LN mu and sig for fdot: {fdot_mean:0.3e}, {fdot_sig:0.3e}')

    if plot:
        fig, ax = plt.subplots(ncols=2, figsize=(6,3))
        ax[0].hist(f, bins=np.logspace(np.log10(min(f)), np.log10(max(f)), num=100), density=True)
        fsamples = np.random.lognormal(f_mean, f_sig, size=int(1e5))
        ax[0].hist(fsamples, histtype='step', bins=np.logspace(np.log10(min(f)), np.log10(max(f)), 100), density=True)
        ax[0].set_xlabel(r'$f\,$(Hz)')
        ax[0].set_xscale('log')
        ax[1].hist(fdot, bins=np.logspace(np.log10(min(fdot)), np.log10(max(fdot)), num=100))
        ax[1].set_xlabel(r'$-\dot{f}\,$(Hz/s)')
        ax[1].set_xscale('log')
        plt.show()

# Estimate parameters that LN distribution of zeta should have given ATNF catalog
def atnf_zeta(num_n=int(1e6)):
    rng = np.random.default_rng()
    nu_k = rng.lognormal(5.101e-01, 9.609e-01, size=num_n) # nu between 0.1, 10 Hz
    nudot_k = rng.lognormal(-3.284e+01, 2.901e+00, size=num_n) # nudot between 1e-17, 1e-12 Hz/s
    tobs_k = 365.24 * 86400 * rng.uniform(3, 6, size=num_n) # tobs between 3 to 6 years
    s_k = nu_k**2 / (nudot_k**4 * tobs_k)
    gamma_k = 1e-6
    sigmasq_k = 10**rng.uniform(-60, -50, size=num_n)
    zeta = np.sqrt(sigmasq_k / gamma_k**2 * s_k)

    print(f'Mean of LN for zeta: {np.mean(np.log(zeta))}')
    print(f'Std. dev. of LN for zeta: {np.std(np.log(zeta))}')

def generate_truth_dic(n_mu, n_sig):
    try:
        _ = iter(n_mu)
    except TypeError:
        truth_dic = {
        "n_mu": n_mu,
        "n_sig": n_sig,
    }
    else:
        NotImplementedError("Double-peaked population distribution not yet implemented")
    return truth_dic

def generate_data_from_truth(truth_dic, num_n=100, rng=None):
    if rng==None:
        rng = np.random.default_rng()
    
    ninh = rng.normal(truth_dic['n_mu'], truth_dic['n_sig'], size=num_n)

    # "pulsar"-specific truths, magic numbers from atnf_fdots() and atnf_zeta(), i.e. broadly matches ATNF catalog
    zeta = rng.lognormal(7.3, 6.7, size=num_n)

    n_meas = rng.normal(ninh, zeta, size=num_n)

    zeta_unc = 0.1 * zeta # assume a 10% uncertainty on each "measurement" of zetea

    data_dic = {"N": num_n,
            "n_meas": n_meas,
            "zeta_meas": zeta,
            "zeta_unc": zeta_unc,
            }
    return data_dic


def get_samples(data_dic, seed=None, new_samples=True, save=False):
    if seed==None:
        seed = np.random.randint(low=0, high=int(1e9))

    if new_samples:
        stanfile = os.path.join(cwd, 'zeta_unc.stan')
        model = CmdStanModel(stan_file=stanfile) # this compiles the stan model if it has changed
        # sampling!
        fit = model.sample(data=data_dic, iter_sampling=1000, seed=seed, show_console=False) 

        # can save samples
        if save:
            fit.save_csvfiles(os.path.join(cwd, 'zeta_unc_draws'))

    # or load samples
    else:
        fit = from_csv(os.path.join(cwd, 'zeta_unc_draws'))

    return fit


def plot_ninh_toy(fit, truth_dic, num_n):
    ninh_post = fit.n_mu
    n_sig_post = fit.n_sig

    # this generates a posterior predictive check, i.e. folding together posterior on n_sig and n_mu 
    # i.e., after seeing all the data and fitting the model, these are our predictions for future samples of the inherent braking index
    n_per_posterior = 100
    post_samples = np.ravel([np.random.normal(p_mean, p_sig, size=n_per_posterior) for (p_mean, p_sig) in zip(ninh_post, n_sig_post)])

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.hist(post_samples, bins=50, histtype='step', density=True, label='Posterior')

    # plot injected distribution of inherent braking indices
    n_plot = np.linspace(2.5, 3.5, num=1000)
    norm_truth = 1 / np.sqrt(2 * np.pi * truth_dic['n_sig']**2) * np.exp(-(n_plot - truth_dic['n_mu'])**2 / truth_dic['n_sig']**2)
    ax.plot(n_plot, norm_truth, label='Injected truth')
    ax.set_xlabel('Inherent braking index')
    ax.set_ylabel('Probability density')
    ax.legend(loc='upper right')
    ax.set_title(f'Run with {num_n:d} fake pulsars')

    plt.savefig(os.path.join(figdir, f'ninh_ppc_{num_n:d}pulsars.png'), dpi=450)
    plt.show()

if __name__=='__main__':
    # atnf_fdots(plot=False)
    # atnf_zeta()

    # generate some synthetic data to test things
    truth_dic = generate_truth_dic(n_mu=3, n_sig=0.2)
    data_dic = generate_data_from_truth(truth_dic, num_n=100)

    # do the fit, or read in the samples
    fit = get_samples(data_dic, new_samples=True, save=False)
    print(fit.summary())

    # plot the result
    plot_ninh_toy(fit, truth_dic, data_dic['N'])

