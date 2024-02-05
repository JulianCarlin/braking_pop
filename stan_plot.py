import numpy as np
from scipy.stats import expon, cauchy, gamma
import matplotlib.pyplot as plt
import os
import arviz as az

cwd = '/Users/julian/Documents/postdoc/braking_pop/'
figdir = os.path.join(cwd, 'figs')

def plot_pair(fit, var_names, latex_labels, truth_dic={}, priors={}, num_prior=int(1e5), figsize=(6, 4)):
    try:
        assert len(var_names) == len(latex_labels)
    except AssertionError:
        raise AssertionError("The number of variables to plot and the number of labels don't match")

    ndim = len(var_names)

    for_az = az.from_cmdstanpy(posterior=fit)
    summary = fit.summary(sig_figs=4)

    try:
        az_keys = for_az['posterior'].keys()
        for var in var_names:
            assert var in az_keys
    except AssertionError:
        raise AssertionError(f"The var_names provided don't exist in the fit object \
                var_names: {var_names} \
                az_keys: {az_keys}")

    label_map = dict((s, l) for (s, l) in zip(var_names, latex_labels))
    labeller = az.labels.MapLabeller(var_name_map=label_map)
    if truth_dic:
        truth_values = [k[1] for k in truth_dic.items()]
        reference_map = dict((s, l) for (s, l) in zip(var_names, truth_values))
        print('Check that the reference_map has the truth_dic variable names and the var_names lined up correctly:')
        print(truth_dic)
        print(reference_map)

    ax = az.plot_pair(for_az, figsize=figsize, kind='kde', marginals=True,
            var_names=var_names,
            labeller=labeller,
            kde_kwargs={'contour_kwargs':{'alpha':0}, 'hdi_probs':[0.1, 0.3, 0.5, 0.7, 0.9], 'fill_last':True})

    diag_ax = []
    for i in range(ndim):
        diag_ax.append(ax[i, i])
    for (i, dax), var_name, latex in zip(enumerate(diag_ax), var_names, latex_labels):
        if (ndim == 2) and (i == ndim -1):
            dax.axhline(summary['5%'][var_name], ls='dashed', color='grey', alpha=0.7)
            dax.axhline(summary['95%'][var_name], ls='dashed', color='grey', alpha=0.7)
            dax.set_ylabel(latex + r'$=%0.2f^{+%0.2f}_{-%0.2f}$'%(
                summary['50%'][var_name],
                summary['95%'][var_name] - summary['50%'][var_name],
                summary['50%'][var_name] - summary['5%'][var_name]),
                fontsize=11, rotation=270, labelpad=15)
            dax.yaxis.set_label_position('right')
        else:
            dax.axvline(summary['5%'][var_name], ls='dashed', color='grey', alpha=0.7)
            dax.axvline(summary['95%'][var_name], ls='dashed', color='grey', alpha=0.7)
            dax.set_title(latex + r'$=%0.2f^{+%0.2f}_{-%0.2f}$'%(
                summary['50%'][var_name],
                summary['95%'][var_name] - summary['50%'][var_name],
                summary['50%'][var_name] - summary['5%'][var_name]),
                fontsize=11)

        if truth_dic:
            truth_var = [v for (k, v) in truth_dic.items() if var_name in k][0]
            if (ndim == 2) and (i == ndim -1):
                dax.axhline(truth_var, ls='dashed', color='red', alpha=0.7)
            else:
                dax.axvline(truth_var, ls='dashed', color='red', alpha=0.7)

        if priors:
            prior = priors[var_name]
            if prior['type'] == 'gaussian':
                sampler = np.random.normal
            if prior['type'] == 'exponential':
                sampler = expon.rvs
            if prior['type'] == 'cauchy':
                sampler = cauchy.rvs
            if prior['type'] == 'gamma':
                sampler = gamma.rvs
            prior_samples = sampler(*prior['params'], size=num_prior)
            if (ndim == 2) and (i == ndim -1):
                lims = dax.get_ylim()
            else:
                lims = dax.get_xlim()
            num_in_range = np.sum((prior_samples > prior['lower_constraint']) & (prior_samples < lims[-1]))
            if num_in_range < (num_prior / 10):
                print(f'Only {num_in_range} prior samples to be plotted for {var_name}, something went wrong maybe')

            if (ndim == 2) and (i == ndim -1):
                dax.hist(prior_samples, bins=100, density=True, ec='none', fc='green', 
                        histtype='stepfilled', alpha=0.5, zorder=0, range=(prior['lower_constraint'], lims[-1]),
                        orientation='horizontal')
                dax.set_ylim(lims)
            else:
                dax.hist(prior_samples, bins=100, density=True, ec='none', fc='green', 
                        histtype='stepfilled', alpha=0.5, zorder=0, range=(prior['lower_constraint'], lims[-1]))
                dax.set_xlim(lims)
                
    offaxs = [ax[i, j] for i in range(ndim) for j in range(ndim) if j < i]
    indices = np.meshgrid(range(ndim), range(ndim))
    offax_tuples = [(indices[0][i, j], indices[1][i, j]) for i in range(ndim) for j in range(ndim) if j < i]
    if truth_dic:
        for offax, offax_ind in zip(offaxs, offax_tuples):
            offax.axvline(truth_values[offax_ind[0]], ls='dashed', color='red', alpha=0.7)
            offax.axhline(truth_values[offax_ind[1]], ls='dashed', color='red', alpha=0.7)

    return plt.gcf(), ax


def plot_ppc(fit, var_names, latex_labels, truth_dic, priors, num_hist=int(1e5), figsize=(5, 1.5)):
    try:
        assert len(var_names) / 2 == len(latex_labels)
    except AssertionError:
        raise AssertionError("The number of variables to plot and the number of labels don't match")

    ndim = int(len(var_names) / 2)

    for_az = az.from_cmdstanpy(posterior=fit)
    summary = fit.summary(sig_figs=4)

    try:
        az_keys = for_az['posterior'].keys()
        for var in var_names:
            assert var in az_keys
    except AssertionError:
        raise AssertionError(f"The var_names provided don't exist in the fit object \
                var_names: {var_names} \
                az_keys: {az_keys}")

    samples = fit.draws_xr(vars=var_names)

    label_map = dict((s, l) for (s, l) in zip(var_names, latex_labels))
    labeller = az.labels.MapLabeller(var_name_map=label_map)
    if truth_dic:
        truth_values = [k[1] for k in truth_dic.items()]
        reference_map = dict((s, l) for (s, l) in zip(var_names, truth_values))
        print('Check that the reference_map has the truth_dic variable names and the var_names lined up correctly:')
        print(truth_dic)
        print(reference_map)

    fig, axarr = plt.subplots(ncols=ndim, figsize=figsize)

    for (i, ax), label in zip(enumerate(axarr), latex_labels):
        var_mean_name = var_names[2 * i]
        var_sig_name = var_names[2 * i + 1]

        true_mean_key = 'true_' + var_mean_name.split('_')[0] + '_mu'
        true_sig_key = 'true_' + var_sig_name.split('_')[0] + '_sig'

        true_mean = truth_dic[true_mean_key]
        true_sig = truth_dic[true_sig_key]

        posterior_means = np.ravel(samples[var_mean_name].to_numpy())
        posterior_sigs = np.ravel(samples[var_sig_name].to_numpy())

        prior = priors[var_mean_name.split('_')[0]]
        if prior['mean']['type'] != 'gaussian':
            raise NotImplementedError(f'The mean for {var_mean_name} must be a Gaussian')
        r_plot = [prior['mean']['lower_constraint'], prior['mean']['params'][0] + 3 * prior['mean']['params'][-1]]

        n_per_posterior = int(num_hist / len(samples[var_mean_name]))
        posterior_samples = np.ravel([np.random.normal(p_mean, p_sig, size=n_per_posterior) for (p_mean, p_sig) in zip(posterior_means, posterior_sigs)])
        ax.hist(posterior_samples, density=True, bins=100, histtype='step', ec='blue', fc='none', range=r_plot, label='Posterior')

        true_samples = np.random.normal(true_mean, true_sig, size=num_hist)
        ax.hist(true_samples, density=True, bins=100, histtype='step', ec='red', fc='none', range=r_plot, zorder=0, label='True')

        mu_prior_samples = np.random.normal(*prior['mean']['params'], size=len(samples[var_mean_name]))
        mu_prior_samples = mu_prior_samples[mu_prior_samples > prior['mean']['lower_constraint']]

        if prior['sd']['type'] == 'gaussian':
            sampler = np.random.normal
        if prior['sd']['type'] == 'exponential':
            sampler = expon.rvs
        if prior['sd']['type'] == 'cauchy':
            sampler = cauchy.rvs
        if prior['sd']['type'] == 'gamma':
            sampler = gamma.rvs
        sd_prior_samples = sampler(*prior['sd']['params'], size=len(samples[var_mean_name]))
        sd_prior_samples = sd_prior_samples[sd_prior_samples > prior['sd']['lower_constraint']]
        num_prior_samples = min([len(sd_prior_samples), len(mu_prior_samples)])

        prior_samples = np.ravel([np.random.normal(mean, sig, size=int(num_hist / num_prior_samples)) for (mean, sig) in zip(mu_prior_samples, sd_prior_samples)])
        ax.hist(prior_samples, density=True, bins=100, histtype='step', ec='green', fc='none', range=r_plot, zorder=-1,   label='Prior')

        ax.set_xlim(r_plot)
        ax.set_xlabel(label)
        ax.set_yticks([])
    axarr[0].set_ylabel('PDF')
    axarr[0].legend(loc='upper left', prop={'size': 8})

    return fig, axarr
