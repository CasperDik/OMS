""""
OMS distribution fitting
Version: 2.1
Date 01/05/2021

This code:
fit a theoretical distribution on empirical data
"""
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import math


# define functions
def fit_distribution(distribution, x, y, data):
    """
    function that fits a theoretical distribution over empirical data using log likelihood
    :param distribution:    a scipy.stats distribution that needs fitting
    :param x:               x bin values
    :param y:               y frequency
    :return                 sse, distribution_parameter
    """
    # define return values
    sse = None
    dist_params = None
    # try to fit the distribution
    try:
        # fit dist to data (ignore warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            # fit distribution
            dist_params = distribution.fit(data)

            # separate parts of parameters
            arguments = dist_params[:-2]
            shape = dist_params[-2]
            scale = dist_params[-1]

            # compute pdf and sum of squared error
            pdf = distribution.pdf(x, loc=shape, scale=scale, *arguments)
            sse = np.sum(np.power(y - pdf, 2.0))
    except Exception:
        print(f"failed to fit the distribution {distribution}")
    return sse, dist_params


def get_data_distribution(dist_params, size, min, max):
    """
    function that makes an array of x values and obtains the relevant y values from a given distribution
    :param dist_params: list with parameters from the distribution
    :param size:                    number of data points to make an array from
    :param min:                     min x
    :param max:                     max x
    :return:                        array of x and accompanying y values
    """
    # split distribution parameters
    arguments = dist_params[:-2]
    shape = dist_params[-2]
    scale = dist_params[-1]
    # construct the pdf and return values
    x = np.linspace(min, max, size)
    y = distribution.pdf(x, loc=shape, scale=scale, *arguments)
    return x, y


def compute_goodness_of_fit(distribution, dist_params, data, bins):
    """
    function that computes the goodness of fit given an distribution and the empirical data
    :param distribution:                scipy.stats distribution object
    :param dist_params:     list of distribution params
    :param data:                        empirical data
    :return:                            various statistics
    """
    # do the Kolmogorov-Smirnov test to get a p-value (located at index 1, hence [1])
    p = st.kstest(data, distribution.name, args=dist_params)[1]
    p_value_kv = np.around(p, 5)

    # get expected frequency for each bin (based on cumulative density function)
    expected_frequency = []
    for i, bin in enumerate(bins[0:-1]):
        cdf_fitted_right = distribution.cdf(bins[i + 1],
                                            *dist_params[:-2],
                                            loc=dist_params[-2],
                                            scale=dist_params[-1])
        cdf_fitted_left = distribution.cdf(bins[i],
                                           *dist_params[:-2],
                                           loc=dist_params[-2],
                                           scale=dist_params[-1])
        expected_cdf_area = cdf_fitted_right - cdf_fitted_left
        expected_frequency.append(expected_cdf_area)

    # control if sufficient expected frequencies in each bin
    min_bin_size = 2
    for i, bin in enumerate(expected_frequency):
        expected_in_bin = bin * data.shape[0]
        if expected_in_bin < min_bin_size:
            raise Exception(f"less than {min_bin_size} expected observations in bin number: {i}")

    # put expected frequency in numpy array
    expected_frequency = np.array(expected_frequency)
    observed_frequency, x = np.histogram(data, bins=bins, density=False)
    observed_frequency = np.array(observed_frequency)

    # calculate chi-square and p-value
    expected_frequency = expected_frequency * (observed_frequency.sum() / expected_frequency.sum())
    chi_square = sum(((observed_frequency - expected_frequency) ** 2) / expected_frequency)
    p_value_chi_sq = st.chisquare(f_obs=observed_frequency, f_exp=expected_frequency)[1]
    return p_value_kv, chi_square, p_value_chi_sq


def get_distribution_paramater_names(distribution):
    shape_names = []
    if distribution.shapes:
        shape_names = [name.strip() for name in distribution.shapes.split(',')]
    return shape_names + ['loc', 'scale']


# activate the code
if __name__ == '__main__':
    # load dataset
    df = pd.read_excel("distribution_fitting.xlsx", converters={"throughput_time": float})  # version for .xlsx
    data = df.loc[:, "throughput_time"]

    # describe data
    print("-*" * 75)
    print("description dataset")
    print(data.describe())
    print("-*" * 75, "\n")

    # define distribution(s)
    """
    below a list with predefined distribution that are can be fit over the data. There are many more distributions.
    see: https://docs.scipy.org/doc/scipy/reference/stats.html for complete overview
    """
    distributions = [
        st.uniform,
        st.expon,
        st.gamma,
        st.invgamma,
        st.lognorm,
        st.powerlognorm,
        st.logistic,
        st.norm
    ]

    # specify bins
    """
    here you can define the bins (i.e. a intervals of a class). Note that you have to specify the intervals for each 
    bin. So if you want two classes with where the first bin has the intervals 1 and 2 and the second bin the intervals 
    2 and 3, then you have make a list that looks like: [1, 2, 3]. you can either choose to make equally distributed 
    bins (e.g. intervals are [1, 2, 3]) or specify the bin distribution manually (e.g. with the intervals [1, 1.5, 3])
    """
    bins = None
    equally_distributed_bins = True
    # find min (max) data and round down (up) to nearest integer
    min_value = int(data.min())
    max_value = int(math.ceil(data.max()))
    if equally_distributed_bins:
        number_of_bins = 10
        bins = np.linspace(min_value, max_value, number_of_bins + 1).tolist()
    else:
        bins = [min_value, 6, 7, 8, 9, 10, 11, max_value]

    # make histogram of of empirical data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0  # compute bin average (because np.histogram returns intervals)

    # control if bins are legal
    for bin_number, nr_observations in np.ndenumerate(y):
        # control if bin is empty when it has a NaN (not a number) value
        if np.isnan(nr_observations):
            raise Exception(f"bin {bin_number} contains no observations")
        # control if bin is empty with eight digits accuracy (because y contains density value instead of counts)
        elif int(nr_observations * (10 ** 8) + 0.5) / (10 ** 8) == 0:
            raise Exception(f"bin {bin_number} contains no observations")

    # set variables
    best_fitting_distribution = None
    least_sse = np.inf
    df_output_distributions = None
    df_results = None

    # loop to fit each distributions
    for i, distribution in enumerate(distributions):
        # fit the distribution
        sse, distribution_parameters = fit_distribution(distribution=distribution,
                                                        x=x,
                                                        y=y,
                                                        data=data)
        # check if distribution fits better than alternative
        if sse < least_sse:
            best_fitting_distribution = distribution.name
            least_sse = sse

        # run the distribution (probability density function) using existing params
        x_pdf, y_pdf = get_data_distribution(dist_params=distribution_parameters,
                                             size=10000,
                                             min=data.min() + 0.01,
                                             max=data.max() - 0.01)

        df = pd.DataFrame([x_pdf, y_pdf]).transpose()
        df.columns = [f"x_{distribution.name}", f"y_{distribution.name}"]

        # add output pdf to database (or make a new database if the database does not yet exist)
        if df_output_distributions is None:
            df_output_distributions = df
        else:
            df_output_distributions = pd.concat([df_output_distributions, df], axis=1)

        # perform statistical goodness of fit test
        p_value_kv, chi_square, p_value_chi_sq = compute_goodness_of_fit(distribution=distribution,
                                                                         dist_params=distribution_parameters,
                                                                         data=data,
                                                                         bins=bins)
        # get distribution parameters
        distribution_parameter_names = get_distribution_paramater_names(distribution=distribution)
        distribution_parameter_dict = {}
        for i, name in enumerate(distribution_parameter_names):
            distribution_parameter_dict[name] = int(distribution_parameters[i] * (10 ** 4) + 0.5) / (10 ** 4)

        df = pd.DataFrame([
            distribution.name,
            p_value_kv,
            chi_square,
            p_value_chi_sq,
            distribution.mean(loc=distribution_parameters[-2],
                              scale=distribution_parameters[-1],
                              *distribution_parameters[:-2]
                              ),
            distribution.var(loc=distribution_parameters[-2],
                             scale=distribution_parameters[-1],
                             *distribution_parameters[:-2]
                             ),
            distribution_parameter_dict
        ]).transpose()
        df.columns = [
            "| distribution |",
            "| p-value (KS) |",
            "| chi-square |",
            "| p-value (chi-sq) |",
            "| mean |",
            "| variance |",
            "| parameters |"
        ]
        # add results to database
        if df_results is None:
            df_results = df
        else:
            df_results = pd.concat([df_results, df], ignore_index=True)

    # print info
    print(
        f"best fitting distribution is {best_fitting_distribution} "
        f"with a sum of squared error of {least_sse}"
    )
    print("statistical results")
    """
    let Pe be the empirical distribution and Pf be the fitted distribution, then the 
    hypothesis of the chi-square goodness of fit test is
    H0: pe == pf
    H1: pe != pf
    so we reject H0 if the p-value is below a said significance level
    """
    print("-*" * 75)
    print(df_results.sort_values(by=['| chi-square |']).iloc[:, 0:4].to_string(index=False))
    print("-*" * 75, "\n")
    print("distribution parameters")
    print("-*" * 75)
    print(df_results.sort_values(by=['| chi-square |']).iloc[:, [0, 4, 5, 6]].to_string(index=False))
    print("-*" * 75, "\n")

    # choose to plot all (False) or the best distribution (True)
    show_best_dist = False

    # plot data
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(data, bins=bins, density=True, color="lightblue", label=f'empirical')
    # make colours
    color_map = plt.cm.get_cmap('hsv', len(distributions))
    if show_best_dist:
        # add only the best fitting distribution
        ax.plot(f'x_{best_fitting_distribution}',
                f'y_{best_fitting_distribution}',
                data=df_output_distributions,
                marker='',
                color='red',
                linewidth=2,
                label=f'{best_fitting_distribution}'
                )
    else:
        # loop to add all distributions to plot
        for i, distribution in enumerate(distributions):
            ax.plot(f'x_{distribution.name}',
                    f'y_{distribution.name}',
                    data=df_output_distributions,
                    marker='',
                    color=color_map(i),
                    linewidth=2,
                    label=f'{distribution.name}'
                    )
    legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
    ax.set_title("Distribution Fitting")
    ax.set_xlabel('Throughput Time')
    ax.set_ylabel('Frequency / Probability Density')
    plt.show()
