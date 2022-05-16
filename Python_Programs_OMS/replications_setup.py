""""
OMS determine number of runs
Version 1.1
Date 29/09/2021

This code:
code to determine the number of runs
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st


# define functions
def replication_confidence_interval(data, run_number):
    """
    create two-tailed confidence intervals at 95% confidence level
    :param data:            data with run numbers
    :param run_number:      imminent run number
    :return:                run mean, lower and upper confidence interval and deviation
    """
    run_mean = data.mean()
    std_dev = data.std()
    """
    Compute interval (0.05 / 2) because we want a two-tailed confidence at 0.05 probability. Furthermore, we use the percentage
    point function (i.e. the inverse of the cumulative density function) get the t-value at 97.5%
    (explaining  1 - (0.05 / 2)). df refers to degrees of freedom which equals the length of the database (minus one).
    """
    confidence_int = st.t.ppf(1 - (0.05 / 2), df=data.shape[0] - 1) * (std_dev / np.sqrt(run_number))

    # lower_confidence
    lower_confidence = run_mean - confidence_int
    # upper_confidence
    upper_confidence = run_mean + confidence_int
    # deviation
    deviation = upper_confidence/run_mean - 1

    return run_mean, lower_confidence, upper_confidence, deviation



# activate the code
if __name__ == '__main__':
    # import the data
    df = pd.read_csv("data/output_part2_meanthroughput.csv", delimiter=";")
    data = df.loc[:, "mean_throughput_time"]

    # make confidence intervals by looping over each replication
    df_statistics = None
    for run_number in range(1, df.shape[0] + 1):
        # get confidence intervals
        running_mean, lower_confidence, upper_confidence, deviation = replication_confidence_interval(
            run_number=run_number,
            data=data.iloc[:run_number]
        )
        # put confidence intervals in pandas dataframe
        add_df = pd.DataFrame([running_mean, lower_confidence, upper_confidence, deviation]).transpose()
        add_df.columns = ['running_mean', 'lower_confidence', 'upper_confidence', 'deviation']
        # add output statistical output to database
        if df_statistics is None:
            df_statistics = add_df
        else:
            df_statistics = pd.concat([df_statistics, add_df], ignore_index=True)

    # add confidence intervals to the existing dataframe
    data = pd.concat([df.loc[:, "run"], data, df_statistics], axis=1)

    # print table results
    print("chi-square results")
    print("-*" * 75)
    print(data.to_string(index=False))
    print("-*" * 75)

    # plot the data
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot("run",
            "mean_throughput_time",
            data=data,
            marker='',
            color='red',
            linewidth=1,
            label=f'mean throughput time'
            )
    ax.plot("run",
            "lower_confidence",
            data=data,
            marker='',
            color='blue',
            linewidth=2,
            linestyle=':',
            label=f'lower confidence'
            )
    ax.plot("run",
            "upper_confidence",
            data=data,
            marker='',
            color='blue',
            linewidth=2,
            linestyle=':',
            label=f'upper confidence'
            )
    ax.plot("run",
            "running_mean",
            data=data,
            marker='',
            color='green',
            linewidth=2,
            label=f'running mean'
            )
    # make labels
    ax.set_title("Confidence intervals")
    ax.set_xlabel('Run')
    ax.set_ylabel('Mean Throughput time')
    ax.legend(loc='best', shadow=True, fontsize='x-large')
    # and show plot
    plt.show()
