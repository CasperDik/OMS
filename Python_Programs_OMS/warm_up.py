""""
OMS determine warm-up length
Version 1.1
Date 01/08/2021

This code:
code to determine the warmup period
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# define functions
def moving_average(data, window_size):
    """
    Simple moving average
    :param      window_size window size moving average
    :param      data pandas dataframe with the data
    :return     numpy array with moving average values
    """
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')


def welch_method(data, window_size):
    """
    Welch method for smoothing data
    :param      window_size window for welch method
    :param      data pandas dataframe with the data
    :return     numpy array with smoothed values
    """
    # make moving average dataset
    adapted_window_size = window_size * 2 + 1
    window = np.ones(int(adapted_window_size)) / float(adapted_window_size)
    df = pd.DataFrame(np.convolve(data, window, 'same'))

    # correct first observations
    for i in range(1, window_size+1, 1):
        df.loc[i-1] = data.loc[-2*(i-1):2*(i-1)].mean()

    # correct last observations
    for i in range(df.shape[0] - window_size, df.shape[0], 1):
        df.loc[i] = data.loc[i:df.shape[0]].mean()
    return df


def marginal_standard_error_rule(data):
    """
    Compute the Marginal Standard Error Rule (MSER), based on White (1997). See Robinson (2014) for more details.
    :param      data pandas dataframe with the data
    :return     list with MSER values
    """
    mser = list()
    m = data.shape[0]
    exclude = 5 # exclude last few points
    for d in range(0, m):
        sum_of_squares = sum((data.loc[d+1:] - data.loc[d+1:].mean()) ** 2)
        result = sum_of_squares / ((m - d) ** 2)
        mser.append(result)
    # add last points to avoid division by zero
    for d in range(m, exclude):
        mser.append(data.mean())
    return mser


# activate the code
if __name__ == '__main__':
    # import the data
    df = pd.read_excel("data/warmup_data_10runs.xlsx", converters={"identifier": float,
                                                                                 "throughput_time": float})

    # get moving average
    ma_window = 50
    df["throughput_time_moving_average"] = moving_average(data=df.loc[:, "throughput_time"], window_size=ma_window)

    # get welch method
    welch_window = 50
    df["throughput_welch_method"] = welch_method(data=df.loc[:, "throughput_time"], window_size=welch_window)

    # get the marginal standard error rule (MSER)
    df["marginal_standard_error_rule"] = marginal_standard_error_rule(data=df.loc[:, "throughput_time"])

    # import plot to visualize warmup
    fig, ax_1 = plt.subplots(figsize=(12, 10))

    # make time-phased plot
    line_1 = ax_1.plot("identifier",
                       "throughput_time",
                       data=df,
                       marker='',
                       color='olive',
                       linewidth=2,
                       label='Throughput Time'
                       )
    line_2 = ax_1.plot("identifier",
                       "throughput_time_moving_average",
                       data=df,
                       marker='',
                       color='red',
                       linewidth=2,
                       label=f'Moving Average with window size {ma_window}'
                       )
    line_3 = ax_1.plot("identifier",
                       "throughput_welch_method",
                       data=df,
                       marker='',
                       color='blue',
                       linewidth=2,
                       label=f'Welch with window size {welch_window}'
                       )

    # create a second x-axis for MSER
    ax_2 = ax_1.twinx()
    line_4 = ax_2.plot("identifier",
                       "marginal_standard_error_rule",
                       data=df,
                       marker='',
                       color='black',
                       linewidth=1,
                       label='MSER'
                       )

    # find min MSER row and indicate in plot
    # exclude last columns (problem specific) because of unreliable results
    min_row = df.iloc[:-5]
    # find row with min MSER
    min_row = min_row.iloc[min_row['marginal_standard_error_rule'].idxmin()]
    # annotate plot
    ax_2.annotate(f'min MSER: {round(min_row.loc["marginal_standard_error_rule"], 4)} at time {round(min_row.loc["identifier"], 0)}',
                  xy=(min_row.loc["identifier"], min_row.loc["marginal_standard_error_rule"]),
                  xytext=(min_row.loc["identifier"], min_row.loc["marginal_standard_error_rule"] * 2),
                  fontsize=12,
                  arrowprops=dict(facecolor='black', shrink=0.1)
                  )

    # make legend. Because of two x-axis's, we need to combine the individual lines and extract label
    lines = line_1 + line_2 + line_3 + line_4
    labels = [l.get_label() for l in lines]
    ax_1.legend(lines, labels, loc='best', shadow=True, fontsize='large', bbox_to_anchor=(0.75, -0.05))

    # make labels
    ax_1.set_title("Time line")
    ax_1.set_xlabel('Time')
    ax_1.set_ylabel('Throughput time')
    ax_2.set_ylabel('MSER')

    # and show
    plt.subplots_adjust(bottom=0.2) # resize the plot to fit the legent below the plot
    plt.show()
