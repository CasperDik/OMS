""""
OMS t-test
Version: 1.1
Date 01/08/2021

This code:
compare two groups using a t-test
"""
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

# import data
df = pd.read_excel("Q3_noCRN.xlsx", converters={"run": int,
                                                   "throughput_time": float,
                                                   "dispatching_rule": str})

# get data for SPT
data_SPT = df[df["dispatching_rule"] == "SPT"]
# get data for FCFS
data_FCFS = df[df["dispatching_rule"] == "FCFS"]

# do t-test and print result
dependent = False
if dependent:
    result = st.ttest_rel(data_SPT.loc[:, "throughput_time"], data_FCFS.loc[:, "throughput_time"])
else:
    result = st.ttest_ind(data_SPT.loc[:, "throughput_time"], data_FCFS.loc[:, "throughput_time"])

# print result
print(f"t-statistic: {round(result[0], 3)} and p-value: {round(result[1], 3)}")

# visualize difference between groups
fig, ax = plt.subplots(figsize=(12, 8))

# make a boxplot
ax.boxplot([data_SPT.loc[:, "throughput_time"], data_FCFS.loc[:, "throughput_time"]], positions = [1, 2], widths = 0.6)

# make labels
ax.set_title("Boxplot")
ax.set_xticklabels(['SPT', 'FCFS'])
ax.set_ylabel('Throughput Time')

# show the plot
plt.show()