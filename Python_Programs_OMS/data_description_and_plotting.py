""""
OMS basic data description and matplotlib plot
Version 1.1
Date 01/08/2021

This code:
illustration of basic matplotlib plot
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# import the data
df = pd.read_csv("data/Simulation_Example_Warmup.csv")

# only select the throughput time for plotting
data = df.loc[:, "throughput_time"]

"""
General statistics 
"""
# print description of the dataset
print(data.describe())

"""
Histogram plot
"""

# import the plot (note: fig are is the figure object and ax are the axis object, matplotlib specific)
fig, ax = plt.subplots(figsize=(12, 8))

# make a histogram
ax.hist(data, bins=50, color="red")

# make labels
ax.set_title("Histogram")
ax.set_xlabel('Throughput Time')
ax.set_ylabel('Frequency')

# show the plot
plt.show()

"""
Boxplot
"""
# import new plot
fig_2, ax = plt.subplots(figsize=(12, 8))

# make a boxplot
ax.boxplot(data)

# make labels
ax.set_title("Boxplot")
ax.set_ylabel('Throughput Time')

# show the plot
plt.show()

"""
Time series
"""
# use identifier as time for a time series plot.
fig_3, ax = plt.subplots(figsize=(12, 8))
ax.plot("identifier",
        "throughput_time",
        data=df.iloc[:50,],     # Additionally, we only select the first 50 rows to make the plot smaller
        marker='',
        color='blue',
        linewidth=2
        )

# make labels
ax.set_title("Time series")
ax.set_xlabel('Time')
ax.set_ylabel('Throughput time')
# and show plot
plt.show()
