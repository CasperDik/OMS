import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# experiment 1, effect of number of trucks
df = pd.read_excel("output_experiments_assignment3.xlsx")

experiment1 = df[(df["Lowerbound unloading speed for load type A"] == 13) & (df["Lowerbound unloading speed for load type C"] == 10) & (df["Truck interarrival interval"] == 900) & (df["Small truck value"] == 0)]
experiment1.sort_values(by=["Number of lift trucks available"], inplace=True)
experiment1.reset_index()

plt.plot(experiment1["Number of lift trucks available"], experiment1["Total costs per day"], label="Total costs per day")
plt.plot(experiment1["Number of lift trucks available"], experiment1["Costs regular hours lift truck drivers"], label="Costs regular hours lift truck drivers")
plt.plot(experiment1["Number of lift trucks available"], experiment1["Costs over time lift truck drivers"], label="Costs over time lift truck drivers")
plt.plot(experiment1["Number of lift trucks available"], experiment1["Costs lead time trucks"], label="Costs lead time trucks")
plt.plot(experiment1["Number of lift trucks available"], experiment1["Costs lift trucks"], label="Costs lift trucks")
plt.vlines(x=experiment1["Number of lift trucks available"][experiment1["Total costs per day"].idxmin()], ymin=0, ymax=experiment1["Total costs per day"].min(), linestyles="dashed")

plt.ylim(0, 15000)
plt.title("Experiment 1")
plt.xlabel("Number of lift trucks")
plt.ylabel("Costs per day")
plt.legend(bbox_to_anchor=(1.1, 0.7), loc="upper right")
plt.show()

