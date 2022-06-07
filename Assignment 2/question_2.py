import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# df = pd.read_excel("Assignment 2/20210928_ProcessingTimeStation2.xlsx")
df = pd.read_excel("20210928_ProcessingTimeStation2.xlsx")

bins = round(np.sqrt(len(df))) + 20
plt.hist(df, bins=bins)
plt.show()

print(df.describe())

# count nan values
print(df.isnull().sum().sum())

# amount of non-positive numbers
print(len(df[df["throughput_time"] <= 0]))

# drop non-positive
df = df[df["throughput_time"] > 0]

df.to_excel("ProcessingTimeStation2_Cleaned.xlsx")

