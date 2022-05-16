import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Python_Programs_OMS/data/output_v2.csv", delimiter=";")

plt.ylabel("Count")
plt.xlabel("TPT")
plt.hist(df["Throughput Time"], bins=1000)
plt.show()
