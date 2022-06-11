import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def experiment_1():
    # experiment 1, effect of number of trucks
    df = pd.read_excel("output_experiments_assignment3.xlsx")

    experiment1 = df[(df["Lowerbound unloading speed for load type A"] == 13) & (df["Lowerbound unloading speed for load type C"] == 10) & (df["Truck interarrival interval"] == 900) & (df["Checks model validity (trucks served in time, figure should be = 0)"] == 0)]
    experiment1.sort_values(by=["Number of lift trucks available"], inplace=True)
    experiment1.reset_index()

    experiment1[["Costs lift trucks", "Costs lead time trucks", "Costs regular hours lift truck drivers",
                 "Costs over time lift truck drivers", "Number of lift trucks available"]].plot(kind="bar",
                                                                                                stacked=True,
                                                                                                x="Number of lift trucks available")
    plt.ylim(0,15000)
    plt.ylabel("Costs per day")
    plt.legend(bbox_to_anchor=(1, 1.1), loc="upper right")
    plt.show()


def experiment_2():
    df = pd.read_excel("output_experiments_assignment3.xlsx")

    experiment1 = df[(df["Lowerbound unloading speed for load type A"] == 10) & (
                df["Lowerbound unloading speed for load type C"] == 13) & (df["Truck interarrival interval"] == 900) & (df["Checks model validity (trucks served in time, figure should be = 0)"] == 0)]

    experiment1.sort_values(by=["Number of lift trucks available"], inplace=True)
    experiment1.reset_index()

    experiment1[["Costs lift trucks", "Costs lead time trucks", "Costs regular hours lift truck drivers",
                 "Costs over time lift truck drivers", "Number of lift trucks available"]].plot(kind="bar",
                                                                                                stacked=True,
                                                                                                x="Number of lift trucks available")

    plt.ylim(0, 15000)
    plt.ylabel("Costs per day")
    plt.legend(bbox_to_anchor=(1, 1.1), loc="upper right")
    plt.show()


def experiment_3():
    df = pd.read_excel("output_experiments_assignment3.xlsx")

    experiment1 = df[(df["Lowerbound unloading speed for load type A"] == 10) & (
            df["Lowerbound unloading speed for load type C"] == 13) & (df["Truck interarrival interval"] == 750) & (df["Checks model validity (trucks served in time, figure should be = 0)"] == 0)]
    experiment1.sort_values(by=["Number of lift trucks available"], inplace=True)
    experiment1.reset_index()

    experiment2 = df[(df["Lowerbound unloading speed for load type A"] == 13) & (
            df["Lowerbound unloading speed for load type C"] == 10) & (df["Truck interarrival interval"] == 750) & (df["Checks model validity (trucks served in time, figure should be = 0)"] == 0)]

    experiment2.sort_values(by=["Number of lift trucks available"], inplace=True)
    experiment2.reset_index()

    ax = experiment1.plot(kind="bar", y=["Costs lift trucks", "Costs lead time trucks", "Costs regular hours lift truck drivers",
                 "Costs over time lift truck drivers"], stacked=True, x="Number of lift trucks available", width=0.3, position=0, legend=False)

    experiment2.plot(kind="bar", y=["Costs lift trucks", "Costs lead time trucks", "Costs regular hours lift truck drivers",
                 "Costs over time lift truck drivers"], stacked=True, x="Number of lift trucks available", ax=ax, width=0.3, position=1, legend=False, alpha=0.6)

    plt.title("Switching vs Not Switching(Transparent)")
    plt.ylabel("Costs per day")
    plt.show()

    print(experiment1["Total costs per day"].min())
    print(experiment2["Total costs per day"].min())


def experiment_4():
    # fix the number of lift truck to 11, A-C no changed
    df = pd.read_excel("output_2LT.xlsx")
    df1 = pd.read_excel("output_experiments_assignment3.xlsx")

    base = df1[(df1["Lowerbound unloading speed for load type A"] == 13) & (
            df1["Lowerbound unloading speed for load type C"] == 10) & (df1["Truck interarrival interval"] == 900) & (
                             df1["Number of lift trucks available"] == 11) & (df1["Checks model validity (trucks served in time, figure should be = 0)"] == 0)]
    experiment = df[(df["Lowerbound unloading speed for load type A"] == 13) & (
            df["Lowerbound unloading speed for load type C"] == 10) & (df["Truck interarrival interval"] == 900) & (
                             df["Number of lift trucks available"] == 11) & (df["Checks model validity (trucks served in time, figure should be = 0)"] == 0)]

    print(experiment["Total costs per day"])
    print(base["Total costs per day"])

    # change only lift truck number
    experiment1 = df[(df["Lowerbound unloading speed for load type A"] == 13) & (
            df["Lowerbound unloading speed for load type C"] == 10) & (df["Truck interarrival interval"] == 900) & (df["Checks model validity (trucks served in time, figure should be = 0)"] == 0)]
    base1 = df1[(df1["Lowerbound unloading speed for load type A"] == 13) & (
            df1["Lowerbound unloading speed for load type C"] == 10) & (df1["Truck interarrival interval"] == 900) & (df1["Checks model validity (trucks served in time, figure should be = 0)"] == 0)]

    ax = experiment1.plot(kind="bar", y=["Costs lift trucks", "Costs lead time trucks", "Costs regular hours lift truck drivers",
                 "Costs over time lift truck drivers"], stacked=True, x="Number of lift trucks available", width=0.3, position=0)

    base1.plot(kind="bar", y=["Costs lift trucks", "Costs lead time trucks", "Costs regular hours lift truck drivers",
                 "Costs over time lift truck drivers"], stacked=True, x="Number of lift trucks available", ax=ax, width=0.3, position=1, legend=False, alpha=0.5)

    plt.ylim(0,15000)
    plt.title("1 lift truck vs 2 lift truck for unloading")
    plt.ylabel("Costs per day")
    plt.show()


    # change location
    experiment2 = df[(df["Lowerbound unloading speed for load type A"] == 10) & (
            df["Lowerbound unloading speed for load type C"] == 13) & (df["Truck interarrival interval"] == 900) & (df["Checks model validity (trucks served in time, figure should be = 0)"] == 0)]
    base2 = df1[(df1["Lowerbound unloading speed for load type A"] == 10) & (
            df1["Lowerbound unloading speed for load type C"] == 13) & (df1["Truck interarrival interval"] == 900) & (df1["Checks model validity (trucks served in time, figure should be = 0)"] == 0)]

    ax = experiment2.plot(kind="bar", y=["Costs lift trucks", "Costs lead time trucks", "Costs regular hours lift truck drivers",
                 "Costs over time lift truck drivers"], stacked=True, x="Number of lift trucks available", width=0.3, position=0)

    base2.plot(kind="bar", y=["Costs lift trucks", "Costs lead time trucks", "Costs regular hours lift truck drivers",
                 "Costs over time lift truck drivers"], stacked=True, x="Number of lift trucks available", ax=ax, width=0.3, position=1, legend=False, alpha=0.5)

    plt.ylim(0, 15000)
    plt.title("1 lift truck vs 2 lift trucks for unloading - A and C switched")
    plt.ylabel("Costs per day")
    plt.show()

    # # change arrival rate + switched location
    experiment3 = df[(df["Lowerbound unloading speed for load type A"] == 10) & (
            df["Lowerbound unloading speed for load type C"] == 13) & (df["Truck interarrival interval"] == 750) & (df["Checks model validity (trucks served in time, figure should be = 0)"] == 0)]
    base3 = df1[(df1["Lowerbound unloading speed for load type A"] == 10) & (
            df1["Lowerbound unloading speed for load type C"] == 13) & (df1["Truck interarrival interval"] == 750) & (df1["Checks model validity (trucks served in time, figure should be = 0)"] == 0)]

    ax = experiment3.plot(kind="bar", y=["Costs lift trucks", "Costs lead time trucks", "Costs regular hours lift truck drivers",
                 "Costs over time lift truck drivers"], stacked=True, x="Number of lift trucks available", width=0.3, position=0)

    base3.plot(kind="bar", y=["Costs lift trucks", "Costs lead time trucks", "Costs regular hours lift truck drivers",
                 "Costs over time lift truck drivers"], stacked=True, x="Number of lift trucks available", ax=ax, width=0.3, position=1, legend=False, alpha=0.5)

    plt.ylim(0, 15000)
    plt.title("1 lift truck vs 2 lift trucks for unloading \n A and C switched - higher arrival rate")
    plt.ylabel("Costs per day")
    plt.show()

    # # change arrival rate + original location
    experiment4 = df[(df["Lowerbound unloading speed for load type A"] == 13) & (
            df["Lowerbound unloading speed for load type C"] == 10) & (df["Truck interarrival interval"] == 750) & (df["Checks model validity (trucks served in time, figure should be = 0)"] == 0)]
    base4 = df1[(df1["Lowerbound unloading speed for load type A"] == 13) & (
            df1["Lowerbound unloading speed for load type C"] == 10) & (df1["Truck interarrival interval"] == 750) & (df1["Checks model validity (trucks served in time, figure should be = 0)"] == 0)]

    ax = experiment4.plot(kind="bar", y=["Costs lift trucks", "Costs lead time trucks", "Costs regular hours lift truck drivers",
                 "Costs over time lift truck drivers"], stacked=True, x="Number of lift trucks available", width=0.3, position=0)

    base4.plot(kind="bar", y=["Costs lift trucks", "Costs lead time trucks", "Costs regular hours lift truck drivers",
                 "Costs over time lift truck drivers"], stacked=True, x="Number of lift trucks available", ax=ax, width=0.3, position=1, legend=False, alpha=0.5)

    plt.ylim(0, 15000)
    plt.title("1 lift truck vs 2 lift trucks for unloading \n A and C not switched - higher arrival rate")
    plt.ylabel("Costs per day")
    plt.show()

if __name__=="__main__":
    # experiment_1()
    # experiment_2()
    experiment_3()
    experiment_4()

    # df = pd.read_excel("output_experiments_assignment3.xlsx")
    #
    # experiment1 = df[(df["Truck interarrival interval"] == 900) & (
    #                          df["Number of lift trucks available"] == 11) & (df["Small truck value"] == 0)]
    # print(experiment1["Total costs per day"])

