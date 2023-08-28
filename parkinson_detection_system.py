import pandas as pd
import scipy.stats as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read csv into a DataFrame
df = pd.read_csv("po1_data.csv")

# split DataFrame into two subsets:
# df_healthy refers to a healthy group people
# df_parkinson referst to a group of people who has a parkinson disease

df_healthy = df[df["status"] == 0]
df_parkinson = df[df["status"] == 1]

# salient_var is to store salient variables that distinguish people who has pd from healthy people
salient_var = []

for key_var in df.columns:
    if key_var != 'status' and key_var != 'subject':
        sample_healthy = df_healthy[key_var].to_numpy()
        sample_parkinson = df_parkinson[key_var].to_numpy()
        
        
        # Creating a box plot to compare distributions between healthy people and people who has parkinson
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=[sample_healthy, sample_parkinson])
        plt.title(f"Box Plot of {key_var}")
        plt.xticks([0, 1], ["Healthy", "Parkinson"])
        plt.show()
        
        # Creating a histogram to know the distributions of data
        plt.figure(figsize=(8, 6))
        plt.hist(sample_healthy, bins=20, alpha=0.5, label='Healthy')
        plt.hist(sample_parkinson, bins=20, alpha=0.5, label='Parkinson')
        plt.title(f"Histogram of {key_var}")
        plt.xlabel(key_var)
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()


        # print values in samples
        # print("Sample of healthy people: ", sample_healthy)
        # print("Sample of parkinson people: ", sample_parkinson)

        # compute basic statistics for two samples
        print("\n Computing basic statistics of samples ...")

        # the basic statistics of sample_healthy:
        x_bar1 = np.mean(sample_healthy)
        s1 = np.std(sample_healthy, ddof=1)  # Use ddof=1 for sample standard deviation
        n1 = len(sample_healthy)
        print(n1)
        print("\t Statistics of sample 1: %.3f (mean), %.3f (std. dev.), and %d (n)." % (
            x_bar1, s1, n1))


       # the basic statistics of sample_parkinson:
        x_bar2 = np.mean(sample_parkinson)
        s2 = np.std(sample_parkinson, ddof=1)  # Use ddof=1 for sample standard deviation
        n2 = len(sample_parkinson)
        print("\t Statistics of sample 2: %.3f (mean), %.3f (std. dev.), and %d (n)." % (
            x_bar2, s2, n2))


        # implementing z-statistics
        # Calculate pooled standard error
        pooled_std_error = np.sqrt((s1**2 / n1) + (s2**2 / n2))

        # Calculate the difference in means
        mean_difference = x_bar1 - x_bar2

        # Calculate the z-statistic
        z_statistic = mean_difference / pooled_std_error
        
        # Calculate the p-value using the cumulative distribution function (CDF)
        p_val = 1 - st.norm.cdf(z_statistic)

        print("\n Conclusion:")
        if p_val < 0.05:
                print("\t We reject the null hypothesis for: ")
                salient_var.append(key_var)
        else:
            print("\t We accept the null hypothesis")

print("salient variables are: ")
print(salient_var)