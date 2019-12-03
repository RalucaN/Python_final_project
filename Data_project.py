import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates

# Step 1 - Reading/importing the data set

# Method 1
def read_data ():
    data = []

    with open('project_data.csv', 'r', encoding="utf8") as project_data_csv:
        spreadsheet = csv.DictReader(project_data_csv)
        for row in spreadsheet:
            data.append(row)

    return data

# Method 2
data_panda = pd.read_csv("project_data.csv")

# Step 2 - Exploring the data set

def run ():
    data = read_data()
    data_pd = pd.DataFrame(data_panda)
    # Testing if the data was imported successfully by printing certain lines
    print(data_pd.loc[[1, 3, 5],])

    # Exploring the columns
    i = 0
    for col in data_pd.columns:
        i = i + 1
        print("The name of column {} is {}.".format(i, col))

    # Finding the type of data of each column
    print("The data type of each column: ")
    print(data_pd.dtypes)

    # Missing data
    # a. Identifying the columns with no data
    print("Number of missing values per each column:")
    print(data_pd.isnull().sum())
    print("Percentage of missing data in each column:")
    sum_miss_data = (data_pd.isnull().sum()) / len(data_pd)
    print(sum_miss_data)

    # b. Dropping columns with more than 50% missing data
    max_missing = 0.5
    data2 = data_pd.loc[:, (sum_miss_data <= max_missing)]
    print(data2.columns)

    #   null_data2 = data2[data2.isna().any(axis=1)]
    #  print(null_data2)

    # Extracting the authors
    data3 = data2.Authors.str.split(', ', expand=True).stack().str.strip().reset_index(level=1, drop=True)
    data3 = pd.DataFrame(data3, columns=["Author"])
    data4 = pd.merge(data2, data3, right_index=True, left_index=True)
    data4 = data4.reset_index(drop=True)
    print(data4.loc[0:3, ])

# Step 3 - Descriptive statistics
    # a. Min, max, average and distribution of the data

    print(data4.describe())
    print(data4["Year"].value_counts())
    print(data4["Author"].value_counts())
    print(data4["Publisher"].value_counts())

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    correlations = data4.corr(method='pearson')
    print(correlations)
    print("start")
    for col2 in list(data4.columns):
        if col2.dtype == np.object:
            print("this is an object")
      #  else:
       #     print(col.descriebe())
    print("end")

    # b. Number of papers by author, year, publication, citations
    pub_year = data4["Year"].value_counts()
    sns.set(style="darkgrid")
    ax=sns.barplot(pub_year.index, pub_year.values, alpha=0.9)
    plt.xticks(rotation=90)
    plt.title('Publications by Year')
    plt.ylabel("Number of Publications", fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.show()





    # c. Most cited authors etc.

# Step 4 - Linear models

run()
