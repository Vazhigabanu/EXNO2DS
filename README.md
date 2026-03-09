# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT:
~~~
# ----------------------------------------
# Step 1: Import Required Packages
# ----------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
~~~
~~~
# ----------------------------------------
# Step 2: Load the Dataset
# ----------------------------------------
data = pd.read_csv("Exp_2_dataset_titanic_dataset.csv")

print("\nDataset Loaded Successfully\n")
print(data.head())
print("\nDataset Info:\n")
print(data.info())
print(data.describe())
~~~
~~~
# ----------------------------------------
# Step 3: Data Cleansing - Handle Missing Values
# ----------------------------------------
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].fillna(data[column].mode()[0])   # Mode for categorical
    else:
        data[column] = data[column].fillna(data[column].median())   # Median for numerical

print("\nMissing values handled successfully.\n")
~~~
~~~
# ----------------------------------------
# Step 4: Boxplot to Analyze Outliers (Age & Fare)
# ----------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x=data["Age"])
plt.title("Boxplot - Age")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=data["Fare"])
plt.title("Boxplot - Fare")
plt.show()
~~~
<img width="625" height="513" alt="Screenshot 2026-03-09 112615" src="https://github.com/user-attachments/assets/d5602853-c8b4-4566-a3bf-627bfe35cab6" />

<img width="654" height="523" alt="Screenshot 2026-03-09 112621" src="https://github.com/user-attachments/assets/371cd731-27e2-4337-9e0c-e04a4e41934f" />

~~~
# ----------------------------------------
# Step 5: Remove Outliers Using IQR Method
# ----------------------------------------
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

data = remove_outliers_iqr(data, "Age")
data = remove_outliers_iqr(data, "Fare")

print("Outliers removed using IQR method.\n")
~~~
~~~
# ----------------------------------------
# Step 6: Countplot for Categorical Data
# ----------------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=data)
plt.title("Countplot - Survival Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=data)
plt.title("Countplot - Gender Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=data)
plt.title("Countplot - Passenger Class Distribution")
plt.show()
~~~
<img width="791" height="513" alt="Screenshot 2026-03-09 112629" src="https://github.com/user-attachments/assets/c38d49f4-0633-40a4-9c59-a090c8e0e5c5" />

<img width="752" height="519" alt="Screenshot 2026-03-09 112635" src="https://github.com/user-attachments/assets/36b0909e-c8d8-4637-ac87-d7122a8291c3" />

<img width="746" height="524" alt="Screenshot 2026-03-09 112640" src="https://github.com/user-attachments/assets/a6a12671-c2e4-48c4-96b2-3019de43787d" />


~~~
# ----------------------------------------
# Step 7: Displot for Univariate Distribution
# ----------------------------------------
sns.displot(data["Age"], kde=True, height=4, aspect=1.5)
plt.title("Displot - Age Distribution")
plt.show()

sns.displot(data["Fare"], kde=True, height=4, aspect=1.5)
plt.title("Displot - Fare Distribution")
plt.show()
~~~
<img width="782" height="538" alt="Screenshot 2026-03-09 112648" src="https://github.com/user-attachments/assets/20a8d045-843b-432b-ac1a-f2c3d404579a" />

<img width="963" height="580" alt="Screenshot 2026-03-09 112654" src="https://github.com/user-attachments/assets/4a234b04-9e3e-4501-a560-852b63facc6e" />


~~~
# ----------------------------------------
# Step 8: Cross Tabulation
# ----------------------------------------
print("\nCross Tabulation: Sex vs Survived\n")
print(pd.crosstab(data["Sex"], data["Survived"]))

print("\nCross Tabulation: Pclass vs Survived\n")
print(pd.crosstab(data["Pclass"], data["Survived"]))
~~~
~~~
# ----------------------------------------
# Step 9: Heatmap for Correlation Analysis
# ----------------------------------------
plt.figure(figsize=(8,6))
correlation_matrix = data.select_dtypes(include=np.number).corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - Titanic Dataset")
plt.show()
~~~
<img width="954" height="733" alt="Screenshot 2026-03-09 112747" src="https://github.com/user-attachments/assets/faa24b26-989f-4046-a87f-622ab3ebfda4" />


# RESULT
        Thus we have cleaned the data and removed the outliers by detection using IQR and Z-score method.
