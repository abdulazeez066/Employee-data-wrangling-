# Employee-data-wrangling-
import pandas as pd
import numpy as np

# Load the large employee dataset
df = pd.read_csv("large_employee_data.csv")

# Display initial information
print("Initial Data Overview:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDataset Shape:", df.shape)

# Handling Missing Values
print("\nHandling Missing Values...")
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Salary'].fillna(df['Salary'].median(), inplace=True)
print("Missing Values After Handling:")
print(df.isnull().sum())

# Convert Joining_Date to a standard format
df['Joining_Date'] = pd.to_datetime(df['Joining_Date'], errors='coerce')
print("\nConverted Joining_Date to datetime format.")

# Remove duplicate records
df.drop_duplicates(subset=['Employee_ID'], keep='first', inplace=True)
print("\nRemoved duplicate records. New shape:", df.shape)

# Standardizing Job_Level text case
df['Job_Level'] = df['Job_Level'].str.lower().str.capitalize()
print("\nStandardized Job_Level column.")

# Detect and handle outliers in Salary using IQR
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['Salary'] = np.where(df['Salary'] > upper_bound, upper_bound, df['Salary'])
df['Salary'] = np.where(df['Salary'] < lower_bound, lower_bound, df['Salary'])
print("\nHandled outliers in Salary column.")

# Feature Engineering: Create Age Groups
df['Age_Group'] = pd.cut(df['Age'], bins=[22, 30, 40, 50, 65], labels=['22-30', '31-40', '41-50', '51-65'])
print("\nCreated Age_Group column.")

# Data Aggregation: Calculate average salary per department
department_avg_salary = df.groupby('Department')['Salary'].mean()
print("\nAverage Salary per Department:")
print(department_avg_salary)

# Normalization of Salary (Min-Max Scaling)
df['Salary_Norm'] = (df['Salary'] - df['Salary'].min()) / (df['Salary'].max() - df['Salary'].min())
print("\nNormalized Salary column.")

# Save the cleaned dataset
df.to_csv("cleaned_large_employee_data.csv", index=False)
print("\nData Wrangling Completed. Cleaned data saved as 'cleaned_large_employee_data.csv'.")
