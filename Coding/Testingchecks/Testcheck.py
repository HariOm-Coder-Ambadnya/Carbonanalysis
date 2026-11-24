import pandas as pd

file_path = r"C:\Users\rishi\Documents\Cloud_G45_Project\AzureFunctionsInvocationTraceForTwoWeeksJan2021\AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())