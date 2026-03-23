import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("Task1\student_data.csv")
print("Original Data:")
print(data)
numeric_data = data.select_dtypes(include=['int64','float64'])
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),  
    ("scaler", StandardScaler())                   
])
prepared_data = pipeline.fit_transform(numeric_data)
prepared_df = pd.DataFrame(prepared_data, columns=numeric_data.columns)
print("\nPrepared Data:")
print(prepared_df)
prepared_df.to_csv("prepared_data.csv", index=False)

print("\nData Pipeline Completed Successfully")
