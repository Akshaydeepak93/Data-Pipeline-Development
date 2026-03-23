# Data-Pipeline-Development

Company Name : CodeTech IT solutions

Name: Akashy Deepak 

Domain : Data Science

Intern ID : CTIS6565

Duration : 4 weeks 

mentor : Neela Santosh

descprition of the task :

This project demonstrates a simple yet effective data preprocessing pipeline using Python libraries such as Pandas and Scikit-learn. The primary objective of this pipeline is to clean and transform raw data into a structured format that is suitable for machine learning models.

The process begins with importing the necessary libraries. Pandas is used for data manipulation and handling, while Scikit-learn provides tools for building the preprocessing pipeline, including imputation and scaling techniques. The dataset is loaded using the read_csv() function from a CSV file named student_data.csv. Once loaded, the original dataset is printed to give a clear view of the raw data, including any missing values or inconsistencies.

Next, the pipeline focuses on selecting only the numerical features from the dataset. This is done using the select_dtypes() function, which filters columns of type integer and float. This step is important because many preprocessing techniques, such as scaling and mean imputation, are applicable only to numerical data.

After isolating the numerical data, a preprocessing pipeline is constructed using Scikit-learn’s Pipeline class. A pipeline allows multiple data transformation steps to be applied sequentially in a clean and efficient manner. In this project, the pipeline consists of two main steps: imputation and scaling.

The first step is handled by the SimpleImputer with the strategy set to "mean". This step replaces any missing values in the dataset with the mean of the respective column. Handling missing data is crucial, as most machine learning algorithms cannot work with incomplete datasets.

The second step in the pipeline is feature scaling using StandardScaler. This technique standardizes the data by transforming it so that each feature has a mean of zero and a standard deviation of one. Scaling ensures that all features contribute equally to the model and prevents bias caused by differences in value ranges.

Once the pipeline is defined, it is applied to the numerical data using the fit_transform() method. This method first learns the parameters (like mean and standard deviation) from the data and then transforms it accordingly. The output is a NumPy array containing the processed data.

To make the transformed data easier to interpret, it is converted back into a Pandas DataFrame with the original column names preserved. The cleaned and scaled dataset is then printed as "Prepared Data" for comparison with the original dataset.

Finally, the processed data is saved into a new CSV file named prepared_data.csv using the to_csv() function. This allows the cleaned dataset to be reused for further analysis or machine learning tasks. A success message is printed at the end to indicate that the data pipeline has been executed successfully.

output of the task :


Overall, this project highlights the importance of data preprocessing and demonstrates how pipelines can simplify and automate data preparation steps in a structured and reusable way.

output of the task :
<img width="866" height="482" alt="Screenshot 2026-03-23 161307" src="https://github.com/user-attachments/assets/053dd4fe-cf14-413e-b62b-c31fed05f545" />
