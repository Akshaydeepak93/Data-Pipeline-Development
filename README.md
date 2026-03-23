# Data-Pipeline-Development

Company Name : CodeTech IT solutions

Name: Akashy Deepak 

Domain : Data Science

Intern ID : 

Duration : 4 weeks 

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

Overall, this project highlights the importance of data preprocessing and demonstrates how pipelines can simplify and automate data preparation steps in a structured and reusable way.

If you want, I can also make:
✅ Short 5-mark answer
✅ Viva questions & answers
✅ Diagram/flowchart of pipeline

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_values = []
for epoch in range(3):
    total_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_values.append(total_loss)
    print("Epoch:", epoch+1, "Loss:", total_loss)
print("Training Completed")
plt.plot(loss_values)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_values = []
for epoch in range(3):
    total_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_values.append(total_loss)
    print("Epoch:", epoch+1, "Loss:", total_loss)
print("Training Completed")
plt.plot(loss_values)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

Here is a ~500-word description of your PyTorch MNIST training code, suitable for assignment, viva, or report:

This project demonstrates the implementation of a basic neural network model using PyTorch to classify handwritten digits from the MNIST dataset. The goal of this project is to train a model that can accurately recognize digits (0–9) based on image input.

The process begins with importing essential libraries such as Torch, Torchvision, and Matplotlib. Torch provides the framework for building and training neural networks, while Torchvision offers access to popular datasets like MNIST. Matplotlib is used for visualizing the training progress.

Next, the MNIST dataset is loaded using torchvision.datasets.MNIST. This dataset consists of grayscale images of handwritten digits, each of size 28×28 pixels. A transformation is applied using transforms.ToTensor(), which converts images into tensor format and scales pixel values between 0 and 1. The dataset is downloaded (if not already present) and stored locally.

To efficiently process the dataset, a DataLoader is created using torch.utils.data.DataLoader. This divides the dataset into smaller batches (batch size of 64) and shuffles the data to ensure randomness during training. Batching helps in faster computation and better generalization of the model.

The neural network model is defined using nn.Sequential, which allows stacking layers in a linear order. The model consists of:

A Flatten layer to convert the 28×28 image into a 1D vector of size 784.
A fully connected layer (Linear) with 128 neurons.
A ReLU activation function to introduce non-linearity.
A final Linear layer with 10 outputs, corresponding to the 10 digit classes.

The loss function used is CrossEntropyLoss, which is suitable for multi-class classification problems. It measures how well the predicted outputs match the actual labels. The optimizer chosen is Adam, a popular optimization algorithm known for its efficiency and adaptive learning rate. The learning rate is set to 0.001.

The training process runs for 3 epochs. In each epoch, the model processes all batches of training data. For each batch, the following steps are performed:

Gradients are reset using optimizer.zero_grad().
A forward pass is performed to compute predictions.
The loss is calculated using the loss function.
Backpropagation is performed using loss.backward() to compute gradients.
Model parameters are updated using optimizer.step().

The total loss for each epoch is accumulated and stored in a list. This helps in tracking how the model improves over time. After each epoch, the loss value is printed to monitor training progress.

Once training is complete, the loss values are plotted using Matplotlib. The graph shows how the loss decreases over epochs, indicating that the model is learning effectively.

Overall, this project provides a simple introduction to deep learning using PyTorch. It covers key concepts such as dataset loading, neural network design, training loops, loss functions, and performance visualization. This forms the foundation for building more complex models in real-world applications.

If you want, I can also give:
✅ 5-mark short answer
✅ Viva questions & answers
✅ Accuracy testing code (very useful for exams)

from flask import Flask, request
import pickle

app = Flask(__name__)

model = pickle.load(open("Task3\data\model.pkl", "rb"))

@app.route("/")
def home():
    return '''
    <h2>Salary Prediction</h2>
    <form action="/predict" method="post">
    Experience: <input type="text" name="exp">
    <input type="submit">
    </form>
    '''

@app.route("/predict", methods=["POST"])
def predict():
    exp = float(request.form["exp"])
    prediction = model.predict([[exp]])
    return f"Predicted Salary: {prediction[0]}"

app.run(debug=True)    import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("Task3\data\salary.csv")

X = data[['experience']]
y = data['salary']

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model created")
 

Here is a ~500-word description of your Flask salary prediction project, suitable for assignment, viva, or report:

This project demonstrates the development of a simple machine learning web application using Flask for predicting salaries based on years of experience. It integrates model training and deployment, showcasing how machine learning models can be used in real-world applications.

The project consists of two main parts: model creation and web application deployment. In the first part, a dataset named salary.csv is loaded using the Pandas library. This dataset contains two columns: experience (independent variable) and salary (dependent variable). The data is then separated into input features (X) and target output (y), where experience is used to predict salary.

A Linear Regression model from Scikit-learn is used to train the data. Linear Regression is a simple and widely used algorithm for predicting continuous values. The model learns the relationship between experience and salary by fitting a straight line to the data. Once trained, the model is saved using the Pickle library as a file named model.pkl. This step is important because it allows the trained model to be reused later without retraining.

In the second part, a web application is created using the Flask framework. Flask is a lightweight web framework in Python that allows easy integration of machine learning models into web interfaces. The saved model (model.pkl) is loaded into the Flask application using Pickle.

The application defines two routes. The first route ("/") serves as the home page and displays a simple HTML form. This form allows the user to input their years of experience. When the user submits the form, the data is sent to the second route ("/predict") using the POST method.

In the "/predict" route, the input value entered by the user is retrieved using request.form. The value is converted into a float and passed to the trained model for prediction. The model then calculates the predicted salary based on the given experience. The result is displayed back to the user in a simple text format.

The application is run using app.run(debug=True), which enables debugging and automatically reloads the server when changes are made to the code. This is useful during development.

One important aspect to consider is the file path used for loading the model. Proper path formatting (using raw strings or forward slashes) ensures that the file is correctly located and loaded without errors.

Overall, this project illustrates the complete workflow of a machine learning application—from data loading and model training to deployment using a web interface. It highlights how simple models can be integrated into user-friendly applications, making predictions accessible to non-technical users.

This project serves as a strong foundation for more advanced applications, such as adding multiple input features, improving the user interface, or deploying the app online using cloud platforms.

If you want, I can also give:
✅ Folder structure (very important for viva)
✅ HTML improved UI
✅ Deployment steps (Render / Railway / PythonAnywhere)

## Business Problem

A manufacturing company produces two products: Product A and Product B.

Each product requires machine hours and labor hours for production.
The company has limited resources and wants to determine the optimal
production quantities to maximize profit.

## Business Data

Profit per unit:

Product A = ₹40  
Product B = ₹30  

Resource availability:

Machine hours available = 100  
Labor hours available = 80  

Resource requirement per product:

Product A:
Machine hours = 2
Labor hours = 1

Product B:
Machine hours = 1
Labor hours = 1

## Mathematical Formulation

Let:

x = units of Product A  
y = units of Product B  

Objective Function:

Maximize

Z = 40x + 30y

Constraints:

2x + y ≤ 100  (Machine hours)

x + y ≤ 80    (Labor hours)

Great, this is a classic Linear Programming (LP) problem. Let’s solve it step by step in a simple and clear way (perfect for exam/viva).

📌 Step 1: Given Problem

We need to maximize profit:

𝑍
=
40
𝑥
+
30
𝑦
Z=40x+30y

Subject to constraints:

Machine hours:

2
𝑥
+
𝑦
≤
100
2x+y≤100
Labor hours:

𝑥
+
𝑦
≤
80
x+y≤80
Non-negativity:

𝑥
≥
0
,
  
𝑦
≥
0
x≥0,y≥0
📊 Step 2: Find Corner Points

We solve the constraints to get feasible region corner points.

1. When x = 0:
From 
𝑥
+
𝑦
≤
80
x+y≤80 → 
𝑦
=
80
y=80
👉 Point: (0, 80)
2. When y = 0:
From 
2
𝑥
≤
100
2x≤100 → 
𝑥
=
50
x=50
👉 Point: (50, 0)
3. Intersection of both constraints:

Solve:

2
𝑥
+
𝑦
=
100
2x+y=100
𝑥
+
𝑦
=
80
x+y=80

Subtract:

(
2
𝑥
+
𝑦
)
−
(
𝑥
+
𝑦
)
=
100
−
80
(2x+y)−(x+y)=100−80
𝑥
=
20
x=20

Substitute:

𝑥
+
𝑦
=
80
⇒
20
+
𝑦
=
80
⇒
𝑦
=
60
x+y=80⇒20+y=80⇒y=60

👉 Point: (20, 60)

4. Origin:

👉 Point: (0, 0)

📈 Step 3: Evaluate Objective Function

Now calculate profit at each point:

Point (x, y)	Z = 40x + 30y
(0, 0)	0
(0, 80)	2400
(50, 0)	2000
(20, 60)	2600
✅ Step 4: Optimal Solution

👉 Maximum profit is ₹2600

At:

x = 20 units of Product A
y = 60 units of Product B
🎯 Final Answer (for exam)
Optimal production:
Product A = 20 units
Product B = 60 units
Maximum Profit = ₹2600
🧠 Simple Explanation

The company should not produce only one product.
Instead, a combination (20 A + 60 B) gives the highest profit while staying within machine and labor limits.

If you want, I can also:
✅ Draw the graph (important for marks)
✅ Give Python code using scipy
✅ Convert this into 10-mark theory answer or PPT format

give the descrpition of the task of about 500 words

Here is a ~500-word description of your Linear Programming business problem, suitable for assignment, report, or viva:

This task focuses on solving a business optimization problem using the concept of Linear Programming (LP). The objective is to help a manufacturing company determine the best combination of products to produce in order to maximize profit, given limited resources such as machine hours and labor hours.

The company produces two products: Product A and Product B. Each product contributes a certain profit per unit, with Product A generating ₹40 and Product B generating ₹30. However, production is constrained by the availability of resources. Specifically, the company has a maximum of 100 machine hours and 80 labor hours available. Each product consumes these resources differently: Product A requires 2 machine hours and 1 labor hour per unit, while Product B requires 1 machine hour and 1 labor hour per unit.

To solve this problem, decision variables are defined. Let x represent the number of units of Product A and y represent the number of units of Product B. The goal is to maximize the total profit, which is expressed as a linear objective function: Z = 40x + 30y. This function calculates the total revenue based on the number of units produced for each product.

The constraints of the problem are formulated based on resource limitations. The machine hour constraint is represented as 2x + y ≤ 100, ensuring that the total machine hours used do not exceed the available capacity. Similarly, the labor hour constraint is expressed as x + y ≤ 80. Additionally, non-negativity constraints (x ≥ 0, y ≥ 0) are applied, as negative production quantities are not feasible in a real-world scenario.

The feasible region is determined by these constraints and represents all possible combinations of x and y that satisfy the conditions. The optimal solution lies at one of the corner points of this region. By evaluating the objective function at each corner point, the maximum profit can be identified.

This task demonstrates how mathematical modeling can be applied to real-world business problems. Linear Programming provides a structured approach to decision-making, enabling companies to utilize their limited resources efficiently. Instead of relying on guesswork, the company can use this method to make data-driven decisions that maximize profitability.

Overall, this problem highlights the importance of optimization techniques in operations management and production planning. It shows how businesses can improve efficiency, reduce waste, and increase profit by applying simple mathematical models. The concepts used in this task are widely applicable in industries such as manufacturing, logistics, finance, and supply chain management, making Linear Programming a valuable tool for solving practical problems.


output  of the task :

<img width="866" height="482" alt="Screenshot 2026-03-23 161307" src="https://github.com/user-attachments/assets/8536c34f-d801-4966-adcd-59470c21641e" />

✅ 5-mark short answer
✅ Viva questions & answers
✅ Python code using PuLP or SciPy
