import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the diabetes dataset
diabetes_df = pd.read_csv("diabetes.csv")

# Visualize the distribution of the data
sns.displot(diabetes_df, x="Outcome", bins=2)

# Split the dataset into training and test sets
X = diabetes_df.drop(columns=["Outcome"])
y = diabetes_df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a Gaussian Naive Bayes classifier on the training data
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict the outcome of the test data
y_pred = gnb.predict(X_test)

# Combine the test data and the predicted outcomes into a new dataframe
test_df = X_test.copy()
test_df["Outcome"] = y_test
test_df["Predicted"] = y_pred

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print the classification report
report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{report}")

# Visualize the classification results
sns.displot(test_df, x="Outcome", hue="Predicted", bins=2)
plt.show()
