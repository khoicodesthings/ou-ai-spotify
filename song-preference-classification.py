import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

# Load the dataset
file_path = 'data/final.csv'
data = pd.read_csv(file_path)

data['key'] = data['key'].astype('category')

data['mode'] = data['mode'].astype('category')

# remove duplicates
data = data.drop_duplicates()

# Prepare the features (X) and target (y)
X = data.drop(columns=[ 'preference'])  # Drop non-numeric and target columns
y = data['preference']

# Shuffle the data
X, y = shuffle(X, y, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize different classifiers
models = {
    # 'Random Forest': RandomForestClassifier(random_state=42)
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    #  'Decision Tree': DecisionTreeClassifier(random_state=42)
    # 'Support Vector Classifier': SVC(random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train.drop(columns=['name']), y_train)
    # Make predictions
    y_pred = model.predict(X_test.drop(columns=['name']))
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    # Print results
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    print("-" * 50)  # Separator for better readability

    # Print a few rows for visualization
    print("Sample Predictions:")
    for i in range(5):  # Change the range if you want more or fewer rows
        print(f"Row {i}:")
        print(f"Actual: {y_test.iloc[i]}, Predicted: {y_pred[i]}")
        print(f"Features: {X_test.iloc[i].values}")
    print("=" * 50)  # Separator for better readability
    # Print misclassified rows
    misclassified_indices = [i for i in range(len(y_test)) if y_test.iloc[i] != y_pred[i]]
    print(f"Total Misclassified: {len(misclassified_indices)}")
    print("Misclassified Songs:")
    for i in misclassified_indices:
        print(f"Song: {X_test.iloc[i]['name']}, Actual: {y_test.iloc[i]}, Predicted: {y_pred[i]}")
        print(f"Features: {X_test.iloc[i].drop('name').values}")
    print("-" * 50) 

