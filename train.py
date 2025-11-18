import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Loading Olivetti faces dataset...")
# Load the dataset
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = data.data
y = data.target

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Number of classes: {len(np.unique(y))}")

# Split the data into 70% train and 30% test
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train Decision Tree Classifier
print("Training Decision Tree Classifier...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on training set
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Training accuracy: {train_accuracy:.4f}")

# Make predictions on test set
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the model
print("Saving model as savedmodel.pth...")
joblib.dump(model, 'savedmodel.pth')
print("Model saved successfully!")

print("\nTraining completed!")
