import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Sample dataset
# Let's pretend we have a dataset with customer demographics (age, income) and buying behavior (purchase_history_score)
# and we want to predict if they will like a product or not (target)

data = {
    'age': [25, 30, 45, 35, 40, 50, 20, 60],
    'income': [50000, 60000, 80000, 120000, 70000, 90000, 30000, 100000],
    'purchase_history_score': [3, 4, 5, 2, 4, 5, 1, 5],
    'will_like_product': [1, 0, 1, 0, 1, 1, 0, 1]  # 1 means 'yes', 0 means 'no'
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df.drop('will_like_product', axis=1)
y = df['will_like_product']

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
