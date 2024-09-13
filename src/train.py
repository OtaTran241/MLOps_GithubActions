from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

if not os.path.exists('models'):
    os.makedirs('models')

# Save the model to disk
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training completed!")
