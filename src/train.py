from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import pickle
import os

# Load dataset
data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train) * 100
test_score = model.score(X_test, y_test) * 100

if not os.path.exists('models'):
    os.makedirs('models')

# Save the model to disk
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open("metrics.txt", 'w') as outfile:
    outfile.write("Metric of Results:\n")
    outfile.write("Training variance explained: %2.1f%%\n" % train_score)
    outfile.write("Test variance explained: %2.1f%%\n" % test_score)

# image formatting
axis_fs = 18 
title_fs = 22 

y_pred = model.predict(X_test) + np.random.normal(0,0.25,len(y_test))
y_jitter = y_test + np.random.normal(0,0.25,len(y_test))
res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])

ax = sns.scatterplot(x="true", y="pred",data=res_df)
ax.set_aspect('equal')
ax.set_xlabel('True ',fontsize = axis_fs) 
ax.set_ylabel('Predicted ', fontsize = axis_fs)#ylabel
ax.set_title('Residuals', fontsize = title_fs)

# Make it pretty- square aspect ratio
ax.plot([1, 10], [1, 10], 'black', linewidth=1)
plt.ylim((2.5,8.5))
plt.xlim((2.5,8.5))

plt.tight_layout()
plt.savefig("residuals.png",dpi=120) 

print("Model training completed!")
