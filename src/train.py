import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,make_scorer, fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

df = pd.read_csv('data/Bank_Customer_Churn_Prediction.csv')

na_rows = df[df.isna().any(axis=1)]
na_rows.shape[0]

label_country_encoder = LabelEncoder()
df['country'] = label_country_encoder.fit_transform(df['country'])
decode_country_dict = {idx: label for idx, label in enumerate(label_country_encoder.classes_)}

label_gender_encoder = LabelEncoder()
df['gender'] = label_gender_encoder.fit_transform(df['gender'])
decode_gender_dict = {idx: label for idx, label in enumerate(label_gender_encoder.classes_)}

X = df.drop(['customer_id', 'churn'],axis=1)
Y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

classifiers = {
    'Logistic Regression': (LogisticRegression(), {'classifier__C': [0.01 ,0.1, 1, 10],'classifier__max_iter': [1000]}),
    'Support Vector Classifier': (SVC(probability=True), {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']}),
    'Naive Bayes': (GaussianNB(), {}),
    'Decision Tree': (DecisionTreeClassifier(), {'classifier__max_depth': [None, 10, 20, 40]}),
    'Random Forest': (RandomForestClassifier(), {'classifier__n_estimators': [50, 100, 200, 500]}),
    'AdaBoost': (AdaBoostClassifier(), {'classifier__n_estimators': [50, 100, 200,500]}),
    'Gradient Boosting': (GradientBoostingClassifier(), {'classifier__n_estimators': [50, 100, 200, 500]}),
    'XGBoost': (XGBClassifier(), {'classifier__n_estimators': [50, 100, 200]}),
    'Catboost': (CatBoostClassifier(silent=True), {'classifier__iterations': [50, 100, 200]}),
    'K Nearest Neighbors': (KNeighborsClassifier(), {'classifier__n_neighbors': [3, 5, 7, 9]}),
    'Gaussian Process': (GaussianProcessClassifier(), {}),
}

scalers={'Standard Scaler' :StandardScaler(),
         'Min Max Scaler':MinMaxScaler(),
         'Roobust Scaler': RobustScaler(),
         'No Scaler': None
        }

stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
precision_scorer = make_scorer(precision_score)
recall_scorer = make_scorer(recall_score)

best_recall = 0
best_classifier = None
best_scaler = None
best_params = None

for classifier_name, (classifier, param_grid) in classifiers.items():
    for scaler_name, scaler in scalers.items():
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', classifier)
        ])

        grid_search_recall = GridSearchCV(pipeline, param_grid, cv=stratified_kfold, scoring=recall_scorer, n_jobs=-1)
        grid_search_recall.fit(X_train, y_train)
        mean_recall = grid_search_recall.best_score_
        
        if mean_recall > best_recall:
            best_recall = mean_recall
            best_classifier = classifier_name
            best_scaler_ = scaler_name
            best_params = grid_search_recall.best_params_

best_model = classifiers[best_classifier][0]
best_scaler_ = scalers[best_scaler]
best_params_ = best_params
best_params_cleaned = {}
for key, value in best_params_.items():
    new_key = key.split('__')[-1]
    best_params_cleaned[new_key] = value

best_churn_model = Pipeline([
    ('scaler', best_scaler_),
    ('classifier', best_model.set_params(**best_params_cleaned))
])
best_churn_model.fit(X_train, y_train)

train_score = best_churn_model.score(X_train, y_train) * 100
test_score = best_churn_model.score(X_test, y_test) * 100

y_train_proba = best_churn_model.predict_proba(X_train)[:, 1]
best_f05_score = 0
best_threshold = 0
for i in range(10000):
    y_pred_thresholded = (y_train_proba >= i/10000).astype(int)
    f05= fbeta_score(y_train, y_pred_thresholded, beta=0.5)

    if f05 > best_f05_score:
        best_f05_score = f05
        best_threshold = i/10000

if not os.path.exists('models'):
    os.makedirs('models')

with open('models/model.pkl', 'wb') as f:
    pickle.dump(best_churn_model, f)

with open("metrics.txt", 'w') as outfile:
    outfile.write("Metric of Results:\n")
    outfile.write("Training variance explained: %2.1f%%\n" % train_score)
    outfile.write("Test variance explained: %2.1f%%\n" % test_score)
    outfile.write("Best model for Recall: %2.1f%%\n", best_classifier)
    outfile.write("Best params for Recall: %2.1f%%\n", best_params)
    outfile.write("Best scaler for Recall: %2.1f%%\n", best_scaler)
    outfile.write("Best mean Recall score: %2.1f%%\n", best_recall)
    outfile.write("Best threshold for f05 score: %2.1f%%\n", best_threshold)

test_cm = confusion_matrix(y_test, y_pred_thresholded)

plt.figure(figsize=(10, 7))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

test_cr = classification_report(y_test, y_pred_thresholded, output_dict=True)
cr_df = pd.DataFrame(test_cr).transpose()

plt.figure(figsize=(10, 7))
plt.title('Classification Report')
ax = plt.gca()
ax.axis('off')
table = plt.table(cellText=cr_df.values,
                  colLabels=cr_df.columns,
                  rowLabels=cr_df.index,
                  cellLoc='center',
                  loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(list(range(len(cr_df.columns))))

plt.savefig('classification_report.png', bbox_inches='tight')

print("Model training completed!")
