import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#Models#
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
########
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('data/train_dataset.csv')

#Get Features and labels, split train and test
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train.drop(["Urine protein", "hearing(left)", "hearing(right)"], axis=1, inplace=True)
X_test.drop(["Urine protein", "hearing(left)", "hearing(right)"], axis=1, inplace=True)

# select five models to compare
models = {}
models['Logistic Regression'] = LogisticRegression()
models['Support Vector Machines'] = LinearSVC()
models['Decision Trees'] = DecisionTreeClassifier()
models['Random Forest'] = RandomForestClassifier()
models['Gradient Boost'] = GradientBoostingClassifier()
accuracy, precision, recall, f1 = {}, {}, {}, {}

for key in models.keys():
    
    # Fit the classifier
    models[key].fit(X_train, np.ravel(y_train))
    
    # Make predictions
    y_pred = models[key].predict(X_test)
    
    # Calculate metrics
    accuracy[key] = accuracy_score(y_pred, y_test)
    precision[key] = precision_score(y_pred, y_test)
    recall[key] = recall_score(y_pred, y_test)
    f1[key] = f1_score(y_pred, y_test)

#summary as df
df_model = pd.DataFrame(index=models.keys(), columns=['F1 Score', 'Accuracy', 'Precision', 'Recall'])
df_model['F1 Score'] = f1.values()
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()

#Random Forest shows best Auc and F1_Score
df_result = df_model.sort_values(by='F1 Score', ascending=False)
print(df_result)