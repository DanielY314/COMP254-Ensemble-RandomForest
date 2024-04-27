import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, cross_validate,StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint



#Load & check the data
df_dongheun = pd.read_csv(r'C:\Users\danie\Desktop\Term4\AI\Assignment\Lab Assignment5_ensemble_and_random_forest\pima-indians-diabetes.csv')

column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age','class']

df_dongheun = pd.read_csv(r'C:\Users\danie\Desktop\Term4\AI\Assignment\Lab Assignment5_ensemble_and_random_forest\pima-indians-diabetes.csv', names=column_names)
column_info=df_dongheun.dtypes
missing_values = df_dongheun.isnull().sum()
statistics = df_dongheun.describe()
clss_distribution = df_dongheun['class'].value_counts(normalize=True)

print("column_info:" , column_info)
print("missing_values:" , missing_values)
print("statistics:" , statistics)
print("clss_distribution:" , clss_distribution)

#Pre-process and prepare the data for machine learning
X = df_dongheun.drop('class', axis=1)
y = df_dongheun['class']

X_train_dongheun, X_test_dongheun, y_train_dongheun, y_test_dongheun = train_test_split(X,y,test_size=0.3, random_state=42)
transformer_dongheun = StandardScaler()

X_train_dongheun_scaled = transformer_dongheun.fit_transform(X_train_dongheun)
X_test_dongheun_scaled = transformer_dongheun.transform(X_test_dongheun)

print(X_train_dongheun_scaled.shape)
print(X_test_dongheun_scaled.shape)

#Exercise 1 Hard Voting
logistic_regression_dongheun = LogisticRegression(max_iter=1400)
random_forest_dongheun = RandomForestClassifier()
svc_dongheun = SVC()
decision_tree_dongheun = DecisionTreeClassifier(criterion="entropy", max_depth=42)
extra_trees_dongheun = ExtraTreesClassifier()

estimators = [('logistic_regression_dongheun', logistic_regression_dongheun),
              ('random_forest_dongheun', random_forest_dongheun),
              ('svc_dongheun', svc_dongheun),
              ('decision_tree_dongheun', decision_tree_dongheun),
              ('extra_tree_dongheun', extra_trees_dongheun)]

voting_classifier_dongheun = VotingClassifier(estimators=estimators, voting='hard')
voting_classifier_dongheun.fit(X_train_dongheun_scaled, y_train_dongheun)
predictions_voting = voting_classifier_dongheun.predict(X_test_dongheun_scaled[:3])
results=[]

classifiers =  [logistic_regression_dongheun, random_forest_dongheun, svc_dongheun, decision_tree_dongheun, extra_trees_dongheun, voting_classifier_dongheun]
for clf in classifiers:
    clf_name = clf.__class__.__name__
    if clf_name != "VotingClassifier":
        clf.fit(X_train_dongheun_scaled, y_train_dongheun)
    predictions = clf.predict(X_test_dongheun_scaled[:3])
    results.append((clf_name, predictions.tolist(), y_test_dongheun[:3].tolist()))

print("Hard voting:", results)
print()

#Soft voting
svc_dongheun_soft = SVC(probability=True)

estimators_soft = [('logistic_regression_dongheun', logistic_regression_dongheun),
              ('random_forest_dongheun', random_forest_dongheun),
              ('svc_dongheun', svc_dongheun_soft),
              ('decision_tree_dongheun', decision_tree_dongheun),
              ('extra_tree_dongheun', extra_trees_dongheun)]
svc_dongheun_soft.fit(X_train_dongheun_scaled, y_train_dongheun)
voting_classifier_dongheun_soft = VotingClassifier(estimators=estimators_soft, voting='soft')


voting_classifier_dongheun_soft.fit(X_train_dongheun_scaled, y_train_dongheun)

predictions = voting_classifier_dongheun_soft.predict(X_test_dongheun_scaled)
accuracy = accuracy_score(y_test_dongheun, predictions)
print("Accuracy:", accuracy)


results_soft = []

classifiers_soft = [logistic_regression_dongheun,random_forest_dongheun,svc_dongheun,decision_tree_dongheun,extra_trees_dongheun]


for clf in classifiers_soft:
    clf_name = clf.__class__.__name__
    
    if clf_name not in ["VotingClassifier", "SVC"]:
        clf.fit(X_train_dongheun_scaled, y_train_dongheun)  # Use training data for fitting
    predictions = clf.predict(X_test_dongheun_scaled[:3])  # Predict on test data
    results_soft.append((clf_name, predictions.tolist(), y_test_dongheun[:3].tolist()))  # Use test labels for comparison

print("Soft voting:", results_soft)

#Random forests & Extra Trees
extra_trees_dongheun = ExtraTreesClassifier()
decision_tree_dongheun = DecisionTreeClassifier(criterion="entropy", max_depth=42)

pipeline1_dongheun = Pipeline([
    ('transformer_dongheun', StandardScaler()),
    ('extra_trees_dongheun', extra_trees_dongheun)
])

pipeline2_dongheun = Pipeline([
    ('transformer_dongheun', StandardScaler()),
    ('decision_tree_dongheun', decision_tree_dongheun)
])

pipeline1_dongheun.fit(X, y)

pipeline2_dongheun.fit(X, y)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_scores_pipeline1 = cross_val_score(pipeline1_dongheun, X, y, cv=cv)
cv_scores_pipeline2 = cross_val_score(pipeline2_dongheun, X, y, cv=cv)

mean_score_pipeline1 = np.mean(cv_scores_pipeline1)
mean_score_pipeline2 = np.mean(cv_scores_pipeline2)

print("mean_score_pipeline1: ", mean_score_pipeline1)
print("mean_score_pipeline2: ", mean_score_pipeline2)

pipelines = [
    ('Pipeline 1 - Extra Trees', pipeline1_dongheun),
    ('Pipeline 2 - Decision Tree', pipeline2_dongheun)
]

# Dictionary to store results for each pipeline
results = {}

# Loop through each pipeline, predict the test set, and calculate metrics
for name, pipeline in pipelines:
    # Use the pipeline to predict the test set
    y_pred = pipeline.predict(X_test_dongheun_scaled)

    # Calculate evaluation metrics
    cm = confusion_matrix(y_test_dongheun, y_pred)
    precision = precision_score(y_test_dongheun, y_pred)
    recall = recall_score(y_test_dongheun, y_pred)
    accuracy = accuracy_score(y_test_dongheun, y_pred)

    # Store the results in the dictionary
    results[name] = {
        'Confusion Matrix': cm,
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy
    }

    # Print the results for each pipeline
    print(f"Results for {name}:")
    print("Confusion Matrix:")
    print(cm)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}\n")
    
n_estimators_range = np.arange(10, 3001, 20)  # Trees from 10 to 3000 with step of 20
max_depth_range = np.arange(1, 1001, 2)  # Depth from 1 to 1000 with step of 2

param_distributions_34 = {
    'extra_trees_dongheun__n_estimators': n_estimators_range,
    'extra_trees_dongheun__max_depth': max_depth_range
}
 
pipeline1_dongheun = Pipeline([
    ('transformer_dongheun', StandardScaler()),
    ('extra_trees_dongheun', ExtraTreesClassifier())
])

# Setup RandomizedSearchCV
random_search_34 = RandomizedSearchCV(
    estimator=pipeline1_dongheun,
    param_distributions=param_distributions_34,
    n_iter=100,  
    cv=5,        
    verbose=1,
    random_state=42,
    n_jobs=-1   
)

X = df_dongheun[['preg']]  
y = df_dongheun['class']    

random_search_34.fit(X, y)


print("Best Parameters:", random_search_34.best_params_)
print("Best Score:", random_search_34.best_score_)

best_model = random_search_34.best_estimator_
X_test_dongheun_single_feature = X_test_dongheun[['preg']]
y_pred = best_model.predict(X_test_dongheun_single_feature)
accuracy = accuracy_score(y_test_dongheun, y_pred)
precision = precision_score(y_test_dongheun, y_pred)
recall = recall_score(y_test_dongheun, y_pred)
cm = confusion_matrix(y_test_dongheun, y_pred)


accuracy = accuracy_score(y_test_dongheun, y_pred)
precision = precision_score(y_test_dongheun, y_pred)
recall = recall_score(y_test_dongheun, y_pred)
cm = confusion_matrix(y_test_dongheun, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:\n", cm)
