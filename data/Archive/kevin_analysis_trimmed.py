import numpy as np 
import pandas as pd 
 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold 
from sklearn.metrics import  confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier 

from imblearn.over_sampling import SMOTE 
import matplotlib.pyplot as plt 

df = pd.read_csv('journeys.csv') 
targets = df['Conversion'] 
df.drop(['Journey Start Date', 'Journey End Date', 'Events Combo', 'User-Journey'], axis=1, inplace=True) 
df = pd.get_dummies(df) 
 
X_train, X_test, y_train, y_test = train_test_split(df, targets, stratify=targets) 
X_train.drop(['Conversion'], axis=1, inplace=True) 
X_test.drop(['Conversion'], axis=1, inplace=True) 

#scaler = StandardScaler() 
#scaler.fit(X_train) 
 
#X_train = scaler.transform(X_train) 
#X_test = scaler.transform(X_test) 
 
clf = RandomForestClassifier(n_jobs=-1) 
 
param_grid = { 
    'min_samples_split': [3, 5, 10],  
    'n_estimators' : [100, 300], 
    'max_depth': [3, 5, 15, 25], 
    'max_features': [3, 5, 10, 20] 
} 
 
scorers = { 
    'f1_score': make_scorer(f1_score), 
} 
 
 
def grid_search_wrapper(refit_score='fb_score'): 
    """ 
    fits a GridSearchCV classifier using refit_score for optimization 
    prints classifier performance metrics 
    """ 
    skf = StratifiedKFold(n_splits=10) 
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score, 
                           cv=skf, return_train_score=True, n_jobs=-1) 
    grid_search.fit(X_train.values, y_train.values) 
 
    # make the predictions 
    y_pred = grid_search.predict(X_test.values) 
 
    print('Best params for {}'.format(refit_score)) 
    print(grid_search.best_params_) 
 
    # confusion matrix on the test data. 
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score)) 
    print(pd.DataFrame(confusion_matrix(y_test, y_pred), 
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos'])) 
    return grid_search 
 
 
grid_search_clf = grid_search_wrapper(refit_score='fb_score') 
y_scores = grid_search_clf.predict_proba(X_test)[:, 1]  
 
random_forest = RandomForestClassifier(n_estimators=100) 
random_forest.fit(X_train, y_train) 
random_forest_preds = random_forest.predict(X_test) 

xgb_model = xgb.XGBClassifier(learning_rate=0.001, 
                            max_depth = 1,  
                            n_estimators = 100, 
                              scale_pos_weight=5) 
xgb_model.fit(x_train, y_train) 
xgb_predict = xgb_model.predict(x_val) 
 
xgb_model = xgb.XGBClassifier(learning_rate=0.001, 
                            max_depth = 10,  
                            n_estimators = 100, 
                              scale_pos_weight=4) 
xgb_model.fit(x_train, y_train) 
 
xgb_predict=xgb_model.predict(x_val) 
 
print(confusion_matrix(y_val,xgb_predict)) 
print(classification_report(y_val,xgb_predict)) 
 
 
xgb_c = xgb.XGBClassifier(objective='binary:logistic') 
 
xgb_c.fit(x_train, y_train) 
 
y_pred = xgb_c.predict(x_test) 
 
params = { 'max_depth': [3,6,10], 
           'learning_rate': [0.01, 0.05, 0.1], 
           'n_estimators': [100, 500, 1000], 
           'colsample_bytree': [0.3, 0.7]} 
 
xgb_c = xgb.XGBClassifier(objective='binary:logistic', seed = 7406) 
clf = GridSearchCV(estimator=xgb_c,  
                   param_grid=params, 
                   scoring='f1',  
                   verbose=1) 
cv_model = clf.fit(x_train, y_train) 
print("Best parameters:", clf.best_params_) 
print("Best F1 Score: ", clf.best_score_) 
 
xgb_predict=cv_model.predict(x_test) 

classifier = KNeighborsClassifier(n_neighbors=5) 
classifier.fit(X_train, y_train) 
 
y_pred = classifier.predict(X_test) 
from sklearn.metrics import classification_report, confusion_matrix 
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred)) 
 
# Calculating error for K values between 1 and 40 
for i in range(1, 40): 
    knn = KNeighborsClassifier(n_neighbors=i) 
    knn.fit(X_train, y_train) 
    pred_i = knn.predict(X_test) 
     
    error.append(np.mean(pred_i != y_test)) 
    f_score.append(metrics.f1_score(y_test, list(pred_i), average='weighted')) 
 
metrics.f1_score(np.array(y_test), pred_i, average='weighted') 
 
plt.figure(figsize=(12, 6)) 
plt.plot(range(1, 40), f_score, color='black', linestyle='dashed', marker='x', 
         markerfacecolor='green', markersize=10) 
plt.title('F1 Score by K Value') 
plt.xlabel('K Value') 
plt.ylabel('F1 Score') 
 
 
