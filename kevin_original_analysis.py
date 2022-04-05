import numpy as np 
import pandas as pd 
 
from sklearn.preprocessing import LabelBinarizer 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold 
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score,  accuracy_score, precision_score, confusion_matrix, f1_score, fbeta_score, 
roc_auc_score 
 
import matplotlib.pyplot as plt 
plt.style.use("ggplot") 
 
df = pd.read_csv('journeys_test.csv') 
 
targets = df['Conversion'] 
 
df.drop(['Journey Start Date', 'Journey End Date', 'Events Combo', 'User-Journey'], axis=1, inplace=True) 
df = pd.get_dummies(df) 
 
X_train, X_test, y_train, y_test = train_test_split(df, targets, stratify=targets) 
X_train.drop(['Conversion'], axis=1, inplace=True) 
X_test.drop(['Conversion'], axis=1, inplace=True) 
 
clf = RandomForestClassifier(n_jobs=-1) 
 
param_grid = { 
    'min_samples_split': [3, 5, 10],  
    'n_estimators' : [100, 300], 
    'max_depth': [3, 5, 15, 25], 
    'max_features': [3, 5, 10, 20] 
} 
 
scorers = { 
'precision_score': make_scorer(precision_score), 
'recall_score': make_scorer(recall_score), 
'accuracy_score': make_scorer(accuracy_score), 
'f1_score': make_scorer(f1_score), 
'fb_score': make_scorer(fbeta_score, beta=2), 
'auc_score': make_scorer(roc_auc_score) 
 
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
grid_search_clf 
y_scores = grid_search_clf.predict_proba(X_test)[:, 1] 
p, r, thresholds = precision_recall_curve(y_test, y_scores) 
 
def adjusted_classes(y_scores, t): 
    """ 
    This function adjusts class predictions based on the prediction threshold (t). 
    Will only work for binary classification problems. 
    """ 
    return [1 if y >= t else 0 for y in y_scores] 
 
def precision_recall_threshold(p, r, thresholds, t=2): 
    """ 
    plots the precision recall curve and shows the current value for each 
    by identifying the classifier's threshold (t). 
    """ 
     
    # generate new class predictions based on the adjusted_classes 
    # function above and view the resulting confusion matrix. 
    y_pred_adj = adjusted_classes(y_scores, t) 
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj), 
                       columns=['pred_neg', 'pred_pos'],  
                       index=['neg', 'pos'])) 
     
    # plot the curve 
    plt.figure(figsize=(8,8)) 
    plt.title("Precision and Recall curve ^ = current threshold") 
    plt.step(r, p, color='b', alpha=0.2, 
             where='post') 
    plt.fill_between(r, p, step='post', alpha=0.2, 
                     color='b') 
    plt.ylim([0.5, 1.01]); 
    plt.xlim([0.5, 1.01]); 
    plt.xlabel('Recall'); 
    plt.ylabel('Precision'); 
     
    # plot the current threshold on the line 
    close_default_clf = np.argmin(np.abs(thresholds - t)) 
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k', 
            markersize=15) 
 
 
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds): 
    """ 
    Modified from: 
    Hands-On Machine learning with Scikit-Learn 
    and TensorFlow; p.89 
    """ 
    plt.figure(figsize=(8, 8)) 
    plt.title("Performance Scores as a function of the Decision Threshold") 
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall") 
    plt.plot(thresholds, 2*(precisions[:-1] * recalls[:-1])/(precisions[:-1] + recalls[:-1]), "r-", label="F1 Score") 
    plt.plot(thresholds, ((1 + 2*2) * precisions[:-1] * recalls[:-1]) / (2*2 * precisions[:-1] + recalls[:-1]), "k-", label="F-beta Score (beta = 2)") 
    plt.ylabel("Score") 
    plt.xlabel("Decision Threshold") 
    plt.legend(loc='best') 
     
    print("Optimal F1 Score: ", max(2*(precisions[:-1] * recalls[:-1])/(precisions[:-1] + recalls[:-1]))) 
     
    y = ((1 + 2*2) * precisions[:-1] * recalls[:-1]) / (2*2 * precisions[:-1] + recalls[:-1]) 
    print("Optimal Decision Threshold: ", thresholds[np.argmax(y)])  
    opt_threshold = thresholds[np.argmax(y)] 
    return opt_threshold 
 
precision_recall_threshold(p, r, thresholds, opt_threshold) 
 
random_forest = RandomForestClassifier(n_estimators=100) 
random_forest.fit(X_train, y_train) 
random_forest_preds = random_forest.predict(X_test) 
 
decisions = (random_forest.predict(X_test) >= opt_threshold).astype(int) 
 
def plot_roc_curve(fpr, tpr, label=None): 
    """ 
    The ROC curve, modified from  
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91 
    """ 
    plt.figure(figsize=(8,8)) 
    plt.title('ROC Curve') 
    plt.plot(fpr, tpr, linewidth=2, label=label) 
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.axis([-0.005, 1, 0, 1.005]) 
    plt.xticks(np.arange(0,1, 0.05), rotation=90) 
    plt.xlabel("False Positive Rate") 
    plt.ylabel("True Positive Rate (Recall)") 
    plt.legend(loc='best') 
 
fpr, tpr, auc_thresholds = roc_curve(y_test, y_scores) 
print(auc(fpr, tpr)) # AUC of ROC 
plot_roc_curve(fpr, tpr, 'F beta optimized') 
 
 
import xgboost as xgb 
import numpy as np 
import pandas as pd 
 
from sklearn.preprocessing import LabelBinarizer 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold 
 
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, f1_score, fbeta_score 
 
import matplotlib.pyplot as plt 
plt.style.use("ggplot") 
 
df = pd.read_csv('journeys_test.csv') 
 
targets = df['Conversion'] 
 
df.drop(['Journey Start Date', 'Journey End Date', 'Events Combo', 'User-Journey'], axis=1, inplace=True) 
df = pd.get_dummies(df) 
 
x = df 
y = targets 
 
 
x_train, x_val, y_train, y_val = train_test_split(df, targets, stratify=targets) 
x_train.drop(['Conversion'], axis=1, inplace=True) 
x_val.drop(['Conversion'], axis=1, inplace=True) 
x.drop(['Conversion'], axis=1, inplace=True) 
 
import xgboost as xgb 
xgb_model = xgb.XGBClassifier(learning_rate=0.001, 
                            max_depth = 1,  
                            n_estimators = 100, 
                              scale_pos_weight=5) 
xgb_model.fit(x_train, y_train) 
 
xgb_predict=xgb_model.predict(x_val) 
xgb_predict 
 
from sklearn.metrics import classification_report,confusion_matrix 
print(confusion_matrix(y_val,xgb_predict)) 
print(classification_report(y_val,xgb_predict)) 
 
xgb_model = xgb.XGBClassifier(learning_rate=0.001, 
                            max_depth = 10,  
                            n_estimators = 100, 
                              scale_pos_weight=4) 
xgb_model.fit(x_train, y_train) 
 
xgb_predict=xgb_model.predict(x_val) 
 
print(confusion_matrix(y_val,xgb_predict)) 
print(classification_report(y_val,xgb_predict)) 
 
 
import xgboost as xgb 
import numpy as np 
import pandas as pd 
 
from sklearn.preprocessing import LabelBinarizer 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score 
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, f1_score, fbeta_score 
 
import matplotlib.pyplot as plt 
plt.style.use("ggplot") 
 
df = pd.read_csv('journeys_test.csv') 
 
targets = df['Conversion'] 
 
df.drop(['Journey Start Date', 'Journey End Date', 'Events Combo', 'User-Journey'], axis=1, inplace=True) 
df = pd.get_dummies(df) 
 
x = df 
y = targets 
 
 
x_train, x_test, y_train, y_val = train_test_split(df, targets, stratify=targets) 
x_train.drop(['Conversion'], axis=1, inplace=True) 
x_test.drop(['Conversion'], axis=1, inplace=True) 
x.drop(['Conversion'], axis=1, inplace=True) 
 
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
 
 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
 
df = pd.read_csv('journeys_test.csv') 
 
targets = df['Conversion'] 
 
df.drop(['Journey Start Date', 'Journey End Date', 'Events Combo', 'User-Journey'], axis=1, inplace=True) 
df = pd.get_dummies(df) 
 
x = df 
y = targets 
 
 
X_train, X_test, y_train, y_test = train_test_split(df, targets, stratify=targets) 
X_train.drop(['Conversion'], axis=1, inplace=True) 
X_test.drop(['Conversion'], axis=1, inplace=True) 
x.drop(['Conversion'], axis=1, inplace=True) 
 
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler() 
scaler.fit(X_train) 
 
X_train = scaler.transform(X_train) 
X_test = scaler.transform(X_test) 
 
from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors=5) 
classifier.fit(X_train, y_train) 
 
y_pred = classifier.predict(X_test) 
from sklearn.metrics import classification_report, confusion_matrix 
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred)) 
 
error = [] 
f_score = [] 
 
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
 
import pandas as pd 
import numpy as np 
from sklearn import preprocessing 
import matplotlib.pyplot as plt  
plt.rc("font", size=14) 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
import seaborn as sns 
sns.set(style="white") 
sns.set(style="whitegrid", color_codes=True) 
  
 
df = pd.read_csv('journeys_test.csv') 
 
targets = df['Conversion'] 
 
df.drop(['Journey Start Date', 'Journey End Date', 'Events Combo', 'User-Journey'], axis=1, inplace=True) 
data_final = pd.get_dummies(df) 
 
 
X = data_final.loc[:, data_final.columns != 'Conversion'] 
y = data_final.loc[:, data_final.columns == 'Conversion'] 
 
from imblearn.over_sampling import SMOTE 
 
os = SMOTE(random_state=0) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 
columns = X_train.columns 
 
os_data_X,os_data_y=os.fit_resample(X_train, y_train) 
os_data_X = pd.DataFrame(data=os_data_X,columns=columns ) 
os_data_y= pd.DataFrame(data=os_data_y,columns=['Conversion']) 
 
data_final_vars=data_final.columns.values.tolist() 
y=['Conversion'] 
X=[i for i in data_final_vars if i not in y] 
from sklearn.feature_selection import RFE 
from sklearn.linear_model import LogisticRegression 
logreg = LogisticRegression() 
rfe = RFE(logreg, step = 20) 
rfe = rfe.fit(os_data_X, os_data_y.values.ravel()) 
print(rfe.support_) 
print(rfe.ranking_) 
cols = [] 
for i in [i for i, X in enumerate(list(rfe.support_)) if X]: 
    cols.append(X[i]) 
cols 
 
X=os_data_X[cols] 
y=os_data_y['Conversion'] 
 
import statsmodels.api as sm 
logit_model=sm.Logit(y,X) 
result=logit_model.fit() 
print(result.summary2()) 
 
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 
logreg = LogisticRegression() 
logreg.fit(X_train, y_train) 
 
y_pred = logreg.predict(X_test) 
 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import roc_curve 
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test)) 
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1]) 
plt.figure() 
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc) 
plt.plot([0, 1], [0, 1],'r--') 
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05]) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('Receiver operating characteristic') 
plt.legend(loc="lower right") 
plt.savefig('Log_ROC') 
plt.show()