from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
import numpy as np
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt


def main():
    plt.style.use("ggplot")

    raw = pd.read_csv("./data/journeys.csv", index_col='id', parse_dates=True, na_values=['nan']) 

    df = raw[['age', 'language', 'journey', 'touchpoint','duration', 'conversion','email', 'facebook', 'house_ads', 'instagram', 'push']]

    categorical_vars = ['age','language'] 


    # creating one hot encoder object 
    label_encoder = preprocessing.LabelEncoder()
    for each in categorical_vars:
        df[each]= label_encoder.fit_transform(df[each]) 


    labels = df.conversion
    X = df.loc[:, df.columns != 'conversion']
    

    
    y = df.conversion

    accuracy = []
    error = []
    f_score = []

    # Calculating error for K values
    k_min = 50
    k_max = 200
    for i in range(k_min, k_max):
        X_train, X_test, y_train, y_test = train_test_split(df,labels, test_size=0.2)
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))
        accuracy.append(np.mean(pred_i == y_test))
        f_score.append(metrics.f1_score(y_test, list(pred_i), average='weighted'))
    metrics.f1_score(np.array(y_test), pred_i, average='weighted')
    plt.figure(figsize=(12, 6))
    # plt.plot(range(k_min, k_max), f_score, color='black', linestyle='dashed', marker='x',
    # markerfacecolor='green', markersize=10)
    plt.plot(range(k_min, k_max), f_score, color='black', linestyle='dashed')
    plt.plot(range(k_min, k_max), accuracy, color='green', linestyle='dashed')
    plt.title('F1 Score by K Value')
    plt.xlabel('K Value')
    plt.ylabel('F1 Score')
    plt.savefig('knn.jpg')



    # reference https://machinelearningknowledge.ai/knn-classifier-in-sklearn-using-gridsearchcv-with-example/
    # defining parameter range
    knn = KNeighborsClassifier()
    k_range = list(range(k_min, k_max))
    param_grid = dict(n_neighbors=k_range)
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)
    
    # fitting the model for grid search
    grid_search=grid.fit(X_train, y_train)
    print(grid_search.best_params_)
    accuracy = grid_search.best_score_ *100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )


    X_train, X_test, y_train, y_test = train_test_split(df,labels, test_size=0.2, random_state=123)

    classifier = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))

    fig = plt.figure()
    plt.matshow(confusion_matrix(y_test, y_pred))
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    fig.savefig('confusion_matrix.jpg')

    print(classification_report(y_test, y_pred))




if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    main()  